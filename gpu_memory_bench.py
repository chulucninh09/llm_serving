#!/usr/bin/env python3
"""
GPU P2P Bandwidth Benchmark
Tests all transfer paths: P2P DMA, SHM, and CPU-bounce
Compares NCCL transport modes for SYS topology
"""

import os, time, subprocess
import torch
import torch.distributed as dist

SEP = "=" * 60

# ── helpers ──────────────────────────────────────────────────

def mb(nbytes): return nbytes / 1e6
def gb(nbytes): return nbytes / 1e9

def sync_all(*devices):
    for d in devices:
        torch.cuda.synchronize(d)

def bw(nbytes, elapsed):
    return nbytes / elapsed / 1e9  # GB/s

def run(label, fn, warmup=3, runs=10):
    for _ in range(warmup):
        fn()
    sync_all(0, 1)

    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        sync_all(0, 1)
        times.append(time.perf_counter() - t)

    times.sort()
    median = times[len(times)//2]
    best   = times[0]
    worst  = times[-1]
    return median, best, worst

def print_row(label, nbytes, median, best, worst):
    print(f"  {label:<35} {bw(nbytes,median):>7.2f} GB/s"
          f"  (best {bw(nbytes,best):.2f}  worst {bw(nbytes,worst):.2f})")

# ── section 1: CUDA direct transfers ─────────────────────────

def bench_cuda_transfers():
    print(f"\n{SEP}")
    print("SECTION 1: CUDA Direct Transfer Bandwidth")
    print(f"{SEP}")

    sizes_mb = [64, 256, 512, 1024, 2048]

    for smb in sizes_mb:
        n      = smb * 1024 * 1024 // 4          # float32 elements
        nbytes = n * 4

        src0 = torch.randn(n, device="cuda:0")
        src1 = torch.randn(n, device="cuda:1")

        print(f"\n  Buffer size: {smb} MB")

        # P2P: 0 → 1
        m, b, w = run("P2P  GPU0 → GPU1", lambda: src0.to("cuda:1"))
        print_row("P2P  GPU0 → GPU1", nbytes, m, b, w)

        # P2P: 1 → 0
        m, b, w = run("P2P  GPU1 → GPU0", lambda: src1.to("cuda:0"))
        print_row("P2P  GPU1 → GPU0", nbytes, m, b, w)

        # CPU bounce: 0 → CPU → 1
        def cpu_bounce():
            cpu = src0.cpu()
            cpu.to("cuda:1")
        m, b, w = run("CPU-bounce  GPU0 → CPU → GPU1", cpu_bounce)
        print_row("CPU-bounce  GPU0 → CPU → GPU1", nbytes, m, b, w)

        del src0, src1
        torch.cuda.empty_cache()

# ── section 2: Pinned memory path ────────────────────────────

def bench_pinned():
    print(f"\n{SEP}")
    print("SECTION 2: Pinned (page-locked) Memory Path")
    print(SEP)
    print("  Uses pinned host buffer as staging — avoids pageable bounce")

    sizes_mb = [256, 1024]
    for smb in sizes_mb:
        n      = smb * 1024 * 1024 // 4
        nbytes = n * 4

        src  = torch.randn(n, device="cuda:0")
        pin  = torch.empty(n, pin_memory=True)
        dst  = torch.empty(n, device="cuda:1")

        def pinned_path():
            pin.copy_(src)         # GPU0 → pinned host
            dst.copy_(pin)         # pinned host → GPU1

        m, b, w = run(f"GPU0→pinned→GPU1 ({smb}MB)", pinned_path)
        print_row(f"GPU0→pinned→GPU1 ({smb}MB)", nbytes, m, b, w)

        del src, pin, dst
        torch.cuda.empty_cache()

# ── section 3: NCCL transport comparison ─────────────────────

def bench_nccl_modes():
    print(f"\n{SEP}")
    print("SECTION 3: NCCL Transport Mode Comparison")
    print(SEP)
    print("  Spawns subprocesses — each tests a different NCCL config")
    print()

    modes = [
        ("P2P DMA (default)",       {"NCCL_P2P_LEVEL": "5",  "NCCL_SHM_DISABLE": "1"}),
        ("Shared Memory only",       {"NCCL_P2P_LEVEL": "0",  "NCCL_SHM_DISABLE": "0"}),
        ("P2P disabled + SHM off",   {"NCCL_P2P_LEVEL": "0",  "NCCL_SHM_DISABLE": "1"}),
        ("NET fallback",             {"NCCL_P2P_LEVEL": "0",  "NCCL_SHM_DISABLE": "1",
                                      "NCCL_NET_DISABLE": "0"}),
    ]

    nccl_script = """
import os, time, torch
import torch.distributed as dist

dist.init_process_group("nccl", rank=int(os.environ["RANK"]),
                        world_size=2,
                        init_method="tcp://127.0.0.1:29600")

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

n = 256 * 1024 * 1024 // 4   # 256 MB float32
nbytes = n * 4
x = torch.randn(n, device=f"cuda:{local_rank}")

# warmup
for _ in range(3):
    dist.all_reduce(x)
torch.cuda.synchronize()

runs = 8
times = []
for _ in range(runs):
    torch.cuda.synchronize()
    t = time.perf_counter()
    dist.all_reduce(x)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t)

if local_rank == 0:
    times.sort()
    med = times[len(times)//2]
    # allreduce moves 2*(n-1)/n * nbytes ≈ 2x for 2 GPUs
    effective = 2 * nbytes / med / 1e9
    print(f"{effective:.2f}")

dist.destroy_process_group()
"""

    import tempfile, sys
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(nccl_script)
        script_path = f.name

    for name, env_overrides in modes:
        env = os.environ.copy()
        env.update(env_overrides)
        # suppress NCCL noise unless it fails
        env["NCCL_DEBUG"] = "WARN"

        try:
            result = subprocess.run(
                [sys.executable, "-m", "torch.distributed.run",
                 "--nproc_per_node=2", "--nnodes=1",
                 "--rdzv_backend=c10d",
                 "--rdzv_endpoint=127.0.0.1:29601",
                 script_path],
                capture_output=True, text=True, timeout=60, env=env
            )
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
            bw_val = next((l for l in lines if l.replace('.','',1).isdigit()), None)
            if bw_val:
                print(f"  {name:<35} {float(bw_val):>7.2f} GB/s (allreduce bus BW)")
            else:
                print(f"  {name:<35} FAILED")
                if result.stderr:
                    print(f"    stderr: {result.stderr[-200:]}")
        except subprocess.TimeoutExpired:
            print(f"  {name:<35} TIMEOUT")
        except Exception as e:
            print(f"  {name:<35} ERROR: {e}")

    os.unlink(script_path)

# ── section 4: Topology summary ──────────────────────────────

def print_topology():
    print(f"\n{SEP}")
    print("SECTION 0: Topology")
    print(SEP)

    print(f"\n  GPU count : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU{i}      : {p.name}  ({p.total_memory//1024**3} GB)")

    print(f"\n  P2P capability matrix:")
    n = torch.cuda.device_count()
    for i in range(n):
        for j in range(n):
            if i != j:
                ok = torch.cuda.can_device_access_peer(i, j)
                print(f"    GPU{i} → GPU{j} : {'✓ P2P' if ok else '✗ no P2P'}")

    print()
    try:
        topo = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"], text=True, stderr=subprocess.DEVNULL
        )
        for line in topo.splitlines():
            print(f"  {line}")
    except Exception:
        pass

    print()
    try:
        p2p = subprocess.check_output(
            ["nvidia-smi", "topo", "-p2p", "r"], text=True, stderr=subprocess.DEVNULL
        )
        print("  P2P read matrix:")
        for line in p2p.splitlines():
            print(f"  {line}")
    except Exception:
        pass

# ── main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    assert torch.cuda.device_count() >= 2, "Need at least 2 GPUs"

    print(f"\n{'#'*60}")
    print("  GPU P2P BENCHMARK")
    print(f"{'#'*60}")

    print_topology()
    bench_cuda_transfers()
    bench_pinned()
    bench_nccl_modes()

    print(f"\n{SEP}")
    print("DONE")
    print(SEP)
    print()
    print("  Interpretation guide:")
    print("  ─────────────────────")
    print("  P2P DMA (real)    : 10–25 GB/s  — direct BAR1 DMA working")
    print("  P2P via QPI/SYS   :  3–8  GB/s  — cross-socket, still useful")
    print("  CPU bounce        :  8–20 GB/s  — depends on RAM BW")
    print("  Pinned staging    : 10–20 GB/s  — avoids pageable overhead")
    print("  NCCL allreduce BW :  5–15 GB/s  — effective bus bandwidth")
    print()
    print("  Your topology is SYS (cross-socket). If P2P DMA < 1 GB/s,")
    print("  the transfer is being caught in a slow IOMMU bounce path.")
    print("  In that case, NCCL SHM mode is likely your fastest option.")
    print()