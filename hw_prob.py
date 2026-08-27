#!/usr/bin/env python3
"""
hw_probe.py

Standalone hardware characterization -- answers "what is my hardware actually
capable of" using nothing but PyTorch + NCCL. No vLLM, no SGLang, no serving
config. This measures the same underlying properties that explained every
finding in your engine-level testing (memory bandwidth ceiling, compute
crossover point, P2P-vs-SHM bandwidth per GPU pair, TP2-vs-TP4 all-reduce
cost) -- just directly, on raw hardware, in minutes instead of hours of
server tuning.

------------------------------------------------------------------------------
INSTALL (only dependency is torch itself)
------------------------------------------------------------------------------
    pip install torch --break-system-packages   # if not already installed

------------------------------------------------------------------------------
USAGE -- one command, run it from wherever the file is, nothing else needed
------------------------------------------------------------------------------
    python3 hw_probe.py                    # runs everything: membw, flops, p2p, allreduce
    python3 hw_probe.py --test membw,flops # only these
    python3 hw_probe.py --test allreduce   # just the all-reduce comparison

No torchrun, no --nproc_per_node, no manual multi-process launch. The
all-reduce test spawns its own worker processes internally (one per visible
GPU) via torch.multiprocessing -- the script handles that itself.
------------------------------------------------------------------------------
"""

import argparse
import datetime
import os
import statistics
import subprocess
import time

import torch


def summarize(values: list) -> dict:
    """mean/median/stdev/min/max across repeated independent samples --
    stdev needs at least 2 samples, reports 0.0 for a single sample rather
    than crashing."""
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def fmt_stat(values: list, unit: str = "", precision: int = 2) -> str:
    """Compact 'mean±stdev [min-max]' string for embedding in table cells."""
    s = summarize(values)
    return (f"{s['mean']:.{precision}f}\u00b1{s['stdev']:.{precision}f}{unit} "
            f"[{s['min']:.{precision}f}-{s['max']:.{precision}f}]")


def timed(fn, warmup=5, iters=20):
    """Times fn() from BOTH sides -- GPU execution time (CUDA events) and
    CPU wall-clock time (issuing the calls, waiting on the result). This is
    the same distinction SGLang's own profiler reports as separate 'Sorted
    by CUDA Time' / 'Sorted by CPU Time' tables. A big gap between them
    means CPU-side dispatch/launch overhead -- not GPU compute -- is what's
    actually limiting you at that op size, which matters a lot at the small,
    decode-representative sizes used below."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    cpu_t0 = time.perf_counter()
    start_evt.record()
    for _ in range(iters):
        fn()
    end_evt.record()
    torch.cuda.synchronize()
    cpu_t1 = time.perf_counter()

    gpu_s = start_evt.elapsed_time(end_evt) / iters / 1000.0
    cpu_s = (cpu_t1 - cpu_t0) / iters
    return gpu_s, cpu_s


def print_topology():
    """GPU interconnect topology -- PIX/PXB/PHB/NODE/SYS matrix, NUMA and CPU
    affinity. Shells out to `nvidia-smi topo -m`, which ships with the driver
    itself -- no extra package needed. This is the same table that would
    otherwise require launching a full engine and reading its startup log to
    infer indirectly."""
    print("\n=== GPU topology ===")
    try:
        out = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            print(f"  [!] nvidia-smi topo -m failed: {out.stderr.strip()}")
            return
        print(out.stdout)
    except FileNotFoundError:
        print("  [!] nvidia-smi not found on PATH -- skipping topology.")
    except subprocess.TimeoutExpired:
        print("  [!] nvidia-smi topo -m timed out -- skipping topology.")


def print_pcie_and_power_config():
    """Per-GPU PCIe link (current vs max generation/width -- directly shows
    lane-width or link-speed downgrades without inferring them from lspci or
    benchmark numbers) and power config (current draw vs configured limit vs
    hardware max, plus current clocks). All via one nvidia-smi query, no
    extra dependency."""
    print("\n=== PCIe link + power config (per GPU) ===")
    fields = [
        "index", "name",
        "pcie.link.gen.current", "pcie.link.gen.max",
        "pcie.link.width.current", "pcie.link.width.max",
        "power.draw", "power.limit", "power.max_limit",
        "clocks.sm", "clocks.mem", "temperature.gpu",
    ]
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            print(f"  [!] nvidia-smi query failed: {out.stderr.strip()}")
            return
    except FileNotFoundError:
        print("  [!] nvidia-smi not found on PATH -- skipping PCIe/power config.")
        return
    except subprocess.TimeoutExpired:
        print("  [!] nvidia-smi query timed out -- skipping PCIe/power config.")
        return

    header = ("GPU", "name", "PCIe gen (cur/max)", "PCIe width (cur/max)",
              "power draw/limit/max (W)", "SM/mem clk (MHz)", "temp (C)")
    print(f"  {header[0]:<4} {header[1]:<22} {header[2]:<20} {header[3]:<22} "
          f"{header[4]:<26} {header[5]:<18} {header[6]}")

    for line in out.stdout.strip().splitlines():
        vals = [v.strip() for v in line.split(",")]
        if len(vals) != len(fields):
            print(f"  [!] unexpected nvidia-smi output line, skipping: {line}")
            continue
        idx, name, gen_cur, gen_max, w_cur, w_max, pdraw, plimit, pmax, sm, mem, temp = vals
        gen_flag = " <-- DOWNGRADED" if gen_cur != gen_max else ""
        width_flag = " <-- DOWNGRADED" if w_cur != w_max else ""
        print(f"  {idx:<4} {name:<22} "
              f"{gen_cur + '/' + gen_max + gen_flag:<20} "
              f"{('x' + w_cur + '/x' + w_max + width_flag):<22} "
              f"{(pdraw + '/' + plimit + '/' + pmax + 'W'):<26} "
              f"{(sm + '/' + mem):<18} {temp}")


def bench_memory_bandwidth(device_id: int, size_gb: float = 2.0, n_samples: int = 10):
    """Read+write bandwidth on a single GPU, repeated as N independent
    samples (each its own warmup+timed run, not just inner-loop iterations)
    to separate real hardware behavior from run-to-run system jitter.
    Compare the mean against the card's spec-sheet memory bandwidth (e.g.
    936 GB/s for RTX 3090) to see how much of theoretical peak you get."""
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    n = int(size_gb * (1024**3) / 4)  # float32 elements
    a = torch.empty(n, dtype=torch.float32, device=device)
    b = torch.empty(n, dtype=torch.float32, device=device)
    a.normal_()

    def op():
        b.copy_(a)

    gbps_samples, overhead_samples = [], []
    for _ in range(n_samples):
        gpu_s, cpu_s = timed(op)
        bytes_moved = n * 4 * 2  # read a + write b
        gbps_samples.append(bytes_moved / gpu_s / 1e9)
        overhead_samples.append(100 * max(0.0, cpu_s - gpu_s) / cpu_s)

    s = summarize(gbps_samples)
    print(f"  [GPU {device_id}] memory bandwidth: {fmt_stat(gbps_samples, ' GB/s', 1)}  "
          f"(n={n_samples}, mean dispatch overhead: {statistics.mean(overhead_samples):.1f}%)")
    del a, b
    torch.cuda.empty_cache()
    return s["mean"]


def bench_compute_flops(device_id: int, dtype=torch.bfloat16, matrix_dim: int = 8192, n_samples: int = 10):
    """Square-matmul FLOPS on a single GPU, repeated as N independent
    samples. Compare the mean against spec-sheet dense tensor-core TFLOPS
    (e.g. ~71 TFLOPS BF16 for RTX 3090)."""
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    a = torch.randn(matrix_dim, matrix_dim, dtype=dtype, device=device)
    b = torch.randn(matrix_dim, matrix_dim, dtype=dtype, device=device)

    def op():
        torch.matmul(a, b)

    flops = 2 * (matrix_dim ** 3)
    tflops_samples, overhead_samples = [], []
    for _ in range(n_samples):
        gpu_s, cpu_s = timed(op, warmup=10, iters=30)
        tflops_samples.append(flops / gpu_s / 1e12)
        overhead_samples.append(100 * max(0.0, cpu_s - gpu_s) / cpu_s)

    s = summarize(tflops_samples)
    print(f"  [GPU {device_id}] {dtype} compute: {fmt_stat(tflops_samples, ' TFLOPS', 1)}  "
          f"(n={n_samples}, mean dispatch overhead: {statistics.mean(overhead_samples):.1f}%)")
    del a, b
    torch.cuda.empty_cache()
    return s["mean"]


# Same small-to-large progression used by the all-reduce test (2KB up to
# 16MB, decode- through prefill-representative), plus the original 256MB
# point kept as a pure bandwidth-ceiling measurement at the top end.
P2P_SHM_SIZES_BYTES = [2 * 1024, 16 * 1024, 128 * 1024, 2 * 1024**2, 16 * 1024**2, 256 * 1024**2]


def _measure_p2p_gbps(src: int, dst: int, n: int, n_samples: int = 10) -> list:
    torch.cuda.set_device(src)
    a = torch.empty(n, dtype=torch.float32, device=f"cuda:{src}")
    a.normal_()
    torch.cuda.synchronize(src)
    torch.cuda.synchronize(dst)

    def p2p_op():
        a.to(f"cuda:{dst}", non_blocking=True)

    for _ in range(3):
        p2p_op()
    torch.cuda.synchronize(src)
    torch.cuda.synchronize(dst)

    samples = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        iters = 10
        for _ in range(iters):
            p2p_op()
        torch.cuda.synchronize(src)
        torch.cuda.synchronize(dst)
        elapsed = (time.perf_counter() - t0) / iters
        samples.append((n * 4) / elapsed / 1e9)
    del a
    torch.cuda.empty_cache()
    return samples


def _measure_shm_gbps(src: int, dst: int, n: int, n_samples: int = 10) -> list:
    torch.cuda.set_device(src)
    src_buf = torch.empty(n, dtype=torch.float32, device=f"cuda:{src}")
    src_buf.normal_()
    host_buf = torch.empty(n, dtype=torch.float32, pin_memory=True)
    torch.cuda.synchronize(src)
    torch.cuda.synchronize(dst)

    def shm_op():
        host_buf.copy_(src_buf, non_blocking=True)              # D2H
        torch.cuda.synchronize(src)
        host_buf.to(f"cuda:{dst}", non_blocking=True)            # H2D

    for _ in range(3):
        shm_op()
    torch.cuda.synchronize(src)
    torch.cuda.synchronize(dst)

    samples = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        iters = 10
        for _ in range(iters):
            shm_op()
        torch.cuda.synchronize(src)
        torch.cuda.synchronize(dst)
        elapsed = (time.perf_counter() - t0) / iters
        samples.append((n * 4) / elapsed / 1e9)
    del src_buf, host_buf
    torch.cuda.empty_cache()
    return samples


def bench_p2p_and_shm_all_pairs(sizes_bytes: list = None, n_samples: int = 10):
    """Direct GPU-to-GPU (P2P) vs staged-through-host-RAM (SHM) transfer
    bandwidth for every pair of visible GPUs, swept across message sizes
    from decode-representative (2KB) up to prefill-representative (256MB),
    each point repeated as N independent samples to separate real hardware
    behavior from run-to-run jitter.

    This is the actual choice NCCL makes at runtime -- P2P for topologies it
    trusts, SHM as the fallback (and, per NCCL's own cost model, sometimes
    preferred on purpose for cross-socket pairs, since inter-socket P2P is
    known to be handled poorly by the CPU). It's also a direct predictor of
    whether an engine's OWN P2P-only fast path would even be eligible to run
    at all -- e.g. vLLM's custom-all-reduce specifically requires
    world_size==2 or a fully-connected topology, precisely because it needs
    genuinely fast P2P between every pair in the group, not just a
    favorable NCCL cost-model score.

    Sweeping sizes matters because P2P's real advantage is usually framed
    around LATENCY at small messages (skipping a second dispatch/sync
    round-trip), which a single large-transfer measurement can't see -- a
    pair can lose on a 256MB bulk transfer and still win at the small sizes
    that actually matter for decode."""
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print("  Only 1 GPU visible -- nothing to test for P2P/SHM.")
        return

    sizes_bytes = sizes_bytes or P2P_SHM_SIZES_BYTES
    print(f"\n  P2P vs SHM bandwidth, all {n_gpus * (n_gpus - 1)} directed pairs, "
          f"{len(sizes_bytes)} sizes, n={n_samples} samples each:")
    print("  (SHM = staged through pinned host RAM: src GPU -> host -> dst GPU,")
    print("   the same mechanism NCCL falls back to when it doesn't use P2P --")
    print("   compare these numbers against the all-reduce results at the same")
    print("   sizes to see whether point-to-point bandwidth actually predicts")
    print("   collective performance for a given pair. 'faster' and the")
    print("   cross-size summary below are decided by MEAN, not a single")
    print("   sample, to avoid one noisy reading flipping the conclusion.)")

    summary = []  # (size_bytes, src, dst, mean_p2p_gbps, mean_shm_gbps)

    for size_bytes in sizes_bytes:
        n = size_bytes // 4  # float32 elements
        size_label = f"{size_bytes/1024:.0f}KB" if size_bytes < 1024**2 else f"{size_bytes/1024**2:.0f}MB"
        print(f"\n  -- {size_label} transfers --")
        print(f"  {'src->dst':<10} {'P2P GB/s (mean\u00b1std [range])':<32} "
              f"{'SHM GB/s (mean\u00b1std [range])':<32} {'faster':>8}")

        for src in range(n_gpus):
            for dst in range(n_gpus):
                if src == dst:
                    continue
                p2p_samples = _measure_p2p_gbps(src, dst, n, n_samples)
                shm_samples = _measure_shm_gbps(src, dst, n, n_samples)
                p2p_mean = statistics.mean(p2p_samples)
                shm_mean = statistics.mean(shm_samples)
                faster = "P2P" if p2p_mean >= shm_mean else "SHM"
                print(f"  {src}->{dst:<7} {fmt_stat(p2p_samples):<32} "
                      f"{fmt_stat(shm_samples):<32} {faster:>8}")
                summary.append((size_bytes, src, dst, p2p_mean, shm_mean))

    # Cross-size summary: for each pair, does the MEAN P2P win consistently,
    # or does it flip depending on message size? That flip is exactly the
    # signal that matters for interpreting engine-level behavior correctly.
    print(f"\n  {'='*60}")
    print("  SUMMARY -- does the P2P vs SHM winner (by mean) flip by size, per pair?")
    print(f"  {'='*60}")
    by_pair = {}
    for size_bytes, src, dst, p2p_mean, shm_mean in summary:
        by_pair.setdefault((src, dst), []).append(p2p_mean >= shm_mean)
    print(f"  {'pair':<10} {'result':<40}")
    for (src, dst), wins in sorted(by_pair.items()):
        if all(wins):
            result = "P2P wins at every size tested"
        elif not any(wins):
            result = "SHM wins at every size tested"
        else:
            result = "MIXED -- winner depends on message size, see detail above"
        print(f"  {src}->{dst:<7} {result}")


def _find_free_port() -> int:
    """Picks an unused local port for the NCCL rendezvous, so we don't
    collide with anything else running (e.g. a serving engine) on the box."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _allreduce_worker(rank: int, world_size: int, master_port: int, sizes: list, n_samples: int = 10):
    """Runs inside a spawned subprocess -- one per GPU. Sets up its own
    process-group env vars manually (this is exactly what torchrun does for
    you; doing it here means the script needs no external launcher at all).

    Tests EVERY possible group -- every combination of 2, 3, ... up to
    world_size GPUs, not just the full world and one hardcoded pair. With 4
    GPUs that's all C(4,2)+C(4,3)+C(4,4) = 11 groups, which is fast enough
    to run exhaustively even with repeated sampling. Each (group, size)
    point is measured N independent times -- not just N inner-loop
    iterations averaged into one number, but N full warmup+timed runs -- so
    a single noisy reading can't flip the ranking between groups, which is
    exactly what happened when this only ran once."""
    import itertools
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )

    summary = []  # (combo, group_size, mean_busbw_at_largest_size) -- rank 0 only

    def run_group(group, combo, group_size):
        label = f"ranks {combo}"
        if rank == 0:
            print(f"\n  All-reduce, {label} (group_size={group_size}), n={n_samples} samples:")
            print(f"  {'msg size':>10} {'GPU ms (mean\u00b1std)':>22} "
                  f"{'busbw GB/s (mean\u00b1std [range])':>34}")
        last_mean_busbw = None
        for n in sizes:
            t = torch.randn(n, dtype=torch.bfloat16, device="cuda")

            def op():
                dist.all_reduce(t, group=group)

            for _ in range(5):
                op()
            torch.cuda.synchronize()
            dist.barrier(group=group)

            gpu_ms_samples, busbw_samples = [], []
            for _ in range(n_samples):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                iters = 30
                start.record()
                for _ in range(iters):
                    op()
                end.record()
                torch.cuda.synchronize()
                gpu_ms = start.elapsed_time(end) / iters
                bytes_ = n * 2
                busbw_gbps = (bytes_ * 2 * (group_size - 1) / group_size) / (gpu_ms / 1000) / 1e9
                gpu_ms_samples.append(gpu_ms)
                busbw_samples.append(busbw_gbps)

            mean_gpu_ms = statistics.mean(gpu_ms_samples)
            std_gpu_ms = statistics.stdev(gpu_ms_samples) if len(gpu_ms_samples) > 1 else 0.0
            last_mean_busbw = statistics.mean(busbw_samples)
            if rank == 0:
                print(f"  {n*2/1024:>8.1f}KB {mean_gpu_ms:>10.4f}\u00b1{std_gpu_ms:<9.4f} "
                      f"{fmt_stat(busbw_samples):>34}")
        if rank == 0:
            summary.append((combo, group_size, last_mean_busbw))

    combos = []
    for size in range(2, world_size + 1):
        combos.extend(itertools.combinations(range(world_size), size))

    for combo in combos:
        # dist.new_group() is a collective call -- EVERY rank must call it,
        # in the same order, regardless of membership in this particular
        # group. Skipping this for non-member ranks is exactly what caused
        # the earlier deadlock.
        group = dist.new_group(ranks=list(combo))
        if rank in combo:
            run_group(group, combo, len(combo))
        # Unconditional, same call for every rank after every combo --
        # resynchronizes everyone on WORLD before moving to the next one.
        dist.barrier()

    if rank == 0 and summary:
        print(f"\n  {'='*60}")
        print(f"  SUMMARY -- all {len(combos)} groups, ranked by MEAN largest-message busbw")
        print(f"  (mean of {n_samples} samples each -- a single reading is not")
        print("   trusted to rank groups against each other)")
        print(f"  {'='*60}")
        print(f"  {'ranks':<16} {'size':>5} {'mean busbw @ largest msg (GB/s)':>32}")
        for combo, gsize, busbw in sorted(summary, key=lambda x: -x[2]):
            print(f"  {str(combo):<16} {gsize:>5} {busbw:>32.2f}")

    dist.destroy_process_group()


def bench_allreduce(n_gpus: int, n_samples: int = 10):
    """Spawns one worker process per GPU itself -- no torchrun, no manual
    launcher. Compares all-reduce cost across message sizes representative
    of decode (tiny, latency-bound) vs prefill (large, bandwidth-bound),
    across EVERY possible group of GPUs, each point repeated as N
    independent samples.

    Ctrl+C during this test terminates every spawned worker explicitly
    (SIGTERM, then SIGKILL for any that don't respond) instead of leaving
    orphaned processes holding GPU memory/NCCL state that would otherwise
    need manual `kill` afterward."""
    if n_gpus < 2:
        print("  Only 1 GPU visible -- nothing to test for all-reduce.")
        return

    sizes = [1024, 8192, 65536, 1048576, 8388608]
    master_port = _find_free_port()

    ctx = torch.multiprocessing.get_context("spawn")
    processes = []
    try:
        for rank in range(n_gpus):
            p = ctx.Process(
                target=_allreduce_worker,
                args=(rank, n_gpus, master_port, sizes, n_samples),
                daemon=False,
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n[!] Interrupted -- terminating all-reduce worker processes...")
        _cleanup_processes(processes)
        print("[!] Cleanup complete. Exiting.")
        raise
    except BaseException:
        # Any other failure (a real crash, not Ctrl+C) should still clean up
        # child processes rather than leaving them behind.
        _cleanup_processes(processes)
        raise


def _cleanup_processes(processes: list, term_timeout: float = 5.0, kill_timeout: float = 5.0):
    """Escalating shutdown: SIGTERM first (lets NCCL/CUDA attempt a clean
    exit), then SIGKILL for anything still alive after the timeout -- SIGKILL
    cannot be caught or delayed by a stuck child, so this guarantees no
    orphaned process survives even if one is wedged inside a blocking
    CUDA/NCCL call."""
    for p in processes:
        if p.is_alive():
            p.terminate()
    for p in processes:
        p.join(timeout=term_timeout)
    for p in processes:
        if p.is_alive():
            print(f"    process {p.pid} did not exit after SIGTERM, sending SIGKILL...")
            p.kill()
            p.join(timeout=kill_timeout)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", default="all",
                     help="comma-separated: info,membw,flops,p2p,allreduce,all (default: all). "
                          "'shm' is accepted as an alias for 'p2p' -- they run together "
                          "as one side-by-side comparison. 'info' prints topology/PCIe/power "
                          "config only, no timing benchmarks.")
    ap.add_argument("--samples", type=int, default=10,
                     help="independent repeated samples per measurement point (default: 10). "
                          "Each is a full warmup+timed run, not just inner-loop iterations -- "
                          "this is what separates real hardware behavior from run-to-run "
                          "system jitter, which mattered enough in practice to flip which "
                          "GPU group looked fastest on a single reading.")
    args = ap.parse_args()
    tests = set(args.test.split(",")) if args.test != "all" else {"info", "membw", "flops", "p2p", "allreduce"}
    if "shm" in tests:
        tests.add("p2p")

    if not torch.cuda.is_available():
        print("[!] No CUDA device visible. Nothing to probe.")
        return

    n_gpus = torch.cuda.device_count()
    print(f"Visible GPUs: {n_gpus}")
    for i in range(n_gpus):
        print(f"  cuda:{i} = {torch.cuda.get_device_name(i)}")

    if "info" in tests:
        print_topology()
        print_pcie_and_power_config()

    if "membw" in tests:
        print("\n=== Memory bandwidth (per GPU) ===")
        for i in range(n_gpus):
            bench_memory_bandwidth(i, n_samples=args.samples)

    if "flops" in tests:
        print("\n=== Compute FLOPS, bf16 (per GPU) ===")
        for i in range(n_gpus):
            bench_compute_flops(i, n_samples=args.samples)

    if "p2p" in tests:
        print("\n=== P2P vs SHM bandwidth (all GPU pairs) ===")
        bench_p2p_and_shm_all_pairs(n_samples=args.samples)

    if "allreduce" in tests:
        print("\n=== NCCL all-reduce ===")
        bench_allreduce(n_gpus, n_samples=args.samples)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted. Exiting.")
        raise SystemExit(130)  # standard shell convention for SIGINT exit