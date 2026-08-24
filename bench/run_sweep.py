#!/usr/bin/env python3
"""Sweep vLLM TP/PP/DP/MTP/DSpark/KV-dtype configs, bench each, resume from JSONL."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from overlay_args import (
    DSPARK_TOKENS,
    ROOT,
    canonical_config_id,
    config_id,
    get_flag,
    overlay,
    parse_args_file,
    pair_args,
)
from bench_client import CCU_LEVELS, CONTEXTS, TRIALS, bench_config, progress

# PP=4 is included; drop with --exclude-pp 4 if the dedicated smoke fails.
# tp * pp * dp == 4 (four GPUs).
TP_PP_DP = ((2, 2, 1), (4, 1, 1), (1, 4, 1), (2, 1, 2), (1, 2, 2))
KV_DTYPES = ("fp8", "float16", "int8_per_token_head")
# 0 = spec off (no --speculative_config). MTP on = num_speculative_tokens 1..3.
# DSpark is on/off only (tokens fixed at DSPARK_TOKENS).
MTP_OFF = 0
MTP_TOKENS_ON = (1, 2, 3)
SPEC_OFF = "off"
SPEC_MTP = "mtp"
SPEC_DSPARK = "dspark"

STARTUP_TIMEOUT_S = 15 * 60
STOP_TIMEOUT_S = 90
FAIL_FAST_MARKERS = (
    "ValueError:",
    "Watchdog caught collective operation timeout",
    "Insufficient space in /dev/shm",
    "Engine core initialization failed",
)
HANG_MARKER = (
    "No available shared memory broadcast block found in 60 seconds"
)
# Worker/log silence during startup. InstantTensor/CUDA-graph hangs previously
# sat here for many minutes with PP0 idle and PP1 at 100% GPU.
# shm_broadcast every 60s is expected during torch.compile / CUDA-graph capture
# (vLLM's own message says so); do not treat a count of those lines as a hang.
# Stall is measured on log growth excluding those lines.
STARTUP_STALL_S = 240


@dataclass(frozen=True)
class SweepConfig:
    tp: int
    pp: int
    mtp: int
    dp: int = 1
    kv_dtype: str = "fp8"
    spec_method: str = ""

    def __post_init__(self) -> None:
        if not self.spec_method:
            object.__setattr__(self, "spec_method", SPEC_MTP if self.mtp > 0 else SPEC_OFF)

    @property
    def spec_tokens(self) -> int:
        if self.spec_method == SPEC_DSPARK:
            return DSPARK_TOKENS
        return self.mtp

    @property
    def id(self) -> str:
        return config_id(
            self.tp,
            self.pp,
            self.mtp,
            dp=self.dp,
            kv_dtype=self.kv_dtype,
            spec_method=self.spec_method,
        )


def all_configs(*, exclude_pp: set[int] | None = None) -> list[SweepConfig]:
    """kv (fp8 first) → layout → spec off, MTP 1..3, then DSpark (MTP fail does not skip DSpark)."""
    skip_pp = exclude_pp or set()
    configs: list[SweepConfig] = []
    for kv_dtype in KV_DTYPES:
        for tp, pp, dp in TP_PP_DP:
            if pp in skip_pp:
                continue
            configs.append(
                SweepConfig(tp, pp, MTP_OFF, dp=dp, kv_dtype=kv_dtype, spec_method=SPEC_OFF)
            )
            for tokens in MTP_TOKENS_ON:
                configs.append(
                    SweepConfig(
                        tp, pp, tokens, dp=dp, kv_dtype=kv_dtype, spec_method=SPEC_MTP
                    )
                )
            configs.append(
                SweepConfig(
                    tp, pp, MTP_OFF, dp=dp, kv_dtype=kv_dtype, spec_method=SPEC_DSPARK
                )
            )
    return configs


def cfg_fields(cfg: SweepConfig) -> dict[str, Any]:
    return {
        "tp": cfg.tp,
        "pp": cfg.pp,
        "dp": cfg.dp,
        "mtp": cfg.mtp,
        "kv_dtype": cfg.kv_dtype,
        "spec_method": cfg.spec_method,
        "spec_tokens": cfg.spec_tokens,
    }


def row_spec_method(row: dict[str, Any]) -> str:
    spec = str(row.get("spec_method") or "")
    if spec:
        return spec
    try:
        return SPEC_MTP if int(row.get("mtp") or 0) > 0 else SPEC_OFF
    except (TypeError, ValueError):
        return SPEC_OFF


def server_meta(base_args: Path) -> dict[str, str]:
    pairs = pair_args(parse_args_file(base_args))
    port = get_flag(pairs, "--port", default="8000") or "8000"
    host = get_flag(pairs, "--host", default="127.0.0.1") or "127.0.0.1"
    bind = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    model = get_flag(pairs, "--served-model-name", default="kCode") or "kCode"
    api_key = get_flag(pairs, "--api-key", default="") or ""
    return {
        "port": port,
        "base_url": f"http://{bind}:{port}/v1",
        "model": model,
        "api_key": api_key,
    }


def pids_on_port(port: int) -> list[int]:
    pids: set[int] = set()
    try:
        out = subprocess.check_output(
            ["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.split():
            if line.strip().isdigit():
                pids.add(int(line.strip()))
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    if pids:
        return sorted(pids)
    try:
        out = subprocess.check_output(
            ["fuser", f"{port}/tcp"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        for tok in out.replace(":", " ").split():
            if tok.isdigit():
                pids.add(int(tok))
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return sorted(pids)


def kill_vllm_orphans() -> None:
    """EngineCore/Worker ranks often survive after the API process on --port dies."""
    subprocess.run(
        ["pkill", "-9", "-f", r"VLLM::"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_port(port: int, timeout_s: float = STOP_TIMEOUT_S) -> None:
    pids = pids_on_port(port)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not pids_on_port(port):
            break
        time.sleep(1)
    for pid in pids_on_port(port):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
    kill_vllm_orphans()
    time.sleep(2)
    clear_dev_shm()


def health_ok(base_url: str, api_key: str) -> bool:
    url = base_url.rstrip("/") + "/models"
    req = Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urlopen(req, timeout=5) as resp:
            return 200 <= resp.status < 300
    except (HTTPError, URLError, TimeoutError, OSError):
        return False


def tail_text(path: Path, n: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def clear_dev_shm() -> None:
    """Drop leftover vLLM SHM (KV offload mmap, psm_*, sem.*) before serve."""
    shm = Path("/dev/shm")
    if not shm.is_dir():
        return
    for entry in shm.iterdir():
        try:
            if entry.is_dir() and not entry.is_symlink():
                subprocess.run(["rm", "-rf", str(entry)], check=False)
            else:
                entry.unlink(missing_ok=True)
        except OSError:
            pass


def start_vllm(args_file: Path, log_path: Path) -> subprocess.Popen[bytes]:
    clear_dev_shm()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "wb")  # noqa: SIM115 — kept for process lifetime
    env = os.environ.copy()
    env["VLLM_ARGS_FILE"] = str(args_file)
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        ["bash", str(ROOT / "run_vllm.sh")],
        cwd=str(ROOT),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def log_fatal(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    for marker in FAIL_FAST_MARKERS:
        if marker in text:
            return marker
    return None


def non_hang_log_len(path: Path) -> int:
    """Bytes of recent log excluding compile-time shm_broadcast warnings."""
    if not path.exists():
        return 0
    text = path.read_bytes()[-400_000:].decode("utf-8", errors="replace")
    kept = "\n".join(line for line in text.splitlines() if HANG_MARKER not in line)
    return len(kept)


def vllm_workers_busy(min_cpu: float = 8.0) -> bool:
    """True if EngineCore/Worker ranks are using CPU (load/compile with no log)."""
    try:
        out = subprocess.check_output(["ps", "-eo", "pcpu,cmd"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    for line in out.splitlines():
        if "VLLM::" not in line:
            continue
        try:
            cpu = float(line.split(None, 1)[0])
        except ValueError:
            continue
        if cpu >= min_cpu:
            return True
    return False


STARTUP_STAGES = (
    ("Uvicorn running", "uvicorn listening"),
    ("Application startup complete", "API ready"),
    ("Multi-modal warmup completed", "mm warmup done"),
    ("init engine (profile", "engine init done"),
    ("Graph capturing finished", "CUDA graphs done"),
    ("Capturing CUDA graphs", "CUDA graph capture"),
    ("torch.compile took", "torch.compile done"),
    ("Dynamo bytecode transform time", "torch.compile"),
    ("Using cache directory", "torch.compile"),
    ("Loading weights took", "weights loaded"),
    ("Starting to load model", "loading weights"),
    ("Initializing a V1 LLM engine", "engine core"),
    ("Resolved architecture", "config"),
)


def startup_stage(log_path: Path) -> str:
    if not log_path.exists():
        return "waiting for log"
    text = log_path.read_bytes()[-200_000:].decode("utf-8", errors="replace")
    for needle, name in STARTUP_STAGES:
        if needle in text:
            return name
    return "starting"


def wait_healthy(
    proc: subprocess.Popen[bytes],
    base_url: str,
    api_key: str,
    log_path: Path,
    timeout_s: float = STARTUP_TIMEOUT_S,
    label: str = "",
) -> None:
    deadline = time.time() + timeout_s
    started = time.time()
    last_progress = time.time()
    last_fp = 0
    last_stage = ""
    last_report = 0.0
    prefix = f"{label} " if label else ""
    progress(f"{prefix}starting vLLM (health timeout {timeout_s:.0f}s)")
    while time.time() < deadline:
        code = proc.poll()
        if code is not None:
            raise RuntimeError(
                f"vLLM exited with {code} before becoming healthy.\n{tail_text(log_path)}"
            )
        fatal = log_fatal(log_path)
        if fatal:
            raise RuntimeError(f"vLLM fail-fast: {fatal}\n{tail_text(log_path)}")
        if log_path.exists():
            fp = non_hang_log_len(log_path)
            if fp != last_fp:
                last_progress = time.time()
                last_fp = fp
            if time.time() - last_progress > STARTUP_STALL_S:
                if vllm_workers_busy():
                    last_progress = time.time()
                else:
                    raise RuntimeError(
                        f"vLLM hang: log stalled {STARTUP_STALL_S:.0f}s during startup "
                        f"(workers idle).\n{tail_text(log_path)}"
                    )
        stage = startup_stage(log_path)
        now = time.time()
        if stage != last_stage or now - last_report >= 15:
            elapsed = now - started
            progress(f"{prefix}startup {elapsed:.0f}s [{stage}]")
            last_stage = stage
            last_report = now
        if health_ok(base_url, api_key):
            elapsed = time.time() - started
            progress(f"{prefix}healthy after {elapsed:.0f}s [{stage}]")
            return
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become healthy within {timeout_s}s.\n{tail_text(log_path)}")


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def _row_ids(cid: str) -> tuple[str, str]:
    canon = canonical_config_id(cid)
    return cid, canon


def load_done_cells(path: Path) -> set[tuple[str, int, int]]:
    done: set[tuple[str, int, int]] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") == "cell" and row.get("config_id"):
                for cid in _row_ids(row["config_id"]):
                    done.add((cid, int(row["context"]), int(row["ccu"])))
            if row.get("type") in {"start_error", "config_done"} and row.get("config_id"):
                # start_error: skip the whole config. config_done: already finished.
                if row["type"] == "start_error":
                    for ctx in CONTEXTS:
                        for ccu in CCU_LEVELS:
                            for cid in _row_ids(row["config_id"]):
                                done.add((cid, ctx, ccu))
    return done


def failed_configs(path: Path) -> set[str]:
    failed: set[str] = set()
    if not path.exists():
        return failed
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") == "start_error" and row.get("config_id"):
                for cid in _row_ids(row["config_id"]):
                    failed.add(cid)
    return failed


FamilyKey = tuple[int, int, int, str]


def load_family_skips(
    path: Path,
) -> tuple[set[FamilyKey], dict[FamilyKey, int]]:
    """Rebuild MTP-off / MTP-on skip state from prior start_error rows."""
    dead_layout: set[FamilyKey] = set()
    skip_mtp_from: dict[FamilyKey, int] = {}
    if not path.exists():
        return dead_layout, skip_mtp_from
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") != "start_error":
                continue
            try:
                key = (
                    int(row["tp"]),
                    int(row["pp"]),
                    int(row.get("dp") or 1),
                    str(row.get("kv_dtype") or "fp8"),
                )
                mtp = int(row["mtp"])
            except (KeyError, TypeError, ValueError):
                continue
            if row_spec_method(row) == SPEC_DSPARK:
                continue
            if mtp == MTP_OFF:
                dead_layout.add(key)
            else:
                nxt = mtp + 1
                prev = skip_mtp_from.get(key)
                skip_mtp_from[key] = nxt if prev is None else min(prev, nxt)
    return dead_layout, skip_mtp_from


def flatten_stat(prefix: str, stat: dict[str, float] | None) -> dict[str, float | str]:
    if not stat:
        return {
            f"{prefix}_median": "",
            f"{prefix}_avg": "",
            f"{prefix}_min": "",
            f"{prefix}_max": "",
        }
    return {
        f"{prefix}_median": stat["median"],
        f"{prefix}_avg": stat["avg"],
        f"{prefix}_min": stat["min"],
        f"{prefix}_max": stat["max"],
    }


def write_summary(jsonl_path: Path, out_dir: Path) -> None:
    cells: list[dict[str, Any]] = []
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("type") == "cell":
                    cells.append(row)

    csv_path = out_dir / "summary.csv"
    fieldnames = [
        "config_id",
        "tp",
        "pp",
        "dp",
        "mtp",
        "spec_method",
        "spec_tokens",
        "kv_dtype",
        "backend",
        "context",
        "ccu",
        "n_ok",
        "n_err",
        "ttft_s_median",
        "ttft_s_avg",
        "ttft_s_min",
        "ttft_s_max",
        "prefill_tps_median",
        "prefill_tps_avg",
        "prefill_tps_min",
        "prefill_tps_max",
        "decode_tps_median",
        "decode_tps_avg",
        "decode_tps_min",
        "decode_tps_max",
        "calibrated_prompt_tokens",
        "cached_tokens_median",
        "cached_tokens_avg",
        "cached_tokens_min",
        "cached_tokens_max",
        "errors",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in cells:
            agg = row.get("aggregate") or {}
            out = {
                "config_id": row.get("config_id", ""),
                "tp": row.get("tp", ""),
                "pp": row.get("pp", ""),
                "dp": row.get("dp", 1),
                "mtp": row.get("mtp", ""),
                "spec_method": row_spec_method(row),
                "spec_tokens": row.get("spec_tokens", row.get("mtp", "")),
                "kv_dtype": row.get("kv_dtype", "fp8"),
                "backend": row.get("backend", ""),
                "context": row.get("context", ""),
                "ccu": row.get("ccu", ""),
                "n_ok": agg.get("n_ok", ""),
                "n_err": agg.get("n_err", ""),
                "calibrated_prompt_tokens": row.get("calibrated_prompt_tokens", ""),
                "errors": " | ".join(agg.get("errors") or []),
            }
            out.update(flatten_stat("ttft_s", agg.get("ttft_s")))
            out.update(flatten_stat("prefill_tps", agg.get("prefill_tps")))
            out.update(flatten_stat("decode_tps", agg.get("decode_tps")))
            out.update(flatten_stat("cached_tokens", agg.get("cached_tokens")))
            writer.writerow(out)

    # Rank configs by median decode tok/s at the sweep CCU (mean of per-context medians).
    rank_ccu = CCU_LEVELS[0]
    by_cfg: dict[str, dict[str, Any]] = {}
    for row in cells:
        if int(row.get("ccu") or 0) != rank_ccu:
            continue
        cid = row["config_id"]
        agg = row.get("aggregate") or {}
        dec = (agg.get("decode_tps") or {}).get("median")
        ttft = (agg.get("ttft_s") or {}).get("median")
        rec = by_cfg.setdefault(
            cid,
            {
                "config_id": cid,
                "tp": row.get("tp"),
                "pp": row.get("pp"),
                "dp": row.get("dp", 1),
                "mtp": row.get("mtp"),
                "spec_method": row_spec_method(row),
                "spec_tokens": row.get("spec_tokens", row.get("mtp")),
                "kv_dtype": row.get("kv_dtype", "fp8"),
                "backend": row.get("backend"),
                "decode_medians": [],
                "ttft_medians": [],
            },
        )
        if dec is not None:
            rec["decode_medians"].append(dec)
        if ttft is not None:
            rec["ttft_medians"].append(ttft)

    ranked: list[dict[str, Any]] = []
    for rec in by_cfg.values():
        decs = rec.pop("decode_medians")
        ttfts = rec.pop("ttft_medians")
        rec[f"mean_median_decode_tps_ccu{rank_ccu}"] = sum(decs) / len(decs) if decs else ""
        rec[f"mean_median_ttft_s_ccu{rank_ccu}"] = sum(ttfts) / len(ttfts) if ttfts else ""
        rec["n_contexts"] = max(len(decs), len(ttfts))
        ranked.append(rec)
    ranked.sort(
        key=lambda r: (
            -(r[f"mean_median_decode_tps_ccu{rank_ccu}"] or 0),
            r[f"mean_median_ttft_s_ccu{rank_ccu}"] or 1e9,
        )
    )
    rank_path = out_dir / f"ranking_ccu{rank_ccu}.csv"
    rank_fields = [
        "rank",
        "config_id",
        "tp",
        "pp",
        "dp",
        "mtp",
        "spec_method",
        "spec_tokens",
        "kv_dtype",
        "backend",
        f"mean_median_decode_tps_ccu{rank_ccu}",
        f"mean_median_ttft_s_ccu{rank_ccu}",
        "n_contexts",
    ]
    with rank_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rank_fields)
        writer.writeheader()
        for i, rec in enumerate(ranked, start=1):
            writer.writerow({"rank": i, **rec})
    print(f"wrote {csv_path} and {rank_path}", file=sys.stderr)


async def run_one_config(
    cfg: SweepConfig,
    *,
    meta: dict[str, str],
    out_dir: Path,
    jsonl_path: Path,
    done: set[tuple[str, int, int]],
    contexts: list[int],
    ccu_levels: list[int],
    trials: int,
    skip_start: bool,
) -> str:
    """Returns 'ok', 'start_error', or 'already_done'."""
    port = int(meta["port"])
    needed = [
        (ctx, ccu)
        for ctx in contexts
        for ccu in ccu_levels
        if (cfg.id, ctx, ccu) not in done
    ]
    if not needed:
        print(f"skip {cfg.id} (already done)", file=sys.stderr)
        return "already_done"

    args_file = out_dir / "args" / f"{cfg.id}.sh"
    log_path = out_dir / "logs" / f"{cfg.id}.log"
    overlay(
        ROOT / "vllm_args.sh",
        args_file,
        tp=cfg.tp,
        pp=cfg.pp,
        mtp_tokens=cfg.mtp,
        dp=cfg.dp,
        kv_dtype=cfg.kv_dtype,
        spec_method=cfg.spec_method,
        spec_tokens=cfg.spec_tokens,
    )
    print(f"==> {cfg.id}", file=sys.stderr, flush=True)

    proc: subprocess.Popen[bytes] | None = None
    if not skip_start:
        stop_port(port)
        proc = start_vllm(args_file, log_path)
        try:
            wait_healthy(
                proc, meta["base_url"], meta["api_key"], log_path, label=cfg.id
            )
        except Exception as exc:  # noqa: BLE001
            progress(f"{cfg.id} start_error: {exc}".split("\n", 1)[0])
            append_jsonl(
                jsonl_path,
                {
                    "type": "start_error",
                    "config_id": cfg.id,
                    **cfg_fields(cfg),
                    "error": str(exc),
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
            )
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
            stop_port(port)
            write_summary(jsonl_path, out_dir)
            return "start_error"

    remaining_contexts = sorted({ctx for ctx, _ in needed})
    n_cells = len(remaining_contexts) * len(ccu_levels)
    progress(
        f"{cfg.id} bench: {len(remaining_contexts)} contexts × "
        f"{len(ccu_levels)} CCU × {trials} trials ({n_cells} cells)"
    )
    seed = int(hashlib.md5(cfg.id.encode()).hexdigest()[:8], 16)

    def on_row(row: dict[str, Any]) -> None:
        row["config_id"] = cfg.id
        row.update(cfg_fields(cfg))
        row["ts"] = datetime.now(timezone.utc).isoformat()
        append_jsonl(jsonl_path, row)
        write_summary(jsonl_path, out_dir)

    try:
        await bench_config(
            base_url=meta["base_url"],
            api_key=meta["api_key"],
            model=meta["model"],
            contexts=remaining_contexts,
            ccu_levels=ccu_levels,
            trials=trials,
            seed=seed,
            label=cfg.id,
            on_row=on_row,
        )
        append_jsonl(
            jsonl_path,
            {
                "type": "config_done",
                "config_id": cfg.id,
                "ts": datetime.now(timezone.utc).isoformat(),
            },
        )
        progress(f"{cfg.id} config_done")
    finally:
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
        if not skip_start:
            stop_port(port)
        write_summary(jsonl_path, out_dir)
    return "ok"


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--contexts", default=",".join(str(c) for c in CONTEXTS))
    p.add_argument("--ccu", default=",".join(str(c) for c in CCU_LEVELS))
    p.add_argument("--trials", type=int, default=TRIALS)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="One 10k / CCU=1 trial against the already-running server (no restart).",
    )
    p.add_argument(
        "--skip-start",
        action="store_true",
        help="Do not restart vLLM; bench the currently running server.",
    )
    p.add_argument("--only", default="", help="Comma-separated config ids to run.")
    p.add_argument(
        "--exclude-pp",
        default="",
        help="Comma-separated pipeline-parallel sizes to drop (e.g. 4).",
    )
    return p.parse_args(argv)


def family_key(cfg: SweepConfig) -> FamilyKey:
    return (cfg.tp, cfg.pp, cfg.dp, cfg.kv_dtype)


async def async_main(ns: argparse.Namespace) -> int:
    meta = server_meta(ROOT / "vllm_args.sh")
    if not meta["api_key"]:
        print("No --api-key in vllm_args.sh", file=sys.stderr)
        return 1

    if ns.smoke:
        ns.contexts = "10000"
        ns.ccu = "1"
        ns.trials = 1
        ns.skip_start = True
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = ns.out_dir or (ROOT / "bench" / "results" / f"smoke-{stamp}")
    else:
        out_dir = ns.out_dir or (ROOT / "bench" / "results" / "sweep")

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"
    done = load_done_cells(jsonl_path)
    contexts = [int(x) for x in ns.contexts.split(",") if x.strip()]
    ccu_levels = [int(x) for x in ns.ccu.split(",") if x.strip()]
    exclude_pp = {int(x) for x in ns.exclude_pp.split(",") if x.strip()}

    configs = all_configs(exclude_pp=exclude_pp)
    if ns.only:
        want = {x.strip() for x in ns.only.split(",") if x.strip()}
        want |= {canonical_config_id(x) for x in want}
        configs = [c for c in configs if c.id in want]
    if ns.smoke:
        configs = [SweepConfig(2, 2, MTP_OFF)]

    print(f"results -> {out_dir}", file=sys.stderr)
    print(
        f"{len(configs)} configs, contexts={contexts}, ccu={ccu_levels}, "
        f"trials={ns.trials}, exclude_pp={sorted(exclude_pp) or 'none'}",
        file=sys.stderr,
    )

    dead_layout, skip_mtp_from = load_family_skips(jsonl_path)

    for cfg in configs:
        key = family_key(cfg)
        if cfg.id in failed_configs(jsonl_path):
            print(f"skip {cfg.id} (previous start_error)", file=sys.stderr)
            continue
        if key in dead_layout:
            append_jsonl(
                jsonl_path,
                {
                    "type": "skipped",
                    "config_id": cfg.id,
                    **cfg_fields(cfg),
                    "reason": "mtp_off_failed",
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(f"skip {cfg.id} (MTP off failed for this layout/kv)", file=sys.stderr)
            continue
        skip_from = skip_mtp_from.get(key)
        if (
            cfg.spec_method == SPEC_MTP
            and cfg.mtp > MTP_OFF
            and skip_from is not None
            and cfg.mtp >= skip_from
        ):
            append_jsonl(
                jsonl_path,
                {
                    "type": "skipped",
                    "config_id": cfg.id,
                    **cfg_fields(cfg),
                    "reason": "mtp_family_failed",
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(
                f"skip {cfg.id} (MTP tokens>={skip_from} skipped after earlier MTP-on failure)",
                file=sys.stderr,
            )
            continue

        status = await run_one_config(
            cfg,
            meta=meta,
            out_dir=out_dir,
            jsonl_path=jsonl_path,
            done=done,
            contexts=contexts,
            ccu_levels=ccu_levels,
            trials=ns.trials,
            skip_start=ns.skip_start,
        )
        done = load_done_cells(jsonl_path)
        if status == "start_error":
            if cfg.spec_method == SPEC_DSPARK:
                pass
            elif cfg.mtp == MTP_OFF:
                dead_layout.add(key)
            else:
                skip_mtp_from[key] = cfg.mtp + 1

    write_summary(jsonl_path, out_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    ns = parse_cli(argv)
    return asyncio.run(async_main(ns))


if __name__ == "__main__":
    sys.exit(main())
