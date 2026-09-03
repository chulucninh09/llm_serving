#!/usr/bin/env python3
"""External LRU janitor for vLLM and SGLang filesystem KV offload pages.

Both engines write unbounded ``*.bin`` pages under a shared root (default
``/mnt/llm-data/kv-cache``):

* vLLM OffloadingConnector FS tier:
  ``<model>_<sha>/config.json`` and ``<model>_<sha>_rN/<hhh>/<hh>_gG/<hash>.bin``
* SGLang HiCache ``--hicache-storage-backend file``:
  flat ``sglang/{sha256}_{served_model}_{tp_rank}_{tp_size}.bin``
  (see ``SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR``)

This process polls filesystem usage via ``shutil.disk_usage`` (statvfs — the
same syscall as ``df``) and, when used bytes cross the high watermark, unlinks
oldest ``*.bin`` files until usage is at or below the target.

Deleting a completed ``.bin`` is safe for both engines: lookup is
existence-based, so a missing file is a cache miss and the page is recomputed.
In-flight SGLang temps look like ``{name}.bin.tmp.{pid}.{tid}.{uuid}`` and are
skipped. vLLM ``config.json`` and this process's lock file are never unlinked.

Note: SGLang stores millions of small files in one flat directory, so a full
``ls`` or the first compact scan of that dir can take tens of seconds.
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import os
import shutil
import signal
import sys
import time
from pathlib import Path

LOG = logging.getLogger("kv_cache_lru_cleaner")

GIB = 1024**3
# Directory inode size above this triggers a one-line slow-scan hint (SGLang
# flat HiCache dirs routinely exceed 100+ MiB of dentries alone).
HUGE_DIR_INODE_BYTES = 16 * 1024 * 1024
LOCK_NAME = ".kv-cache-cleaner.lock"
DEFAULT_CACHE_DIR = "/mnt/llm-data/kv-cache"
DEFAULT_INTERVAL_S = 30.0
DEFAULT_TRIGGER_GIB = 230.0
DEFAULT_TARGET_GIB = 180.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Poll filesystem usage and LRU-evict vLLM FS-tier and SGLang "
            "HiCache *.bin KV pages."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(DEFAULT_CACHE_DIR),
        help=(
            f"Shared KV offload root covering vLLM hashed dirs and "
            f"sglang/ HiCache (default: {DEFAULT_CACHE_DIR})"
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_S,
        help=f"Seconds between idle polls (default: {DEFAULT_INTERVAL_S:g})",
    )
    parser.add_argument(
        "--trigger-used-gb",
        type=float,
        default=DEFAULT_TRIGGER_GIB,
        help=f"Start compaction when filesystem used >= this many GiB "
        f"(default: {DEFAULT_TRIGGER_GIB:g})",
    )
    parser.add_argument(
        "--target-used-gb",
        type=float,
        default=DEFAULT_TARGET_GIB,
        help=f"Stop compaction when filesystem used <= this many GiB "
        f"(default: {DEFAULT_TARGET_GIB:g})",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one poll/compact cycle and exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logs (unscannable dirs, etc.)",
    )
    args = parser.parse_args(argv)
    if args.trigger_used_gb <= args.target_used_gb:
        parser.error("--trigger-used-gb must be greater than --target-used-gb")
    if args.interval <= 0:
        parser.error("--interval must be > 0")
    return args


def fs_used_bytes(path: Path) -> int:
    return shutil.disk_usage(path).used


def format_gib(n_bytes: int | float) -> str:
    return f"{n_bytes / GIB:.2f} GiB"


def acquire_lock(cache_dir: Path) -> int:
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / LOCK_NAME
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise SystemExit(f"another cleaner already holds {lock_path}") from None
    os.write(fd, f"{os.getpid()}\n".encode())
    os.fsync(fd)
    return fd


def _maybe_log_huge_dir(path: Path, st_size: int) -> None:
    if st_size < HUGE_DIR_INODE_BYTES:
        return
    LOG.info(
        "scanning large directory %s (inode size %s); "
        "flat HiCache trees can take tens of seconds to list",
        path,
        format_gib(st_size),
    )


def iter_bin_files(cache_dir: Path):
    """Yield (atime, mtime, size, path) for completed *.bin files under cache_dir.

    Skips the cleaner lock file, names ending in ``.tmp``, and SGLang in-flight
    writes whose names contain ``.tmp.`` (e.g. ``foo.bin.tmp.123.1.abcd``).
    """
    stack = [cache_dir]
    while stack:
        current = stack.pop()
        try:
            try:
                dir_st = current.stat()
                _maybe_log_huge_dir(current, dir_st.st_size)
            except OSError:
                pass
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        name = entry.name
                        if name == LOCK_NAME:
                            continue
                        # Completed pages end in .bin; temps may be *.tmp or
                        # *.bin.tmp.<pid>.<tid>.<uuid> (SGLang HiCache).
                        if ".tmp." in name or name.endswith(".tmp"):
                            continue
                        if not name.endswith(".bin"):
                            continue
                        st = entry.stat(follow_symlinks=False)
                        yield st.st_atime, st.st_mtime, st.st_size, entry.path
                    except OSError:
                        continue
        except OSError as exc:
            LOG.debug("skip unscannable dir %s: %s", current, exc)


def prune_empty_dirs(cache_dir: Path) -> int:
    removed = 0
    for dirpath, _dirnames, _filenames in os.walk(cache_dir, topdown=False):
        path = Path(dirpath)
        if path == cache_dir:
            continue
        try:
            os.rmdir(path)
            removed += 1
        except OSError:
            pass
    return removed


def compact(cache_dir: Path, target_bytes: int) -> tuple[int, int]:
    """LRU-unlink *.bin until filesystem used <= target_bytes.

    Returns (files_deleted, bytes_unlinked).
    """
    used = fs_used_bytes(cache_dir)
    if used <= target_bytes:
        return 0, 0

    files = sorted(iter_bin_files(cache_dir))
    if not files:
        LOG.warning(
            "used %s is above target %s but no *.bin files under %s",
            format_gib(used),
            format_gib(target_bytes),
            cache_dir,
        )
        return 0, 0

    LOG.info(
        "compacting %s: used %s -> target %s (%d .bin files)",
        cache_dir,
        format_gib(used),
        format_gib(target_bytes),
        len(files),
    )

    deleted = 0
    unlinked_bytes = 0
    batch_est = 0
    need = used - target_bytes

    for _atime, _mtime, size, path in files:
        try:
            os.unlink(path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            LOG.warning("failed to unlink %s: %s", path, exc)
            continue

        deleted += 1
        unlinked_bytes += size
        batch_est += size

        if batch_est < need:
            continue

        used = fs_used_bytes(cache_dir)
        if used <= target_bytes:
            break
        need = used - target_bytes
        batch_est = 0
    else:
        used = fs_used_bytes(cache_dir)

    pruned = prune_empty_dirs(cache_dir)
    used = fs_used_bytes(cache_dir)
    LOG.info(
        "compacted: unlinked %d files (%s); pruned %d empty dirs; used now %s",
        deleted,
        format_gib(unlinked_bytes),
        pruned,
        format_gib(used),
    )
    if used > target_bytes:
        LOG.warning(
            "target %s not reached (used %s); remaining usage is outside *.bin cache",
            format_gib(target_bytes),
            format_gib(used),
        )
    return deleted, unlinked_bytes


def poll_once(cache_dir: Path, trigger_bytes: int, target_bytes: int) -> None:
    usage = shutil.disk_usage(cache_dir)
    LOG.info(
        "poll used=%s free=%s total=%s trigger=%s target=%s",
        format_gib(usage.used),
        format_gib(usage.free),
        format_gib(usage.total),
        format_gib(trigger_bytes),
        format_gib(target_bytes),
    )
    if usage.used >= trigger_bytes:
        compact(cache_dir, target_bytes)


def run_loop(
    cache_dir: Path,
    interval: float,
    trigger_bytes: int,
    target_bytes: int,
    once: bool,
) -> None:
    stop = False

    def _handle_stop(signum: int, _frame) -> None:
        nonlocal stop
        stop = True
        LOG.info("received signal %d, exiting after this cycle", signum)

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    while not stop:
        try:
            poll_once(cache_dir, trigger_bytes, target_bytes)
        except FileNotFoundError:
            LOG.error("cache dir vanished: %s", cache_dir)
            if once:
                raise
        except OSError as exc:
            LOG.error("poll failed: %s", exc)
            if once:
                raise
        if once or stop:
            break
        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cache_dir = args.cache_dir.resolve()
    trigger_bytes = int(args.trigger_used_gb * GIB)
    target_bytes = int(args.target_used_gb * GIB)

    lock_fd = acquire_lock(cache_dir)
    try:
        used = fs_used_bytes(cache_dir)
        LOG.info(
            "watching %s every %.3gs; trigger=%s target=%s; used now %s "
            "(vLLM hashed dirs + sglang/ HiCache)",
            cache_dir,
            args.interval,
            format_gib(trigger_bytes),
            format_gib(target_bytes),
            format_gib(used),
        )
        run_loop(cache_dir, args.interval, trigger_bytes, target_bytes, args.once)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
