#!/usr/bin/env bash
# Patch vLLM scheduler for OffloadingConnector + prefix caching on hybrid (Mamba) models.
#
# Replaces entire function bodies after confirming the on-disk version matches
# the expected upstream snapshot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$SCRIPT_DIR/.venv}"
PATCH_DIR="$SCRIPT_DIR/patches/vllm_offload_scheduler"
SCHEDULER="$VENV/lib/python3.13/site-packages/vllm/v1/core/sched/scheduler.py"

if [[ ! -f "$SCHEDULER" ]]; then
  echo "scheduler.py not found at $SCHEDULER" >&2
  exit 1
fi

for payload in \
  __init__.upstream.py \
  __init__.patched.py \
  _mamba_block_aligned_split.upstream.py \
  _mamba_block_aligned_split.patched.py \
  schedule.upstream.py \
  schedule.patched.py
do
  if [[ ! -f "$PATCH_DIR/$payload" ]]; then
    echo "Missing patch payload: $PATCH_DIR/$payload" >&2
    exit 1
  fi
done

"$VENV/bin/python" <<PY
import sys
from pathlib import Path

sys.path.insert(0, "${SCRIPT_DIR}/scripts")
from vllm_function_patch import replace_whole_body

SCHEDULER = Path("${SCHEDULER}")
PATCH_DIR = Path("${PATCH_DIR}")
changed = False

for label, upstream_name, patched_name in (
    ("Scheduler.__init__", "__init__.upstream.py", "__init__.patched.py"),
    (
        "Scheduler._mamba_block_aligned_split",
        "_mamba_block_aligned_split.upstream.py",
        "_mamba_block_aligned_split.patched.py",
    ),
    ("Scheduler.schedule", "schedule.upstream.py", "schedule.patched.py"),
):
    if replace_whole_body(
        SCHEDULER,
        PATCH_DIR / upstream_name,
        PATCH_DIR / patched_name,
        label,
        skip_if_upstream_missing=True,
    ):
        changed = True

if not changed:
    print("All offload scheduler patches already applied")
PY

"$VENV/bin/python" -m py_compile "$SCHEDULER"
echo "Syntax check OK"
