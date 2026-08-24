#!/usr/bin/env bash
# Pin KV offload CPU memory synchronously so cudaHostRegister does not race
# with CUDA graph capture on multi-GPU workers.
#
# Replaces the entire CPUOffloadingWorker.__init__ body after confirming the
# on-disk version matches the expected upstream snapshot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$SCRIPT_DIR/.venv}"
PATCH_DIR="$SCRIPT_DIR/patches/vllm_kv_offload_pin"
GPU_WORKER="$VENV/lib/python3.13/site-packages/vllm/v1/kv_offload/cpu/gpu_worker.py"

if [[ ! -f "$GPU_WORKER" ]]; then
  echo "gpu_worker.py not found at $GPU_WORKER" >&2
  exit 1
fi

for payload in gpu_worker.upstream.py gpu_worker.patched.py; do
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

GPU_WORKER = Path("${GPU_WORKER}")
PATCH_DIR = Path("${PATCH_DIR}")

replace_whole_body(
    GPU_WORKER,
    PATCH_DIR / "gpu_worker.upstream.py",
    PATCH_DIR / "gpu_worker.patched.py",
    "CPUOffloadingWorker.__init__",
)
PY

"$VENV/bin/python" -m py_compile "$GPU_WORKER"
echo "Syntax check OK"
