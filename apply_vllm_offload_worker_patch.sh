#!/usr/bin/env bash
# Fix OffloadingConnector KV registration for hybrid models (Qwen3.6, etc.)
# where layers in the same shared_by slot can have different byte strides.
#
# Replaces the entire register_kv_caches method body after confirming the
# on-disk version matches the expected upstream snapshot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$SCRIPT_DIR/.venv}"
PATCH_DIR="$SCRIPT_DIR/patches/vllm_offload_worker"
WORKER="$VENV/lib/python3.13/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py"

if [[ ! -f "$WORKER" ]]; then
  echo "offloading/worker.py not found at $WORKER" >&2
  exit 1
fi

for payload in register_kv_caches.upstream.py register_kv_caches.patched.py; do
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

WORKER = Path("${WORKER}")
PATCH_DIR = Path("${PATCH_DIR}")

replace_whole_body(
    WORKER,
    PATCH_DIR / "register_kv_caches.upstream.py",
    PATCH_DIR / "register_kv_caches.patched.py",
    "OffloadingConnectorWorker.register_kv_caches",
)
PY

"$VENV/bin/python" -m py_compile "$WORKER"
echo "Syntax check OK"
