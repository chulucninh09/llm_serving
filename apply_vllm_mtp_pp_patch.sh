#!/usr/bin/env bash
# Enable Qwen3.5/Qwen3.6 MTP speculative decoding under pipeline parallelism.
#
# vLLM already places the entire draft on the last PP rank. The draft class
# still failed ModelConfig.verify_with_parallel_config because Qwen3_5MTP did
# not implement SupportsPP, and its forward used get_pp_group().is_first_rank
# which is false on that last rank.
#
# Replaces snippets after confirming the on-disk version matches the expected
# upstream snapshot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$SCRIPT_DIR/.venv}"
PATCH_DIR="$SCRIPT_DIR/patches/vllm_mtp_pp"

for payload in \
  imports.upstream.py \
  imports.patched.py \
  class_bases.upstream.py \
  class_bases.patched.py \
  mtp_forward.upstream.py \
  mtp_forward.patched.py \
  init_tail.upstream.py \
  init_tail.patched.py \
  verify_draft_pp.upstream.py \
  verify_draft_pp.patched.py
do
  if [[ ! -f "$PATCH_DIR/$payload" ]]; then
    echo "Missing patch payload: $PATCH_DIR/$payload" >&2
    exit 1
  fi
done

if [[ -x "$VENV/bin/python" ]]; then
  PYTHON="$VENV/bin/python"
else
  PYTHON="python3"
fi

sites=()
if [[ -n "${VLLM_SITE:-}" ]]; then
  sites+=("$VLLM_SITE")
fi

VENDORED="$SCRIPT_DIR/vllm-ampere-optimized/vllm/vllm"
if [[ -d "$VENDORED" ]]; then
  sites+=("$VENDORED")
fi

# The local .venv is often a different vLLM than the Ampere fork. Only patch
# it when explicitly requested (payloads are snapshotted against the fork).
if [[ "${VLLM_PATCH_VENV:-0}" == "1" && -x "$VENV/bin/python" ]]; then
  PYTHON_VER="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  VENV_SITE="$VENV/lib/python${PYTHON_VER}/site-packages/vllm"
  if [[ -d "$VENV_SITE" ]]; then
    sites+=("$VENV_SITE")
  fi
fi

if [[ ${#sites[@]} -eq 0 ]]; then
  echo "No vLLM install found. Set VLLM_SITE or install vLLM in $VENV" >&2
  exit 1
fi

# Deduplicate while preserving order
declare -A seen=()
unique_sites=()
for site in "${sites[@]}"; do
  if [[ -z "${seen[$site]:-}" ]]; then
    seen[$site]=1
    unique_sites+=("$site")
  fi
done

patched_any=0
for site in "${unique_sites[@]}"; do
  MTP="$site/model_executor/models/qwen3_5_mtp.py"
  SPEC="$site/config/speculative.py"
  if [[ ! -f "$MTP" ]]; then
    echo "skip $site (qwen3_5_mtp.py not found)"
    continue
  fi
  if [[ ! -f "$SPEC" ]]; then
    echo "skip $site (speculative.py not found)"
    continue
  fi

  echo "Patching $site"
  "$PYTHON" <<PY
import sys
from pathlib import Path

sys.path.insert(0, "${SCRIPT_DIR}/scripts")
from vllm_function_patch import replace_whole_body

MTP = Path("${MTP}")
SPEC = Path("${SPEC}")
PATCH_DIR = Path("${PATCH_DIR}")
changed = False

for path, label, upstream_name, patched_name in (
    (MTP, "qwen3_5_mtp.imports", "imports.upstream.py", "imports.patched.py"),
    (MTP, "Qwen3_5MTP.bases", "class_bases.upstream.py", "class_bases.patched.py"),
    (
        MTP,
        "Qwen3_5MultiTokenPredictor.forward",
        "mtp_forward.upstream.py",
        "mtp_forward.patched.py",
    ),
    (MTP, "Qwen3_5MTP.__init__", "init_tail.upstream.py", "init_tail.patched.py"),
    (
        SPEC,
        "SpeculativeConfig._verify_args draft PP",
        "verify_draft_pp.upstream.py",
        "verify_draft_pp.patched.py",
    ),
):
    if replace_whole_body(
        path,
        PATCH_DIR / upstream_name,
        PATCH_DIR / patched_name,
        label,
        skip_if_upstream_missing=True,
    ):
        changed = True

if not changed:
    print("All MTP+PP patches already applied")
PY

  "$PYTHON" -m py_compile "$MTP" "$SPEC"
  patched_any=1
done

if [[ "$patched_any" -eq 0 ]]; then
  echo "No qwen3_5_mtp.py / speculative.py targets were patched" >&2
  exit 1
fi

echo "Syntax check OK"
echo "If serving via docker, bind-mount the patched files or rebuild the image."
