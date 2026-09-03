#!/usr/bin/env bash
# Opt-in FP8 KV cache on sub-SM89 NVIDIA GPUs (e.g. SM86 RTX 3090).
#
# vLLM's Triton attention backend hard-requires native fp8e4nv tensor cores
# (SM89+ / Hopper) for an FP8 KV cache. On SM80/SM86 (Ampere-class) there is
# no native fp8e4nv mma, so upstream vLLM refuses to start with
# `--kv-cache-dtype fp8`. This patch adds an opt-in escape hatch controlled by
# the `VLLM_ALLOW_FP8_KV_CACHE_BELOW_SM89=1` environment variable:
#
#   * envs.py            -- registers the new env var (annotation + mapping)
#   * triton_attn.py     -- allows fp8 KV cache when the env var is set
#   * triton_reshape_and_cache_flash.py -- allows the fp8 reshape kernel below SM89
#
# When enabled, the Triton backend stores keys/values in fp8_e4m3 and
# dequantizes in-kernel (the paged-attention/unified kernel loads the fp8 cache
# and upcasts before the dot product). This is SLOWER than bf16/int8 and can
# lose accuracy; it exists only for experiments that specifically need
# "FULL decode in fp8 kv cache" on an Ampere GPU.
#
# Mirrors the upstream/patched payload style of apply_vllm_dspark_arch_patch.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$SCRIPT_DIR/.venv}"
PATCH_DIR="$SCRIPT_DIR/patches/vllm_fp8_sm86"

PYTHON_VER="$("$VENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
VLLM_SITE="${VLLM_SITE:-$VENV/lib/python${PYTHON_VER}/site-packages/vllm}"

if [[ ! -d "$VLLM_SITE" ]]; then
  echo "No vLLM install found at $VLLM_SITE" >&2
  exit 1
fi

ENVS="$VLLM_SITE/envs.py"
TRITON_ATTN="$VLLM_SITE/v1/attention/backends/triton_attn.py"
RESHAPE="$VLLM_SITE/v1/attention/ops/triton_reshape_and_cache_flash.py"

for f in "$ENVS" "$TRITON_ATTN" "$RESHAPE"; do
  if [[ ! -f "$f" ]]; then
    echo "skip $VLLM_SITE (missing $f)" >&2
    exit 1
  fi
done

"$VENV/bin/python" <<PY
from pathlib import Path

ENVS = Path("${ENVS}")
TRITON_ATTN = Path("${TRITON_ATTN}")
RESHAPE = Path("${RESHAPE}")
PATCH_DIR = Path("${PATCH_DIR}")


def apply(target, upstream_name, patched_name, label):
    upstream = (PATCH_DIR / upstream_name).read_text()
    patched = (PATCH_DIR / patched_name).read_text()
    text = target.read_text()
    if patched in text:
        print(f"  {label}: already patched")
        return
    if upstream not in text:
        raise SystemExit(
            f"Patch failed: {label} upstream block not found in {target} "
            f"(vLLM version may have changed; update {PATCH_DIR / upstream_name})"
        )
    target.write_text(text.replace(upstream, patched, 1))
    print(f"  {label}: applied")


apply(ENVS, "env_fp8_force.upstream.py", "env_fp8_force.patched.py", "envs annotation")
apply(ENVS, "env_fp8_force_map.upstream.py", "env_fp8_force_map.patched.py", "envs mapping")
apply(TRITON_ATTN, "triton_attn_gate.upstream.py", "triton_attn_gate.patched.py", "triton_attn fp8 gate")
apply(RESHAPE, "reshape_import.upstream.py", "reshape_import.patched.py", "reshape import os")
apply(RESHAPE, "reshape_gate.upstream.py", "reshape_gate.patched.py", "reshape fp8 gate")
PY

"$VENV/bin/python" -m py_compile "$ENVS" "$TRITON_ATTN" "$RESHAPE"

echo "Syntax check OK"
echo
echo "To enable fp8 KV cache on this GPU, launch vllm with:"
echo "  VLLM_ALLOW_FP8_KV_CACHE_BELOW_SM89=1 \\"
echo "    vllm serve ... --attention-backend TRITON_ATTN --kv-cache-dtype fp8"
