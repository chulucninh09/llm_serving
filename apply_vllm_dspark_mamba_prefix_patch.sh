#!/usr/bin/env bash
# Fix vLLM prefix-cache reuse for DSpark + hybrid-Mamba (Qwen3.8-27B).
#
# With method=dspark no KV cache group can be identified as the drafter's
# (the non_causal_multi_token_decode marker lives only on MLAAttentionSpec),
# so the "flag all groups as draft" fallback marks every group -- including
# the Mamba groups -- as EAGLE. Align-mode Mamba checkpoints are only taken at
# exact chunk boundaries and can never satisfy an eagle-widened lookup window,
# so Mamba prefix reuse drops to zero.
#
# Fix A: exclude recurrent-state (Mamba) groups from that fallback in both
# consumers (KVCacheCoordinator and the offloading connector scheduler), via a
# shared is_mamba_group() helper in kv_cache_interface.py. Also align the
# _warn_if_unannotated_eagle_mamba gate with use_eagle_block_drop() and reword
# its message, and log the fallback state.
#
# Targets the installed venv (vllm >= 0.27 with non_causal_multi_token_decode).
# The vendored `vllm-ampere-optimized` fork is v0.25.1 and predates the whole
# `_annotate_eagle_groups` machinery, so it is intentionally skipped.
#
# Mirrors the upstream/patched payload style of apply_vllm_dspark_arch_patch.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$SCRIPT_DIR/.venv}"
PATCH_DIR="$SCRIPT_DIR/patches/vllm_dspark_mamba_prefix"

# Auto-detect the Python version used by the venv.
PYTHON_VER="$("$VENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
VLLM_SITE="${VLLM_SITE:-$VENV/lib/python${PYTHON_VER}/site-packages/vllm}"

sites=()
if [[ -d "$VLLM_SITE" ]]; then
  sites+=("$VLLM_SITE")
fi

# Optional override for an out-of-venv checkout (e.g. the /home/coder/vllm
# @ 514c731 checkout referenced by the fix plan). Detection is capability-
# based: the site must already carry `_annotate_eagle_groups`.
if [[ -n "${VLLM_SITE_EXTRA:-}" ]]; then
  sites+=("$VLLM_SITE_EXTRA")
fi

# Deduplicate while preserving order.
declare -A seen=()
unique_sites=()
for site in "${sites[@]}"; do
  if [[ -z "${seen[$site]:-}" ]]; then
    seen[$site]=1
    unique_sites+=("$site")
  fi
done

if [[ ${#unique_sites[@]} -eq 0 ]]; then
  echo "No vLLM install found. Set VLLM_SITE or install vLLM in $VENV" >&2
  exit 1
fi

# Payloads shared across sites.
for payload in \
  coordinator_import.upstream.py \
  coordinator_import.patched.py \
  coordinator_fallback.upstream.py \
  coordinator_fallback.patched.py \
  offloading_import.upstream.py \
  offloading_import.patched.py \
  offloading_fallback.upstream.py \
  offloading_fallback.patched.py \
  is_mamba_group.upstream.py \
  is_mamba_group.patched.py \
  warn_gate.upstream.py \
  warn_gate.patched.py \
  warn_message.upstream.py \
  warn_message.patched.py
do
  if [[ ! -f "$PATCH_DIR/$payload" ]]; then
    echo "Missing patch payload: $PATCH_DIR/$payload" >&2
    exit 1
  fi
done

patched_any=0
for site in "${unique_sites[@]}"; do
  COORDINATOR="$site/v1/core/kv_cache_coordinator.py"
  OFFLOADING="$site/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
  INTERFACE="$site/v1/kv_cache_interface.py"
  KV_UTILS="$site/v1/core/kv_cache_utils.py"

  if [[ ! -f "$COORDINATOR" || ! -f "$OFFLOADING" || ! -f "$INTERFACE" || ! -f "$KV_UTILS" ]]; then
    echo "skip $site (target files not found)"
    continue
  fi

  # Capability gate: this patch only makes sense on vLLM that already has the
  # `_annotate_eagle_groups` spec-driven annotation (>= 0.27). Older checkouts
  # (e.g. the vendored v0.25.1 fork) have no such machinery.
  if ! grep -q "_annotate_eagle_groups" "$KV_UTILS"; then
    echo "skip $site (no _annotate_eagle_groups; pre-marker vLLM)"
    continue
  fi

  echo "Patching $site"
  "$VENV/bin/python" <<PY
from pathlib import Path

COORDINATOR = Path("${COORDINATOR}")
OFFLOADING = Path("${OFFLOADING}")
INTERFACE = Path("${INTERFACE}")
KV_UTILS = Path("${KV_UTILS}")
PATCH_DIR = Path("${PATCH_DIR}")


def apply(target, upstream_name, patched_name, label):
    upstream = (PATCH_DIR / upstream_name).read_text()
    patched = (PATCH_DIR / patched_name).read_text()
    text = target.read_text()
    if upstream == patched:
        print(f"  {label}: skipped (upstream and patched are identical)")
        return
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


apply(COORDINATOR, "coordinator_import.upstream.py", "coordinator_import.patched.py", "coordinator import")
apply(COORDINATOR, "coordinator_fallback.upstream.py", "coordinator_fallback.patched.py", "coordinator fallback")
apply(OFFLOADING, "offloading_import.upstream.py", "offloading_import.patched.py", "offloading import")
apply(OFFLOADING, "offloading_fallback.upstream.py", "offloading_fallback.patched.py", "offloading fallback")
apply(INTERFACE, "is_mamba_group.upstream.py", "is_mamba_group.patched.py", "is_mamba_group helper")
apply(KV_UTILS, "warn_gate.upstream.py", "warn_gate.patched.py", "warn gate")
apply(KV_UTILS, "warn_message.upstream.py", "warn_message.patched.py", "warn message")
PY

  "$VENV/bin/python" -m py_compile "$COORDINATOR" "$OFFLOADING" "$INTERFACE" "$KV_UTILS"
  patched_any=1
done

if [[ "$patched_any" -eq 0 ]]; then
  echo "No vLLM targets were patched" >&2
  exit 1
fi

"$VENV/bin/python" <<PY
import importlib.util

if importlib.util.find_spec("vllm") is None:
    print("vllm not importable in this venv; skipped behavior verification")
else:
    import torch
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MambaSpec,
        is_mamba_group,
    )

    mamba = MambaSpec(block_size=16, shapes=((1, 16),), dtypes=(torch.float32,))
    fa = FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16
    )
    assert is_mamba_group(KVCacheGroupSpec(["mamba"], mamba)) is True
    assert is_mamba_group(KVCacheGroupSpec(["attn"], fa)) is False
    print("is_mamba_group helper verified OK")
PY

echo "Syntax check OK"
