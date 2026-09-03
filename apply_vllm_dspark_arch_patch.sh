#!/usr/bin/env bash
# Recognize Qwen3-based DSpark drafts as Qwen3DSparkModel instead of routing
# them to the DeepSeek-V4 class.
#
# DSpark draft checkpoints (e.g. RadixArk/Qwen3.8-27B-DSpark, model_type=qwen3)
# declare "architectures": ["DSparkDraftModel"]. The registry maps that arch to
# vllm.models.deepseek_v4/DSparkDeepseekV4ForCausalLM, and speculative.py then
# rewrites model_type to deepseek_v4, so loading fails. Qwen3DSparkModel selects
# qwen3_dspark.Qwen3DSparkForCausalLM, which expects exactly the fields these
# checkpoints already ship (dflash_config, markov_rank, enable_confidence_head,
# confidence_head_with_markov, layer_types, head_dim).
#
# The fix lives in SpeculativeConfig.hf_config_override, which runs on the draft
# config before method detection: a draft declaring DSparkDraftModel with
# model_type=qwen3 is rewritten to Qwen3DSparkModel. Real DeepSeek-V4 in-target
# DSpark (model_type=deepseek_v4) is left untouched.
#
# The checkout root is the default when VENV does not contain vLLM: site-packages
# in the venv may hold sglang (this box serves via uv run vllm from the
# vllm-ampere-optimized checkout), and the apply_* scripts that target the
# checkout support VLLM_SITE / vendored layout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$SCRIPT_DIR/.venv}"
PATCH_DIR="$SCRIPT_DIR/patches/vllm_dspark_arch"

# Auto-detect the Python version used by the venv
PYTHON_VER="$("$VENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
VLLM_SITE="${VLLM_SITE:-$VENV/lib/python${PYTHON_VER}/site-packages/vllm}"

sites=()
if [[ -d "$VLLM_SITE" ]]; then
  sites+=("$VLLM_SITE")
fi

VENDORED="$SCRIPT_DIR/vllm-ampere-optimized/vllm/vllm"
if [[ -d "$VENDORED" ]]; then
  sites+=("$VENDORED")
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

if [[ ${#unique_sites[@]} -eq 0 ]]; then
  echo "No vLLM install found. Set VLLM_SITE or install vLLM in $VENV" >&2
  exit 1
fi

for payload in \
  dspark_arch_override.upstream.py \
  dspark_arch_override.patched.py
do
  if [[ ! -f "$PATCH_DIR/$payload" ]]; then
    echo "Missing patch payload: $PATCH_DIR/$payload" >&2
    exit 1
  fi
done

patched_any=0
for site in "${unique_sites[@]}"; do
  SPECULATIVE="$site/config/speculative.py"
  if [[ ! -f "$SPECULATIVE" ]]; then
    echo "skip $site (config/speculative.py not found)"
    continue
  fi

  echo "Patching $site"
  "$VENV/bin/python" <<PY
from pathlib import Path

SPECULATIVE = Path("${SPECULATIVE}")
PATCH_DIR = Path("${PATCH_DIR}")
text = SPECULATIVE.read_text()

upstream = (PATCH_DIR / "dspark_arch_override.upstream.py").read_text()
patched = (PATCH_DIR / "dspark_arch_override.patched.py").read_text()

if upstream == patched:
    print("  skipped (upstream and patched are identical; no patch needed)")
elif patched in text:
    print("  already patched")
elif upstream not in text:
    raise SystemExit(
        "Patch failed: upstream hf_config_override body not found in "
        f"{SPECULATIVE} (vLLM version may have changed; update "
        f"{PATCH_DIR / 'dspark_arch_override.upstream.py'})"
    )
else:
    SPECULATIVE.write_text(text.replace(upstream, patched, 1))
    print("  applied")
PY

  "$VENV/bin/python" -m py_compile "$SPECULATIVE"
  patched_any=1
done

if [[ "$patched_any" -eq 0 ]]; then
  echo "No speculative.py targets were patched" >&2
  exit 1
fi

"$VENV/bin/python" <<PY
import importlib.util

from transformers import PretrainedConfig

if importlib.util.find_spec("vllm") is None:
    # The venv may not import the patched checkout (vllm deps can be heavy).
    # Verification of behavior happens on the box that serves vLLM.
    print("vllm not importable in this venv; skipped behavior verification")
else:
    from vllm.config.speculative import SpeculativeConfig

    qwen3_dspark = SpeculativeConfig.hf_config_override(
        PretrainedConfig(model_type="qwen3", architectures=["DSparkDraftModel"])
    )
    assert qwen3_dspark.architectures == ["Qwen3DSparkModel"], (
        f"expected Qwen3DSparkModel, got {qwen3_dspark.architectures}"
    )

    deepseek_v4 = SpeculativeConfig.hf_config_override(
        PretrainedConfig(
            model_type="deepseek_v4",
            architectures=["DSparkDraftModel"],
        )
    )
    assert "Qwen3DSparkModel" not in deepseek_v4.architectures, (
        f"DeepSeek-V4 DSpark must not become Qwen3DSparkModel, "
        f"got {deepseek_v4.architectures}"
    )

    qwen3_plain = SpeculativeConfig.hf_config_override(
        PretrainedConfig(model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    )
    assert qwen3_plain.architectures == ["Qwen3ForCausalLM"], (
        f"unrelated qwen3 arch must not be rewritten, got {qwen3_plain.architectures}"
    )
    print("hf_config_override DSpark arch recognition verified OK")
PY

echo "Syntax check OK"
