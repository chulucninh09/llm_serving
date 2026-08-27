#!/usr/bin/env bash
# Keep Qwen3.5 fused in_proj_ba unquantized.
#
# GDN checkpoints leave in_proj_a / in_proj_b in bf16. SGLang fuses them into
# in_proj_ba (2 * num_v_heads wide → 24 at TP=4), which Marlin cannot repack.
# compressed-tensors ignore matching expands fused names BACK to those shards
# and refuses parent-prefix matches, so rewriting ignore lists to in_proj_ba
# (or ignoring linear_attn) does not skip the layer.
#
# Force quant_config=None on create_ba_proj. Also revert the old WeightsMapper
# ignore rewrite if a previous install applied it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$SCRIPT_DIR/.venv}"
PATCH_DIR="$SCRIPT_DIR/patches/sglang_qwen3_5_hf_mapper"

PYTHON_VER="$("$VENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
SITE="$VENV/lib/python${PYTHON_VER}/site-packages"
QWEN3_5="$SITE/sglang/srt/models/qwen3_5.py"
GPTQ="$SITE/sglang/srt/layers/quantization/gptq/gptq.py"

if [[ ! -f "$QWEN3_5" ]]; then
  echo "qwen3_5.py not found at $QWEN3_5" >&2
  exit 1
fi

for payload in \
  create_ba_proj.upstream.py \
  create_ba_proj.patched.py \
  hf_to_sglang_mapper.upstream.py \
  hf_to_sglang_mapper.patched.py \
  causal_lm_mapper.upstream.py \
  causal_lm_mapper.patched.py \
  gptq_config_mapper.upstream.py \
  gptq_config_mapper.patched.py \
  gptq_marlin_mapper.upstream.py \
  gptq_marlin_mapper.patched.py
do
  if [[ ! -f "$PATCH_DIR/$payload" ]]; then
    echo "Missing patch payload: $PATCH_DIR/$payload" >&2
    exit 1
  fi
done

"$VENV/bin/python" <<PY
from pathlib import Path

PATCH_DIR = Path("${PATCH_DIR}")

def swap_snippet(path: Path, label: str, from_name: str, to_name: str, *, replace_all: bool = False) -> bool:
    if not path.is_file():
        print(f"{label}: skipped (missing {path})")
        return False

    text = path.read_text()
    src = (PATCH_DIR / from_name).read_text()
    dst = (PATCH_DIR / to_name).read_text()

    if dst in text and src not in text:
        print(f"{label}: already at target")
        return False

    if src not in text:
        print(f"{label}: source snippet not present, skipping")
        return False

    n = text.count(src)
    text = text.replace(src, dst) if replace_all else text.replace(src, dst, 1)
    path.write_text(text)
    applied = n if replace_all else 1
    print(f"{label}: applied ({applied} occurrence{'s' if applied != 1 else ''})")
    return True

qwen = Path("${QWEN3_5}")
gptq = Path("${GPTQ}")
changed = False

# Drop the old ignore-list rewrite if a previous install applied it.
changed |= swap_snippet(
    qwen,
    "revert Qwen3_5*ForConditionalGeneration.hf_to_sglang_mapper",
    "hf_to_sglang_mapper.patched.py",
    "hf_to_sglang_mapper.upstream.py",
    replace_all=True,
)
changed |= swap_snippet(
    qwen,
    "revert Qwen3_5ForCausalLM.hf_to_sglang_mapper",
    "causal_lm_mapper.patched.py",
    "causal_lm_mapper.upstream.py",
)
changed |= swap_snippet(
    gptq,
    "revert GPTQConfig.apply_weight_name_mapper",
    "gptq_config_mapper.patched.py",
    "gptq_config_mapper.upstream.py",
)
changed |= swap_snippet(
    gptq,
    "revert GPTQMarlinConfig.apply_weight_name_mapper",
    "gptq_marlin_mapper.patched.py",
    "gptq_marlin_mapper.upstream.py",
)
changed |= swap_snippet(
    qwen,
    "Qwen3_5GatedDeltaNet.create_ba_proj",
    "create_ba_proj.upstream.py",
    "create_ba_proj.patched.py",
)

if not changed:
    print("All Qwen3.5 in_proj_ba patches already applied")
PY

"$VENV/bin/python" -m py_compile "$QWEN3_5"
if [[ -f "$GPTQ" ]]; then
  "$VENV/bin/python" -m py_compile "$GPTQ"
fi

"$VENV/bin/python" <<PY
from pathlib import Path

qwen = Path("${QWEN3_5}").read_text()
assert "quant_config=None" in qwen, "create_ba_proj still passes quant_config through"
assert (
    "orig_to_new_substr={\n            \"in_proj_qkv\": \"in_proj_qkvz\"" not in qwen
), "old in_proj ignore mapper is still present"

import inspect
from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

src = inspect.getsource(Qwen3_5GatedDeltaNet.create_ba_proj)
assert "quant_config=None" in src, "create_ba_proj source does not skip quantization"
print("Qwen3.5 in_proj_ba unquantized patch verified OK")
PY

echo "Qwen3.5 in_proj_ba patch applied successfully"
