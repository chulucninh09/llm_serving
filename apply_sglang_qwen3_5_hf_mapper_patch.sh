#!/usr/bin/env bash
# Alias Qwen3.5 pre-fusion GDN projection names onto fused runtime tensors.
#
# Checkpoints declare ignore/dynamic lists with original names
# (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a). SGLang fuses those at load
# time into in_proj_qkvz / in_proj_ba. Without hf_to_sglang_mapper, GPTQ/Marlin
# never skip the fused 24-wide in_proj_ba and crash.
#
# Same class of fix as vLLM #34697 (packed_modules_mapping), implemented
# through SGLang's WeightsMapper.
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

if [[ ! -f "$GPTQ" ]]; then
  echo "gptq.py not found at $GPTQ" >&2
  exit 1
fi

for payload in \
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

def apply_snippet(path: Path, label: str, upstream_name: str, patched_name: str, *, replace_all: bool = False) -> bool:
    text = path.read_text()
    upstream = (PATCH_DIR / upstream_name).read_text()
    patched = (PATCH_DIR / patched_name).read_text()

    if patched in text and upstream not in text:
        print(f"{label}: already patched")
        return False

    if upstream not in text:
        raise SystemExit(
            f"Patch failed: {label} upstream snippet not found in {path} "
            f"(sglang version may have changed; update {PATCH_DIR / upstream_name})"
        )

    n = text.count(upstream)
    text = text.replace(upstream, patched) if replace_all else text.replace(upstream, patched, 1)
    path.write_text(text)
    applied = n if replace_all else 1
    print(f"{label}: applied ({applied} occurrence{'s' if applied != 1 else ''})")
    return True

qwen = Path("${QWEN3_5}")
gptq = Path("${GPTQ}")
changed = False

changed |= apply_snippet(
    qwen,
    "Qwen3_5*ForConditionalGeneration.hf_to_sglang_mapper",
    "hf_to_sglang_mapper.upstream.py",
    "hf_to_sglang_mapper.patched.py",
    replace_all=True,
)
changed |= apply_snippet(
    qwen,
    "Qwen3_5ForCausalLM.hf_to_sglang_mapper",
    "causal_lm_mapper.upstream.py",
    "causal_lm_mapper.patched.py",
)
changed |= apply_snippet(
    gptq,
    "GPTQConfig.apply_weight_name_mapper",
    "gptq_config_mapper.upstream.py",
    "gptq_config_mapper.patched.py",
)
changed |= apply_snippet(
    gptq,
    "GPTQMarlinConfig.apply_weight_name_mapper",
    "gptq_marlin_mapper.upstream.py",
    "gptq_marlin_mapper.patched.py",
)

if not changed:
    print("All Qwen3.5 HF mapper patches already applied")
PY

"$VENV/bin/python" -m py_compile "$QWEN3_5" "$GPTQ"

"$VENV/bin/python" <<PY
from pathlib import Path

from sglang.srt.models.utils import WeightsMapper

qwen = Path("${QWEN3_5}").read_text()
gptq = Path("${GPTQ}").read_text()

assert "hf_to_sglang_mapper = None" not in qwen, "qwen3_5.py still has hf_to_sglang_mapper = None"
assert '"in_proj_qkv": "in_proj_qkvz"' in qwen
assert '"in_proj_a": "in_proj_ba"' in qwen
assert "def apply_weight_name_mapper(self, hf_to_sglang_mapper):" in gptq

mapper = WeightsMapper(
    orig_to_new_substr={
        "in_proj_qkv": "in_proj_qkvz",
        "in_proj_z": "in_proj_qkvz",
        "in_proj_b": "in_proj_ba",
        "in_proj_a": "in_proj_ba",
    },
)
assert mapper.apply_list(["in_proj_a", "in_proj_b"]) == ["in_proj_ba", "in_proj_ba"]
assert mapper.apply_list(["in_proj_qkv", "in_proj_z"]) == ["in_proj_qkvz", "in_proj_qkvz"]
print("Qwen3.5 HF mapper aliases verified OK")
PY

echo "Qwen3.5 HF mapper patch applied successfully"
