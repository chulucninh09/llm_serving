#!/usr/bin/env bash
# Restore DFlashQwen3Model.decoder_layer_cls hook removed by vLLM #52560.
# DFlash2Qwen3Model overrides decoder_layer_cls to DFlash2Qwen3DecoderLayer
# (with attention_conv); without this hook, DFlash2 draft weights fail to load.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$SCRIPT_DIR/.venv}"
PATCH_DIR="$SCRIPT_DIR/patches/vllm_dflash2_decoder_layer"

PYTHON_VER="$("$VENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
QWEN3_DFLASH="$VENV/lib/python${PYTHON_VER}/site-packages/vllm/model_executor/models/qwen3_dflash.py"

if [[ ! -f "$QWEN3_DFLASH" ]]; then
  echo "qwen3_dflash.py not found at $QWEN3_DFLASH" >&2
  exit 1
fi

for payload in \
  DFlashQwen3Model_class.upstream.py \
  DFlashQwen3Model_class.patched.py \
  layers_init.upstream.py \
  layers_init.patched.py
do
  if [[ ! -f "$PATCH_DIR/$payload" ]]; then
    echo "Missing patch payload: $PATCH_DIR/$payload" >&2
    exit 1
  fi
done

"$VENV/bin/python" <<PY
from pathlib import Path

QWEN3_DFLASH = Path("${QWEN3_DFLASH}")
PATCH_DIR = Path("${PATCH_DIR}")
text = QWEN3_DFLASH.read_text()
changed = False

for label, upstream_name, patched_name in (
    ("DFlashQwen3Model class", "DFlashQwen3Model_class.upstream.py", "DFlashQwen3Model_class.patched.py"),
    ("layers init", "layers_init.upstream.py", "layers_init.patched.py"),
):
    upstream = (PATCH_DIR / upstream_name).read_text()
    patched = (PATCH_DIR / patched_name).read_text()

    if upstream == patched:
        print(f"{label}: skipped (upstream and patched are identical; no patch needed)")
        continue

    if patched in text:
        print(f"{label}: already patched")
        continue

    if upstream not in text:
        raise SystemExit(
            f"Patch failed: {label} upstream snippet not found in "
            f"{QWEN3_DFLASH} (vLLM version may have changed; update {PATCH_DIR / upstream_name})"
        )

    text = text.replace(upstream, patched, 1)
    print(f"{label}: applied")
    changed = True

if changed:
    QWEN3_DFLASH.write_text(text)
else:
    print("All DFlash2 decoder_layer_cls patches already applied")
PY

"$VENV/bin/python" -m py_compile "$QWEN3_DFLASH"

"$VENV/bin/python" <<PY
from pathlib import Path

text = Path("${QWEN3_DFLASH}").read_text()
assert "decoder_layer_cls = DFlashQwen3DecoderLayer" in text, "decoder_layer_cls attribute missing"
assert "self.decoder_layer_cls(" in text, "self.decoder_layer_cls() call missing"
print("DFlash2 decoder_layer_cls hook verified OK")
PY

echo "DFlash2 decoder_layer_cls patch applied successfully"
