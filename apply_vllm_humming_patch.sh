#!/usr/bin/env bash
# Patch humming schema to support AutoRound ("auto-round") weight format
# by mapping it to GPTQWeightSchema in humming/schema/__init__.py.
#
# AutoRound checkpoints use the same qweight/scales/qzeros/g_idx layout as
# GPTQ, so the existing GPTQWeightSchema handles them natively once the
# WEIGHT_SCHEMA_MAP entry is added.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$SCRIPT_DIR/.venv}"
PATCH_DIR="$SCRIPT_DIR/patches/vllm_humming"
HUMMING_SCHEMA_INIT="$VENV/lib/python3.13/site-packages/humming/schema/__init__.py"

if [[ ! -f "$HUMMING_SCHEMA_INIT" ]]; then
  echo "humming schema __init__.py not found at $HUMMING_SCHEMA_INIT" >&2
  exit 1
fi

for payload in \
  WEIGHT_SCHEMA_MAP.upstream.py \
  WEIGHT_SCHEMA_MAP.patched.py
do
  if [[ ! -f "$PATCH_DIR/$payload" ]]; then
    echo "Missing patch payload: $PATCH_DIR/$payload" >&2
    exit 1
  fi
done

"$VENV/bin/python" <<PY
from pathlib import Path

SCHEMA_INIT = Path("$HUMMING_SCHEMA_INIT")
PATCH_DIR = Path("$PATCH_DIR")
UPSTREAM = PATCH_DIR / "WEIGHT_SCHEMA_MAP.upstream.py"
PATCHED = PATCH_DIR / "WEIGHT_SCHEMA_MAP.patched.py"

text = SCHEMA_INIT.read_text()
upstream = UPSTREAM.read_text()
patched = PATCHED.read_text()

if patched in text:
    print("WEIGHT_SCHEMA_MAP: already patched")
else:
    if upstream not in text:
        raise SystemExit(
            "Patch failed: upstream WEIGHT_SCHEMA_MAP not found in "
            f"{SCHEMA_INIT} (humming version may have changed; update {UPSTREAM})"
        )

    SCHEMA_INIT.write_text(text.replace(upstream, patched, 1))
    print("WEIGHT_SCHEMA_MAP: replaced with AutoRound support")
PY

"$VENV/bin/python" -m py_compile "$HUMMING_SCHEMA_INIT"

"$VENV/bin/python" <<'PY'
from humming.schema import WEIGHT_SCHEMA_MAP
assert "auto-round" in WEIGHT_SCHEMA_MAP, "auto-round not in WEIGHT_SCHEMA_MAP"
assert "auto_round" in WEIGHT_SCHEMA_MAP, "auto_round not in WEIGHT_SCHEMA_MAP"
assert WEIGHT_SCHEMA_MAP["auto-round"].__name__ == "GPTQWeightSchema"
assert WEIGHT_SCHEMA_MAP["auto_round"].__name__ == "GPTQWeightSchema"
print("humming AutoRound support verified OK")
PY

echo "humming AutoRound patch applied successfully"
