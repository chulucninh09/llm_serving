#!/usr/bin/env bash
# Patch vLLM to accept thinking_token_budget in --reasoning-config and apply it
# as the default SamplingParams.thinking_token_budget at inference time.
#
# Replaces entire function bodies after confirming the on-disk version matches
# the expected upstream snapshot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$SCRIPT_DIR/.venv}"
PATCH_DIR="$SCRIPT_DIR/patches/vllm_thinking_budget"

# Auto-detect the Python version used by the venv
PYTHON_VER="$("$VENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
VLLM_SITE="$VENV/lib/python${PYTHON_VER}/site-packages/vllm"
REASONING="$VLLM_SITE/config/reasoning.py"
INPUT_PROCESSOR="$VLLM_SITE/v1/engine/input_processor.py"

if [[ ! -f "$REASONING" ]]; then
  echo "reasoning.py not found at $REASONING" >&2
  exit 1
fi

if [[ ! -f "$INPUT_PROCESSOR" ]]; then
  echo "input_processor.py not found at $INPUT_PROCESSOR" >&2
  exit 1
fi

for payload in \
  ReasoningConfig.upstream.py \
  ReasoningConfig.patched.py \
  process_inputs.upstream.py \
  process_inputs.patched.py
do
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

REASONING = Path("${REASONING}")
INPUT_PROCESSOR = Path("${INPUT_PROCESSOR}")
PATCH_DIR = Path("${PATCH_DIR}")
changed = False

for path, label, upstream_name, patched_name in (
    (REASONING, "ReasoningConfig", "ReasoningConfig.upstream.py", "ReasoningConfig.patched.py"),
    (
        INPUT_PROCESSOR,
        "process_inputs",
        "process_inputs.upstream.py",
        "process_inputs.patched.py",
    ),
):
    if replace_whole_body(
        path,
        PATCH_DIR / upstream_name,
        PATCH_DIR / patched_name,
        label,
    ):
        changed = True

if not changed:
    print("All thinking budget patches already applied")
PY

"$VENV/bin/python" -m py_compile "$REASONING" "$INPUT_PROCESSOR"
"$VENV/bin/python" -c "
import json
from vllm.config.reasoning import ReasoningConfig
ReasoningConfig(
    **json.loads(
        '{\"reasoning_start_str\": \"<t>\", '
        '\"reasoning_end_str\": \"</t>\", '
        '\"thinking_token_budget\": 2048}'
    )
)
print('ReasoningConfig thinking_token_budget OK')
"
echo "Syntax check OK"
