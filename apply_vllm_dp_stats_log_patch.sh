#!/usr/bin/env bash
# Patch vLLM so EngineCore prints Avg prompt/generation throughput stats even
# when --api-server-count > 1 (default with -dp > 1). Frontend loggers stay
# disabled in that mode because each API server only sees a slice of traffic.
#
# Replaces entire function bodies after confirming the on-disk version matches
# the expected upstream snapshot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$SCRIPT_DIR/.venv}"
PATCH_DIR="$SCRIPT_DIR/patches/vllm_dp_stats_log"

# Auto-detect the Python version used by the venv
PYTHON_VER="$("$VENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
VLLM_SITE="$VENV/lib/python${PYTHON_VER}/site-packages/vllm"
CORE="$VLLM_SITE/v1/engine/core.py"
LOGGERS="$VLLM_SITE/v1/metrics/loggers.py"

if [[ ! -f "$CORE" ]]; then
  echo "core.py not found at $CORE" >&2
  exit 1
fi

if [[ ! -f "$LOGGERS" ]]; then
  echo "loggers.py not found at $LOGGERS" >&2
  exit 1
fi

for payload in \
  _attach_iteration_details.upstream.py \
  _attach_iteration_details.patched.py \
  StatLoggerManager_init.upstream.py \
  StatLoggerManager_init.patched.py
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

CORE = Path("${CORE}")
LOGGERS = Path("${LOGGERS}")
PATCH_DIR = Path("${PATCH_DIR}")
changed = False

for path, label, upstream_name, patched_name in (
    (
        CORE,
        "EngineCore._attach_iteration_details",
        "_attach_iteration_details.upstream.py",
        "_attach_iteration_details.patched.py",
    ),
    (
        LOGGERS,
        "StatLoggerManager.__init__",
        "StatLoggerManager_init.upstream.py",
        "StatLoggerManager_init.patched.py",
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
    print("All DP stats log patches already applied")
PY

"$VENV/bin/python" -m py_compile "$CORE" "$LOGGERS"
"$VENV/bin/python" -c "
from pathlib import Path
core = Path('${CORE}').read_text()
loggers = Path('${LOGGERS}').read_text()
assert '_console_stat_logger' in core, 'EngineCore console stats hook missing'
assert 'console stats are logged from each EngineCore instead' in loggers
print('EngineCore console stats hook OK')
"
echo "Syntax check OK"
