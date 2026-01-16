#!/usr/bin/env bash
set -euo pipefail

ts() { date '+%Y-%m-%d %H:%M:%S'; }


# Ensure progress logs show up immediately.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# Default args (can be overridden by passing your own flags)
DEFAULT_ARGS=(
  "--warmup" "1"
  "--runs" "100"
)

PY_BIN="${ENGRAM_PYTHON:-}"
if [[ -z "${PY_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PY_BIN="python3"
  else
    PY_BIN="python"
  fi
fi

echo "[$(ts)] run_benchmark.sh: python=${PY_BIN}" >&2
"${PY_BIN}" -V >&2 || true
echo "[$(ts)] run_benchmark.sh: args: ${DEFAULT_ARGS[*]} $*" >&2

exec "${PY_BIN}" benchmark_engram_forward.py "${DEFAULT_ARGS[@]}" "$@"
