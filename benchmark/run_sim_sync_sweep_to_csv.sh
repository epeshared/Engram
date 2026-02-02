#!/usr/bin/env bash
set -euo pipefail

# Sweep --non-engram-block-sim-ms over {1,5,10} with ENGRAM_SIM_SYNC=1
# and write the parsed [TIME] summary line to a CSV.
#
# Usage:
#   ./run_sim_sync_sweep_to_csv.sh [output.csv]
#
# Notes:
# - The demo prints one or more lines starting with: [TIME] forward_ms=...
#   This script takes the *last* such line from each run.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

OUT_CSV=${1:-sim_sync_sweep_results.csv}

if [[ ! -f "$OUT_CSV" ]]; then
  printf '%s\n' "timestamp,sim_ms,forward_ms,cpu_retrieve_tot,cpu_wait_cpu_future_ms,cpu_wait_sum_ms,cpu_wait_sum_pct" > "$OUT_CSV"
fi

SIM_VALUES=(1 5 10)

for SIM_MS in "${SIM_VALUES[@]}"; do
  echo "[RUN] sim_ms=$SIM_MS -> $OUT_CSV" >&2

  TMP_LOG=$(mktemp)
  # Ensure temp file is removed even if the command fails.
  trap 'rm -f "$TMP_LOG"' EXIT

  set +e
  ENGRAM_SIM_SYNC=1 python engram_offload_prefetch_demo.py \
    --non-engram-block-sim-ms "$SIM_MS" \
    --only-random --batch-size 30 --seq-len 100 --seed 123 --warmup-iters 1 \
    2>&1 | tee "$TMP_LOG"
  STATUS=${PIPESTATUS[0]}
  set -e

  if [[ $STATUS -ne 0 ]]; then
    echo "[ERROR] run failed for sim_ms=$SIM_MS (exit=$STATUS). See log above." >&2
    exit $STATUS
  fi

  TIME_LINE=$(grep -E '^\[TIME\] ' "$TMP_LOG" | tail -n 1 || true)
  if [[ -z "$TIME_LINE" ]]; then
    echo "[ERROR] No [TIME] line found for sim_ms=$SIM_MS" >&2
    exit 2
  fi

  python - <<'PY' "$OUT_CSV" "$SIM_MS" "$TIME_LINE"
import csv
import datetime as dt
import re
import sys

out_csv = sys.argv[1]
sim_ms = sys.argv[2]
line = sys.argv[3]

# Example line:
# [TIME] forward_ms=128.091 cpu_retrieve_tot=(layer1=7.322ms layer15=6.482ms) cpu_wait_cpu_future_ms=(layer1=4.053ms) cpu_wait_sum_ms=4.053 cpu_wait_sum_pct=3.16%

def m(pattern: str, s: str):
    mo = re.search(pattern, s)
    return mo.group(1) if mo else ""

forward_ms = m(r"forward_ms=([0-9.]+)", line)
retrieve_tot = m(r"cpu_retrieve_tot=\((.*?)\)", line)
wait_future = m(r"cpu_wait_cpu_future_ms=\((.*?)\)", line)
wait_sum_ms = m(r"cpu_wait_sum_ms=([0-9.]+)", line)
wait_sum_pct = m(r"cpu_wait_sum_pct=([0-9.]+)%", line)

ts = dt.datetime.now().isoformat(timespec="seconds")

with open(out_csv, "a", newline="") as f:
    w = csv.writer(f)
    w.writerow([ts, sim_ms, forward_ms, retrieve_tot, wait_future, wait_sum_ms, wait_sum_pct])
PY

  rm -f "$TMP_LOG"
  trap - EXIT

done

echo "[DONE] Wrote results to: $OUT_CSV" >&2
