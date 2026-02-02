#!/usr/bin/env bash
set -euo pipefail

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# turn on ENGRAM_TIMELINE
# TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 ENGRAM_TIMELINE=1  python engram_offload_prefetch_demo.py 

# random input
# ENGRAM_TIMELINE_VERBOSE=1 
# ENGRAM_TIMELINE=1 ENGRAM_SIM_SYNC=1 python engram_offload_prefetch_demo.py \
#   --non-engram-block-sim-ms 1 \
#   --only-random --batch-size 30 --seq-len 100 --seed 123 \
#   --profile-breakdown  --profile-breakdown-blocks --warmup-iters 1



ENGRAM_SIM_SYNC=1 python engram_offload_prefetch_demo.py \
  --non-engram-block-sim-ms 1 \
  --only-random --batch-size 30 --seq-len 100 --seed 123 --warmup-iters 1