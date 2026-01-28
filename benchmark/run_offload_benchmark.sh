#!/usr/bin/env bash
set -euo pipefail

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# turn on ENGRAM_TIMELINE
# TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 ENGRAM_TIMELINE=1  python engram_offload_prefetch_demo.py 

# random input
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 ENGRAM_TIMELINE=1  python engram_offload_prefetch_demo.py --only-random --batch-size 30 --seq-len 100 --seed 123