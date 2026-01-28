# Benchmarks

This folder contains two benchmarks:

1) A **CPU-only micro-benchmark** that profiles `Engram.forward` step-by-step latency.
2) An **offload/prefetch demo benchmark** that runs a tiny Transformer-like backbone and overlaps CPU retrieval with GPU compute (when CUDA is available).

## Requirements

Pick one:

- CPU-only:
  ```bash
  pip install -r requirements-cpu.txt
  ```
- GPU (CUDA wheels):
  ```bash
  pip install -r requirements-gpu.txt
  ```
  Notes:
  - `requirements-gpu.txt` defaults to the CUDA 12.1 wheel index (`cu121`).
  - If you need CUDA 11.8, edit the `--index-url` in that file to `https://download.pytorch.org/whl/cu118`.

Offline notes:
- Many runs set `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1`.
- If Hugging Face assets are not cached locally, either disable offline mode or use the offload demo which includes a tokenizer fallback.

## 1) Engram forward micro-benchmark (CPU)

File: `benchmark_engram_forward.py`

What it does:
- Runs hashing (NumPy, CPU) + embedding lookup (Torch, CPU) + gating/value/short-conv (Torch, CPU)
- Prints a breakdown of avg time per sub-step.

Basic usage:

```bash
python3 benchmark_engram_forward.py
```

Typical runs:

```bash
python3 benchmark_engram_forward.py --warmup 1 --runs 5
python3 benchmark_engram_forward.py --batch-size 4 --runs 20
python3 benchmark_engram_forward.py --token-len 2048 --runs 10
```

Common flags:
- `--offline/--no-offline` (default: `--offline`)
- `--dtype {fp32,bf16,fp16}` (default: `bf16`)
- `--layer-id <int>` (default: `engram_cfg.layer_ids[0]`)
- `--batch-size <int>` (default: `1`)
- `--token-len <int>` (default: `0`)
- `--warmup <int>` (default: `1`)
- `--runs <int>` (default: `5`)

Note on `rope_parameters ...` messages:
- These come from `transformers` config validation while loading the tokenizer/config; they are not produced by the Engram code path.

Wrapper script: `run_engram_forward_benchmark.sh`

```bash
./run_engram_forward_benchmark.sh
./run_engram_forward_benchmark.sh --runs 10 --batch-size 4
```

Env vars:
- `ENGRAM_PYTHON`: Python executable (defaults to `python3` if available).

## 2) Offload + prefetch benchmark (GPU if available)

File: `engram_offload_prefetch_demo.py`

What it does:
- Schedules Engram hashing + embedding lookup on CPU threads.
- Runs a simplified backbone on the target device (GPU if CUDA is available).
- At Engram layers, pulls prefetched embeddings and fuses on device.
- Prints `[TIME] forward_ms=...` plus per-layer CPU retrieval totals and their share of `forward_ms`.

Tokenizer behavior:
- If offline mode is enabled and the HF tokenizer is not cached locally, this script falls back to a minimal `DummyTokenizer` so the benchmark can still run.

Run directly:

```bash
# Default text prompt
python3 engram_offload_prefetch_demo.py

# Add a random fixed-length batch (token IDs)
python3 engram_offload_prefetch_demo.py --random --batch-size 8 --seq-len 512 --seed 1

# Only run random batch
python3 engram_offload_prefetch_demo.py --only-random --batch-size 30 --seq-len 100 --seed 123
```

Timeline logging:
- `ENGRAM_TIMELINE=1` enables timeline prints.
- `ENGRAM_TIMELINE_VERBOSE=1` prints per-block markers.
- `ENGRAM_TIMELINE_LAYERS=1,15` filters timeline to specific layer IDs.

Wrapper script: `run_offload_benchmark.sh`

```bash
./run_offload_benchmark.sh
```