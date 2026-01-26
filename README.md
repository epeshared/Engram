# Engram Forward Benchmark

This repo includes a CPU-only micro-benchmark for profiling `Engram.forward` step-by-step latency.

## Requirements

- Python 3.8+
- Dependencies:

```bash
pip install torch numpy transformers
```

Notes:
- The benchmark is CPU-only.
- Hashing is implemented with NumPy and runs on CPU.
- By default, the benchmark runs in offline mode (no Hugging Face network calls). If you don't have the tokenizer cached locally, pass `--no-offline`.

## 1) benchmark_engram_forward.py

### Basic usage

```bash
python benchmark_engram_forward.py
```

Typical runs:

```bash
python benchmark_engram_forward.py --warmup 1 --runs 5
python benchmark_engram_forward.py --batch-size 4 --runs 20
python benchmark_engram_forward.py --token-len 2048 --runs 10
```

### Arguments

All arguments are optional (they have defaults):

- `--offline` / `--no-offline`
  - Default: `--offline`
  - Meaning: avoid network calls to Hugging Face Hub (requires cached tokenizer files).
- `--dtype {fp32,bf16,fp16}`
  - Default: `bf16`
  - Meaning: dtype used to build the Engram module and dummy hidden states.
- `--layer-id <int>`
  - Default: `engram_cfg.layer_ids[0]` (from `engram_demo_v1.py`)
  - Meaning: which layer-id to benchmark.
- `--batch-size <int>`
  - Default: `1`
  - Meaning: batch size $B$. Uses the same text repeated $B$ times.
- `--token-len <int>`
  - Default: `0`
  - Meaning: if `>0`, generate random token ids of exactly this length and benchmark on those ids.
- `--warmup <int>`
  - Default: `1`
  - Meaning: warmup iterations (not included in reported averages).
- `--runs <int>`
  - Default: `5`
  - Meaning: timed iterations used to compute the reported average.

## 2) run_benchmark.sh

This script is a thin wrapper around `benchmark_engram_forward.py`.

### Basic usage

```bash
bash run_benchmark.sh
```

It supplies default flags:
- `--warmup 1`
- `--runs 1`

You can override or add any arguments supported by `benchmark_engram_forward.py`:

```bash
bash run_benchmark.sh --runs 10 --batch-size 4
bash run_benchmark.sh --no-offline --runs 5
```

### Environment variables

- `ENGRAM_PYTHON`: Python executable to use (defaults to `python3` if available, else `python`).

Example:

```bash
ENGRAM_PYTHON=/path/to/python bash run_benchmark.sh --runs 5
```