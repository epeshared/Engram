"""Benchmark Engram.forward step-by-step latency.

Usage examples:
    python benchmark_engram_forward.py --runs 5
    python benchmark_engram_forward.py --runs 20 --warmup 5    

Notes:
- The n-gram hashing is implemented with NumPy and runs on CPU.
- This benchmark is CPU-only.
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

# Import the demo definitions.
from engram_demo_v1 import Engram, MultiHeadEmbedding, engram_cfg, backbone_config
from transformers import AutoTokenizer


_BENCH_START = time.perf_counter()


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log(msg: str) -> None:
    dt_s = time.perf_counter() - _BENCH_START
    print(f"[{_now_ts()} +{dt_s:8.2f}s] {msg}", flush=True)


def _progress_every(total: int) -> int:
    if total <= 10:
        return 1
    return max(1, total // 10)


@dataclass
class BenchResult:
    output: torch.Tensor
    times_ms: Dict[str, float]


class StepTimer:
    def __init__(self):
        self.times_ms: Dict[str, float] = {}

    @contextmanager
    def measure(self, name: str):
        start_ns = time.perf_counter_ns()
        yield
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6

        self.times_ms[name] = self.times_ms.get(name, 0.0) + elapsed_ms


def _prepare_hidden_states_from_text(
    input_ids: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Mimic the demo: token embedding then expand to HC_MULT."""
    emb = torch.nn.Embedding(backbone_config.vocab_size, backbone_config.hidden_size).to(dtype=dtype)
    x = emb(input_ids)  # [B,L,D]
    x = x.unsqueeze(2).expand(-1, -1, backbone_config.hc_mult, -1).contiguous()  # [B,L,HC,D]
    return x



@torch.no_grad()
def engram_forward_profile(
    engram: Engram,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
) -> BenchResult:
    """Run Engram forward with a timing breakdown.

    input_ids must be on CPU because the hashing uses NumPy.
    hidden_states must be on CPU.
    """

    timer = StepTimer()

    # 1) Hashing (CPU / NumPy)
    with timer.measure("hash_mapping.hash (cpu)"):
        hash_dict = engram.hash_mapping.hash(input_ids)
        hash_np = hash_dict[engram.layer_id]  # [B,L,H]

    # 2) NumPy -> Torch
    with timer.measure("torch.from_numpy"):
        hash_t_cpu = torch.from_numpy(hash_np)
    hash_t = hash_t_cpu

    # 3) Embedding lookup + flatten
    with timer.measure("multi_head_embedding + flatten"):
        # Just use the component which might be virtual or real.
        out_raw = engram.multi_head_embedding(hash_t)
        embeddings = out_raw.flatten(start_dim=-2)

    # 4) Gates (accumulate across hc_mult)
    gates = []
    for hc_idx in range(backbone_config.hc_mult):
        with timer.measure("gating:key_proj+norm1"):
            key = engram.key_projs[hc_idx](embeddings)
            normed_key = engram.norm1[hc_idx](key)

        with timer.measure("gating:query_norm2"):
            query = hidden_states[:, :, hc_idx, :]
            normed_query = engram.norm2[hc_idx](query)

        with timer.measure("gating:dot+nonlinear"):
            gate = (normed_key * normed_query).sum(dim=-1) / (backbone_config.hidden_size ** 0.5)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)

    with timer.measure("gating:stack"):
        gates_t = torch.stack(gates, dim=2)

    # 5) Value projection + apply gates
    with timer.measure("value_proj"):
        v = engram.value_proj(embeddings)

    with timer.measure("apply_gates"):
        value = gates_t * v.unsqueeze(2)

    # 6) ShortConv and residual-like add
    with timer.measure("short_conv"):
        conv_out = engram.short_conv(value)

    with timer.measure("add"):
        out = value + conv_out

    return BenchResult(output=out, times_ms=timer.times_ms)


def _avg_times(results: Dict[str, float], runs: int) -> Dict[str, float]:
    return {k: v / max(1, runs) for k, v in results.items()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Avoid network calls to Hugging Face Hub (requires cached files).",
    )
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--layer-id", type=int, default=engram_cfg.layer_ids[0])
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (B). Uses the same text repeated B times.",
    )
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--runs", type=int, default=5)
    args = p.parse_args()

    log(
        "stage=begin "
        f"offline={int(args.offline)} dtype={args.dtype} layer_id={args.layer_id} "
        f"batch_size={args.batch_size} warmup={args.warmup} runs={args.runs} "
    )

    if args.offline:
        # Prevent Transformers/HF Hub from attempting HEAD/GET requests.
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    # Build Engram module.
    log("stage=build_engram_begin")
    
    engram = Engram(layer_id=args.layer_id).to(dtype=dtype)
    engram.eval()
    log("stage=build_engram_done")

    # Real text
    text = "Only Alexander the Great could tame the horse Bucephalus."

    log("stage=tokenizer.resolve_begin")
    tokenizer_name_or_path = engram_cfg.tokenizer_name_or_path
    log(f"stage=tokenizer.resolve_done path={tokenizer_name_or_path}")

    log("stage=tokenizer.load_begin")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=True,
        local_files_only=args.offline,
    )
    log("stage=tokenizer.load_done")

    log("stage=tokenize_begin")
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    if args.batch_size > 1:
        # Repeat the same sequence across the batch dimension: [1,L] -> [B,L]
        input_ids = input_ids.repeat(int(args.batch_size), 1)
    log(f"stage=tokenize_done input_ids.shape={tuple(input_ids.shape)}")

    log("stage=prepare_hidden_states_begin")
    hidden_states = _prepare_hidden_states_from_text(input_ids, dtype=dtype)
    log(f"stage=prepare_hidden_states_done hidden_states.shape={tuple(hidden_states.shape)}")

    # Warmup
    warmup_n = max(0, int(args.warmup))
    if warmup_n > 0:
        log(f"stage=warmup.begin n={warmup_n}")
    for i in range(warmup_n):
        if (i == 0) or (i == warmup_n - 1) or ((i + 1) % _progress_every(warmup_n) == 0):
            log(f"stage=warmup.step i={i+1}/{warmup_n}")       
        _ = engram_forward_profile(
            engram,
            hidden_states,
            input_ids,
        )
    if warmup_n > 0:
        log("stage=warmup.done")

    # Timed runs
    runs_n = max(1, int(args.runs))
    log(f"stage=runs.begin n={runs_n}")
    agg: Dict[str, float] = {}
    last_out: Optional[torch.Tensor] = None

    # os.system("emon -collect-edp > emon.dat &")
    for i in range(runs_n):
        if (i == 0) or (i == runs_n - 1) or ((i + 1) % _progress_every(runs_n) == 0):
            log(f"stage=runs.step i={i+1}/{runs_n}")        
        r = engram_forward_profile(
            engram,
            hidden_states,
            input_ids,
        )
        last_out = r.output
        for k, v in r.times_ms.items():
            agg[k] = agg.get(k, 0.0) + v

    avg = _avg_times(agg, runs=max(1, args.runs))
    log("stage=runs.done")
    # os.system("emon -stop")

    # Print results (in forward execution order)
    ordered_keys = [
        "hash_mapping.hash (cpu)",
        "torch.from_numpy",
        "multi_head_embedding + flatten",
        # gating happens inside a loop, but we report aggregated totals per sub-step
        "gating:key_proj+norm1",
        "gating:query_norm2",
        "gating:dot+nonlinear",
        "gating:stack",
        "value_proj",
        "apply_gates",
        "short_conv",
        "add",
    ]
    total = sum(avg.values())
    print("=== Engram.forward step timing (avg ms) ===")
    printed = set()
    for k in ordered_keys:
        if k in avg:
            print(f"{k:28s} {avg[k]:10.3f} ms")
            printed.add(k)
    # If anything else was measured, print it after the known steps.
    for k in avg.keys():
        if k not in printed:
            print(f"{k:28s} {avg[k]:10.3f} ms")
    print(f"{'TOTAL':28s} {total:10.3f} ms")
    if last_out is not None:
        print(f"output.shape={tuple(last_out.shape)} device={last_out.device} dtype={last_out.dtype}")

    log("stage=done")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
