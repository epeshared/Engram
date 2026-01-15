"""\
================================================================================
[Engram Architecture Demo Implementation - BF16-capable]

DISCLAIMER:
1. Demo Purpose Only:
    This code is a demonstration version intended to illustrate the core logic and
    data flow of the Engram module.

2. Production Readiness:
    This implementation requires further optimization for actual production use
    (e.g., custom CUDA kernels, distributed training support).

3. Simplifications:
    Standard components (Normalization, Attention, MoE) and complex Hyper-connection
    mechanisms are omitted or mocked in this version to focus exclusively on the
    Engram module implementation.
================================================================================

Usage examples:
  python3 engram_demo_v1_bf16.py --dtype bf16
  python3 engram_demo_v1_bf16.py --dtype fp32
  python3 engram_demo_v1_bf16.py --dtype bf16 --offline

Notes:
- This demo is CPU-oriented.
- BF16 performance depends on CPU ISA support (e.g., AVX512_BF16/AMX).
"""

from __future__ import annotations

# pip install torch numpy transformers sympy tokenizers

## built-in

import argparse
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List

## third-party
import numpy as np
import torch
import torch.nn as nn
from sympy import isprime
from tokenizers import Regex, normalizers
from transformers import AutoTokenizer


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."),
        ["", "K", "M", "B", "T"][magnitude],
    )


@contextmanager
def _suppress_stdout():
    """Temporarily suppress stdout (useful for warmup)."""
    old_stdout = sys.stdout
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            yield
    finally:
        sys.stdout = old_stdout


@dataclass
class EngramConfig:
    # ~0.67B-ish scale demo config
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
    # engram_vocab_size: List[int] = field(default_factory=lambda: [98_000_000, 98_000_000])    
    engram_vocab_size: List[int] = field(default_factory=lambda: [129280 * 5, 129280 * 5])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    # layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    layer_ids: List[int] = field(default_factory=lambda: [1])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4


@dataclass
class BackBoneConfig:
    hidden_size: int = 1024
    hc_mult: int = 4
    vocab_size: int = 129280
    num_layers: int = 30


engram_cfg = EngramConfig()
backbone_config = BackBoneConfig()


class CompressedTokenizer:
    def __init__(
        self,
        tokenizer_name_or_path: str,
        offline: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            trust_remote_code=True,
            local_files_only=offline,
        )

        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence(
            [
                normalizers.NFKC(),
                normalizers.NFD(),
                normalizers.StripAccents(),
                normalizers.Lowercase(),
                normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                normalizers.Replace(Regex(r"^ $"), SENTINEL),
                normalizers.Strip(),
                normalizers.Replace(SENTINEL, " "),
            ]
        )

        self.lookup_table, self.num_new_token = self._build_lookup_table()

    def __len__(self):
        return self.num_new_token

    def _build_lookup_table(self):
        old2new = {}
        key2new = {}
        new_tokens = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)

            if "�" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid

        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)

    def _compress(self, input_ids):
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        out[pos_mask] = self.lookup_table[valid_ids]
        return out

    def __call__(self, input_ids):
        return self._compress(input_ids)


class ShortConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation

        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        self.norms = nn.ModuleList([nn.RMSNorm(hidden_size, eps=norm_eps) for _ in range(hc_mult)])

        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (B,L,HC_MULT,D) -> Output: (B,L,HC_MULT,D)"""

        B, T, G, C = x.shape
        assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))

        x_norm = torch.cat(normed_chunks, dim=-1)
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]

        if self.activation:
            y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()

        return y


def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


class NgramHashMapping:
    def __init__(
        self,
        engram_vocab_size,
        max_ngram_size,
        n_embed_per_ngram,
        n_head_per_ngram,
        layer_ids,
        tokenizer_name_or_path,
        pad_id,
        seed,
        offline: bool = False,
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path,
            offline=offline,
        )
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007

        self.layer_multipliers = {}

        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64,
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}

        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []

                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1

                for _ in range(num_head):
                    found_prime = find_next_prime(current_prime_search_start, seen_primes)
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime

                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes

        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        input_ids: np.ndarray,
        layer_id: int,
    ) -> np.ndarray:
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0:
                return x
            shifted = np.pad(
                x,
                ((0, 0), (k, 0)),
                mode="constant",
                constant_values=self.pad_id,
            )[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []

        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = tokens[0] * multipliers[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]

            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))

        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
        return hash_ids_for_all_layers


class MultiHeadEmbedding(nn.Module):
    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D

        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)

        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        shifted_input_ids = input_ids + self.offsets
        output = self.embedding(shifted_input_ids)
        return output


class Engram(nn.Module):
    def __init__(self, layer_id: int, offline: bool = False):
        super().__init__()
        self.layer_id = layer_id
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size=engram_cfg.max_ngram_size,
            n_embed_per_ngram=engram_cfg.n_embed_per_ngram,
            n_head_per_ngram=engram_cfg.n_head_per_ngram,
            layer_ids=engram_cfg.layer_ids,
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            pad_id=engram_cfg.pad_id,
            seed=engram_cfg.seed,
            offline=offline,
        )
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N=[x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y],
            D=engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram,
        )
        self.short_conv = ShortConv(
            hidden_size=backbone_config.hidden_size,
            kernel_size=engram_cfg.kernel_size,
            dilation=engram_cfg.max_ngram_size,
            hc_mult=backbone_config.hc_mult,
        )
        engram_hidden_size = (engram_cfg.max_ngram_size - 1) * engram_cfg.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, backbone_config.hidden_size)
        self.key_projs = nn.ModuleList(
            [nn.Linear(engram_hidden_size, backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)]
        )
        self.norm1 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])

        list_of_N = [x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y]
        total_N = sum(list_of_N)
        D = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram
        print(
            f"[Engram]: tables={len(list_of_N)} total_N={human_format(total_N)} D={D} params={human_format(total_N * D)}"
        )

    def forward(self, hidden_states, input_ids):
        """hidden_states: [B, L, HC_MULT, D], input_ids: [B, L]"""

        t0 = time.time()
        hash_input_ids = torch.from_numpy(self.hash_mapping.hash(input_ids)[self.layer_id])
        t1 = time.time()
        print(f"[Engram] hash_mapping: {(t1 - t0) * 1000:.3f} ms")

        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
        t2 = time.time()
        print(f"[Engram] embedding_lookup: {(t2 - t1) * 1000:.3f} ms")

        gates = []
        g_key_norm_ms = 0.0
        g_query_norm_ms = 0.0
        g_dot_nonlinear_ms = 0.0
        g_stack_ms = 0.0
        g_total_start_ns = time.perf_counter_ns()
        for hc_idx in range(backbone_config.hc_mult):
            s0 = time.perf_counter_ns()
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            g_key_norm_ms += (time.perf_counter_ns() - s0) / 1e6

            s1 = time.perf_counter_ns()
            query = hidden_states[:, :, hc_idx, :]
            normed_query = self.norm2[hc_idx](query)
            g_query_norm_ms += (time.perf_counter_ns() - s1) / 1e6

            s2 = time.perf_counter_ns()
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(backbone_config.hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)
            g_dot_nonlinear_ms += (time.perf_counter_ns() - s2) / 1e6

        s3 = time.perf_counter_ns()
        gates = torch.stack(gates, dim=2)
        g_stack_ms = (time.perf_counter_ns() - s3) / 1e6
        g_total_ms = (time.perf_counter_ns() - g_total_start_ns) / 1e6

        t3 = time.time()
        print(f"[Engram] gating:key_proj+norm1: {g_key_norm_ms:.3f} ms")
        print(f"[Engram] gating:query_norm2: {g_query_norm_ms:.3f} ms")
        print(f"[Engram] gating:dot+nonlinear: {g_dot_nonlinear_ms:.3f} ms")
        print(f"[Engram] gating:stack: {g_stack_ms:.3f} ms")
        print(f"[Engram] gating_total: {g_total_ms:.3f} ms")

        value = gates * self.value_proj(embeddings).unsqueeze(2)
        t4 = time.time()
        print(f"[Engram] value_proj_and_apply: {(t4 - t3) * 1000:.3f} ms")        
        output = value + self.short_conv(value)
        t5 = time.time()
        print(f"[Engram] short_conv_and_add: {(t5 - t4) * 1000:.3f} ms")
        print(f"[Engram] total_forward: {(t5 - t0) * 1000:.3f} ms")

        return output


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, offline: bool = False):
        super().__init__()
        self.attn = lambda x: x
        self.moe = lambda x: x
        self.engram = None
        if layer_id in engram_cfg.layer_ids:
            self.engram = Engram(layer_id=layer_id, offline=offline)

    def forward(self, input_ids, hidden_states):
        if self.engram is not None:
            hidden_states = self.engram(hidden_states=hidden_states, input_ids=input_ids) + hidden_states
        hidden_states = self.attn(hidden_states) + hidden_states
        hidden_states = self.moe(hidden_states) + hidden_states
        return hidden_states


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of silent warmup forwards before the measured run.",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of measured forward passes to execute (with prints).",
    )
    p.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Avoid network calls to Hugging Face Hub (requires cached files).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
        if not torch.cuda.is_available():
            print("[warn] fp16 on CPU may be slower/unsupported for some ops.")
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    # model = nn.ModuleList(
    #     [
    #         nn.Embedding(backbone_config.vocab_size, backbone_config.hidden_size),
    #         *[TransformerBlock(layer_id=layer_id, offline=args.offline) for layer_id in range(backbone_config.num_layers)],
    #         nn.Linear(backbone_config.hidden_size, backbone_config.vocab_size),
    #     ]
    # )

    LLM = [
        nn.Embedding(backbone_config.vocab_size,backbone_config.hidden_size),
        *[TransformerBlock(layer_id=layer_id) for layer_id in range(backbone_config.num_layers)],
        nn.Linear(backbone_config.hidden_size, backbone_config.vocab_size)
    ]    

    # model.to(dtype=dtype)

    text = "Only Alexander the Great could tame the horse Bucephalus."
    tokenizer = AutoTokenizer.from_pretrained(
        engram_cfg.tokenizer_name_or_path,
        trust_remote_code=True,
        local_files_only=args.offline,
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    def _forward_once() -> torch.Tensor:
        hidden_states = None
        output = None
        with torch.no_grad(),  torch.autocast('cpu', dtype=torch.bfloat16):
            for idx, layer in enumerate(LLM):
                if idx == 0:
                    # hidden_states = layer(input_ids)
                    hidden_states = LLM[0](input_ids)                    
                    # mock hyper-connection
                    hidden_states = (
                        hidden_states.unsqueeze(2)
                        .expand(-1, -1, backbone_config.hc_mult, -1)
                        .contiguous()
                    )
                elif idx == len(LLM) - 1:
                    # mock hyper-connection
                    hidden_states = hidden_states[:, :, 0, :]
                    output = layer(hidden_states)
                else:
                    hidden_states = layer(input_ids=input_ids, hidden_states=hidden_states)

        assert output is not None
        return output

    warmup_n = max(0, int(args.warmup))
    if warmup_n > 0:
        with torch.no_grad(), _suppress_stdout():
            for _ in range(warmup_n):
                _ = _forward_once()

    runs_n = max(1, int(args.runs))
    with torch.no_grad():
        output = None
        for _ in range(runs_n):
            output = _forward_once()
        assert output is not None

    print("✅ Forward Complete!")
    print(f"dtype={dtype} offline={args.offline}")
    print(f"{input_ids.shape=}\n{output.shape=}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
