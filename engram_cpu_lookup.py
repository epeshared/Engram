"""
Engram CPU-side lookup demo (standalone, config-driven + synthetic input).

This file keeps ONLY the CPU work needed to produce Engram lookup embeddings:

    hash_input_ids = torch.from_numpy(hash_mapping.hash(input_ids)[layer_id])
    embeddings     = multi_head_embedding(hash_input_ids).flatten(start_dim=-2)

Notes
- Hashing is done with NumPy on CPU.
- Embedding lookup is done with torch.nn.Embedding on CPU.
- No gating / conv / transformer blocks are included.
- Config is driven by EngramConfig / BackBoneConfig (no CLI for model hyperparams).

Usage (synthetic, recommended for fixed shapes)
    python engram_cpu_lookup.py \
            --batch-size 32 --seq-len 1024 --layer-id 1 --runs 100

Output
- Prints hash_id tensor shape and embedding tensor shape.
- Prints average time per run (hash + embedding lookup).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sympy import isprime
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex


def human_format(num: int) -> str:
    magnitude = 0
    n = float(num)
    while abs(n) >= 1000:
        magnitude += 1
        n /= 1000.0
    return "{}{}".format(
        ("{:f}".format(n).rstrip("0").rstrip(".")),
        ["", "K", "M", "B", "T"][magnitude],
    )


def bytes_to_mib(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


# =========================
# Configs
# =========================
@dataclass
class EngramConfig:
    # 0.67B-level demo config (you can override these constants in code).
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
    # per n-gram order: for max_ngram_size=3 => [2-gram, 3-gram]
    # engram_vocab_size: List[int] = field(default_factory=lambda: [129280 * 5, 129280 * 5])
    engram_vocab_size: List[int] = field(default_factory=lambda: [98_000_000, 98_000_000])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4  # not used in this lookup-only demo


@dataclass
class BackBoneConfig:
    hidden_size: int = 1536
    hc_mult: int = 4
    vocab_size: int = 129280
    num_layers: int = 30


# Instantiate configs here (edit these values to match your target setup)
engram_cfg = EngramConfig()
backbone_cfg = BackBoneConfig()


# =========================
# Core logic (unchanged)
# =========================
class CompressedTokenizer:
    """Builds a surjective mapping old_token_id -> normalized_token_id."""

    def __init__(self, tokenizer_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)

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

    def __len__(self) -> int:
        return int(self.num_new_token)

    def _build_lookup_table(self):
        old2new: Dict[int, int] = {}
        key2new: Dict[str, int] = {}
        new_tokens: List[str] = []

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

    def __call__(self, input_ids) -> np.ndarray:
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        out[pos_mask] = self.lookup_table[valid_ids]
        return out


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return int(candidate)
        candidate += 1


class NgramHashMapping:
    """Deterministic multi-head hashing for suffix n-grams."""

    def __init__(
        self,
        engram_vocab_size: List[int],
        max_ngram_size: int,
        n_head_per_ngram: int,
        layer_ids: List[int],
        tokenizer_name_or_path: str,
        pad_id: int | None,
        seed: int,
    ):
        self.vocab_size_per_ngram = list(engram_vocab_size)
        self.max_ngram_size = int(max_ngram_size)
        self.n_head_per_ngram = int(n_head_per_ngram)
        self.pad_id = pad_id
        self.layer_ids = list(layer_ids)

        if len(self.vocab_size_per_ngram) != (self.max_ngram_size - 1):
            raise ValueError(
                f"engram_vocab_size length must be max_ngram_size-1 "
                f"({self.max_ngram_size - 1}), got {len(self.vocab_size_per_ngram)}"
            )

        self.compressed_tokenizer = CompressedTokenizer(tokenizer_name_or_path=tokenizer_name_or_path)
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007

        self.layer_multipliers: Dict[int, np.ndarray] = {}
        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64)
            multipliers = r * 2 + 1
            self.layer_multipliers[int(layer_id)] = multipliers

        self.vocab_size_across_layers = self._calculate_vocab_size_across_layers()

    def _calculate_vocab_size_across_layers(self) -> Dict[int, List[List[int]]]:
        seen_primes: set[int] = set()
        vocab_size_across_layers: Dict[int, List[List[int]]] = {}

        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes: List[List[int]] = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes: List[int] = []

                vocab_size = int(self.vocab_size_per_ngram[ngram - 2])
                current_prime_search_start = vocab_size - 1

                for _ in range(self.n_head_per_ngram):
                    found_prime = find_next_prime(current_prime_search_start, seen_primes)
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime

                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[int(layer_id)] = all_ngram_vocab_sizes

        return vocab_size_across_layers

    def _get_ngram_hashes(self, input_ids: np.ndarray, layer_id: int) -> np.ndarray:
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape
        multipliers = self.layer_multipliers[int(layer_id)]

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

        all_hashes: List[np.ndarray] = []
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]

            mix = tokens[0] * multipliers[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])

            head_vocab_sizes = self.vocab_size_across_layers[int(layer_id)][n_gram_index]
            for j in range(self.n_head_per_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))

        return np.stack(all_hashes, axis=2)  # [B, T, heads_total]

    def hash(self, input_ids) -> Dict[int, np.ndarray]:
        input_ids = self.compressed_tokenizer(input_ids)
        out: Dict[int, np.ndarray] = {}
        for layer_id in self.layer_ids:
            out[int(layer_id)] = self._get_ngram_hashes(input_ids, layer_id=int(layer_id))
        return out


class MultiHeadEmbedding(nn.Module):
    """A multi-table embedding implemented as one big table with per-head offsets."""

    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + int(n))
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        total_N = int(sum(list_of_N))
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=int(D))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        shifted_input_ids = input_ids + self.offsets
        return self.embedding(shifted_input_ids)


def build_list_of_N(vocab_size_across_layers_for_one_layer: List[List[int]]) -> List[int]:
    return [x for heads in vocab_size_across_layers_for_one_layer for x in heads]


# =========================
# Main
# =========================
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layer-id", type=int, default=1)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=100)

    # Synthetic input
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1024)

    args = p.parse_args()

    # Config-driven layer ids
    layer_ids = list(engram_cfg.layer_ids)
    if args.layer_id not in layer_ids:
        raise SystemExit(f"--layer-id {args.layer_id} must be in engram_cfg.layer_ids={layer_ids}")

    hash_mapping = NgramHashMapping(
        engram_vocab_size=engram_cfg.engram_vocab_size,
        max_ngram_size=engram_cfg.max_ngram_size,
        n_head_per_ngram=engram_cfg.n_head_per_ngram,
        layer_ids=layer_ids,
        tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
        pad_id=engram_cfg.pad_id,
        seed=engram_cfg.seed,
    )

    list_of_N = build_list_of_N(hash_mapping.vocab_size_across_layers[args.layer_id])
    if engram_cfg.n_embed_per_ngram % engram_cfg.n_head_per_ngram != 0:
        raise SystemExit(
            f"n_embed_per_ngram({engram_cfg.n_embed_per_ngram}) must be divisible by "
            f"n_head_per_ngram({engram_cfg.n_head_per_ngram})"
        )
    D_head = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram

    mhe = MultiHeadEmbedding(list_of_N=list_of_N, D=D_head)
    mhe.eval()

    # Parameter & memory stats (CPU-side lookup model = embedding tables + small buffers)
    params_numel = int(sum(p.numel() for p in mhe.parameters()))
    params_bytes = int(sum(p.numel() * p.element_size() for p in mhe.parameters()))
    sd_bytes = 0
    for v in mhe.state_dict().values():
        if torch.is_tensor(v):
            sd_bytes += int(v.numel() * v.element_size())

    total_N = int(sum(list_of_N))
    print(
        f"[Init] layer_id={args.layer_id} tables={len(list_of_N)} total_N={human_format(total_N)} "
        f"D_head={D_head} params~={human_format(total_N * D_head)}"
    )
    print(
        f"[Params] numel={human_format(params_numel)} bytes={human_format(params_bytes)} "
        f"({bytes_to_mib(params_bytes):.2f} MiB) state_dict={human_format(sd_bytes)} ({bytes_to_mib(sd_bytes):.2f} MiB)"
    )
    print(
        f"[Cfg ] tokenizer={engram_cfg.tokenizer_name_or_path} "
        f"max_ngram={engram_cfg.max_ngram_size} heads/ngram={engram_cfg.n_head_per_ngram} "
        f"n_embed_per_ngram={engram_cfg.n_embed_per_ngram} layer_ids={engram_cfg.layer_ids} "
        f"engram_vocab_size={engram_cfg.engram_vocab_size} seed={engram_cfg.seed} pad_id={engram_cfg.pad_id}"
    )

    tok = AutoTokenizer.from_pretrained(engram_cfg.tokenizer_name_or_path, trust_remote_code=True)
    tok_vocab_size = len(tok)

    # Build input_ids (synthetic only)
    torch.manual_seed(int(engram_cfg.seed))
    B, T = int(args.batch_size), int(args.seq_len)
    input_ids_pt = torch.randint(
        low=0,
        high=tok_vocab_size,
        size=(B, T),
        dtype=torch.long,
    )

    input_ids_np = input_ids_pt.numpy()
    print(
        f"[Input] input_ids shape={tuple(input_ids_pt.shape)} seq_len={input_ids_pt.shape[1]} "
        f"synthetic=True"
    )

    def one_pass() -> tuple[torch.Tensor, torch.Tensor]:
        hash_input_ids = torch.from_numpy(hash_mapping.hash(input_ids_np)[args.layer_id])
        embeddings = mhe(hash_input_ids).flatten(start_dim=-2)
        return hash_input_ids, embeddings

    for _ in range(max(0, args.warmup)):
        _ = one_pass()

    t0 = time.perf_counter()
    for _ in range(max(1, args.runs)):
        hash_ids, emb = one_pass()
    t1 = time.perf_counter()

    dt_ms = (t1 - t0) * 1000.0 / max(1, args.runs)

    print(f"[Hash] hash_ids shape={tuple(hash_ids.shape)} dtype={hash_ids.dtype}")
    print(f"[Emb ] embeddings shape={tuple(emb.shape)} dtype={emb.dtype}")
    print(f"[Time] avg per run = {dt_ms:.3f} ms over runs={args.runs} (warmup={args.warmup})")


if __name__ == "__main__":
    main()
