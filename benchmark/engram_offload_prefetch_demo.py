"""
================================================================================
[Engram Architecture Demo Implementation]

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
"""

"""
pip install torch numpy transformers sympy
"""

## built-in
from typing import List
from dataclasses import dataclass, field
import os
import math
import time
import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Optional, Tuple
import threading

## third-party
try:
    from sympy import isprime as _sympy_isprime
except Exception:
    _sympy_isprime = None
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex 


_RETRIEVE_STATS_LOCK = threading.Lock()
_RETRIEVE_TOT_MS: Dict[int, float] = {}

_WAIT_STATS_LOCK = threading.Lock()
_WAIT_CPU_FUTURE_MS: Dict[int, float] = {}


def reset_retrieve_stats() -> None:
    with _RETRIEVE_STATS_LOCK:
        _RETRIEVE_TOT_MS.clear()


def reset_wait_stats() -> None:
    with _WAIT_STATS_LOCK:
        _WAIT_CPU_FUTURE_MS.clear()


def record_retrieve_tot_ms(layer_id: int, tot_ms: float) -> None:
    with _RETRIEVE_STATS_LOCK:
        _RETRIEVE_TOT_MS[int(layer_id)] = float(tot_ms)


def get_retrieve_tot_ms() -> Dict[int, float]:
    with _RETRIEVE_STATS_LOCK:
        return dict(_RETRIEVE_TOT_MS)


def record_wait_cpu_future_ms(layer_id: int, wait_ms: float) -> None:
    with _WAIT_STATS_LOCK:
        _WAIT_CPU_FUTURE_MS[int(layer_id)] = float(wait_ms)


def get_wait_cpu_future_ms() -> Dict[int, float]:
    with _WAIT_STATS_LOCK:
        return dict(_WAIT_CPU_FUTURE_MS)


def _make_random_input_ids(
    *,
    tokenizer,
    batch_size: int,
    seq_len: int,
    seed: int,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")

    # Engram hashing path expects IDs compatible with the tokenizer.
    # Also keep within backbone embedding vocab.
    vocab_limit = int(min(backbone_config.vocab_size, len(tokenizer)))
    if vocab_limit <= 0:
        raise RuntimeError(f"Invalid vocab_limit={vocab_limit}")

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randint(
        low=0,
        high=vocab_limit,
        size=(int(batch_size), int(seq_len)),
        dtype=torch.long,
        generator=g,
    )


def _is_probable_prime_u64(n: int) -> bool:
    """Deterministic Miller-Rabin for 64-bit integers."""
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False

    # Write n-1 = d * 2^s with d odd.
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    def check(a: int) -> bool:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                return True
        return False

    # Deterministic bases for testing 64-bit ints.
    # See: https://miller-rabin.appspot.com/ (common set for <2^64)
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        if not check(a):
            return False
    return True


def isprime(n: int) -> bool:
    if _sympy_isprime is not None:
        return bool(_sympy_isprime(n))
    return _is_probable_prime_u64(int(n))

def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

@dataclass
class EngramConfig:
    # 0.67B级别配置
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
    # engram_vocab_size: List[int] = field(default_factory=lambda: [98_000_000, 98_000_000])
    # engram_vocab_size: List[int] = field(default_factory=lambda: [65_333_333, 65_333_333])
    engram_vocab_size: List[int] = field(default_factory=lambda: [129280*5, 129280*5])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4
    
@dataclass
class BackBoneConfig:
    hidden_size: int = 1024
    # hidden_size: int = 1536
    hc_mult: int = 4
    vocab_size: int = 129280
    num_layers: int = 30
    
engram_cfg = EngramConfig()
backbone_config = BackBoneConfig()


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) in {"1", "true", "True", "yes", "YES"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


_TIMELINE_ENABLED = _env_flag("ENGRAM_TIMELINE", "0")
_TIMELINE_VERBOSE = _env_flag("ENGRAM_TIMELINE_VERBOSE", "0")
_TIMELINE_START_NS = time.perf_counter_ns()


def _parse_layer_filter(raw: str) -> Optional[set[int]]:
    raw = (raw or "").strip()
    if not raw or raw.lower() in {"all", "*"}:
        return None
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


_TIMELINE_LAYER_FILTER = _parse_layer_filter(os.environ.get("ENGRAM_TIMELINE_LAYERS", ""))


def tlog(msg: str, *, layer_id: Optional[int] = None) -> None:
    """Timeline logging with relative timestamps.

    Enable with: `ENGRAM_TIMELINE=1`.
    Optional filter: `ENGRAM_TIMELINE_LAYERS=1,15`.
    """
    if not _TIMELINE_ENABLED:
        return
    if layer_id is not None and _TIMELINE_LAYER_FILTER is not None and int(layer_id) not in _TIMELINE_LAYER_FILTER:
        return

    dt_ms = (time.perf_counter_ns() - _TIMELINE_START_NS) / 1e6
    thr = threading.current_thread().name
    prefix = f"[+{dt_ms:10.3f} ms][{thr}]"
    if layer_id is not None:
        print(f"{prefix}[layer={int(layer_id)}] {msg}", flush=True)
    else:
        print(f"{prefix} {msg}", flush=True)


def _format_bytes(n_bytes: int) -> str:
    n = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(n)}{unit}"
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n_bytes}B"


def _hf_offline() -> bool:
    return (
        os.environ.get("TRANSFORMERS_OFFLINE") in {"1", "true", "True"}
        or os.environ.get("HF_HUB_OFFLINE") in {"1", "true", "True"}
    )


class _DummyTokenizer:
    """Minimal tokenizer fallback for offline benchmarking.

    This avoids hard-failing when HF assets are not cached and networking is disabled.
    It is NOT equivalent to the real tokenizer; it's just enough to run the demo/benchmarks.
    """

    def __init__(self, vocab_size: int):
        self._vocab_size = int(vocab_size)

    def __len__(self) -> int:
        return self._vocab_size

    def decode(self, ids, skip_special_tokens: bool = False):
        # Return a deterministic string per id.
        if isinstance(ids, (list, tuple)) and len(ids) > 0:
            tid = int(ids[0])
        else:
            tid = int(ids)
        return f"tok{tid}"

    def convert_ids_to_tokens(self, tid: int):
        return f"tok{int(tid)}"

    def __call__(self, text: str, return_tensors: str = "pt"):
        # Simple whitespace split + stable hash to ids.
        toks = (text or "").strip().split()
        if not toks:
            toks = ["_"]
        ids = [abs(hash(t)) % self._vocab_size for t in toks]
        input_ids = torch.tensor([ids], dtype=torch.long)

        class _Out:
            def __init__(self, input_ids):
                self.input_ids = input_ids

        return _Out(input_ids)


def _load_tokenizer(tokenizer_name_or_path: str):
    """Load HF tokenizer with a robust offline fallback."""
    offline = _hf_offline()
    try:
        return AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            trust_remote_code=True,
            local_files_only=offline,
        )
    except Exception as e:
        if offline:
            # Offline + not cached: fall back to a minimal local tokenizer so benchmarks can run.
            print(
                f"[WARN] Tokenizer '{tokenizer_name_or_path}' not found in local cache and offline mode is enabled; "
                f"falling back to DummyTokenizer(vocab_size={backbone_config.vocab_size}).",
                flush=True,
            )
            return _DummyTokenizer(vocab_size=backbone_config.vocab_size)
        # If networking is blocked, retry in offline mode.
        return AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            trust_remote_code=True,
            local_files_only=True,
        )


# Optional: make the demo runnable on a single machine by shrinking the massive tables.
# This does not change the core dataflow; it just avoids allocating huge embeddings.
if _env_flag("ENGRAM_DEMO_SMALL", "0"):
    # Roughly "a few tokenizer vocab" per (n,k) head.
    engram_cfg.engram_vocab_size = [engram_cfg.engram_vocab_size[0] // 100, engram_cfg.engram_vocab_size[1] // 100]


def _simulate_block_compute(hidden_states: torch.Tensor, sim_ms: float, *, layer_id: Optional[int] = None) -> None:
    """Simulate compute for a Transformer block without Engram.

    - CPU: sleeps for sim_ms.
    - CUDA: enqueues a CUDA sleep kernel (no host-side blocking) when available.

    This is intended for benchmarking overlap between CPU retrieval and backbone compute.
    """
    sim_ms = float(sim_ms)
    if sim_ms <= 0.0:
        return

    dev = hidden_states.device
    if dev.type == "cuda" and torch.cuda.is_available():
        # Prefer torch.cuda._sleep when available because it enqueues on the current stream
        # and approximates a fixed duration without synchronizing.
        if hasattr(torch.cuda, "_sleep"):
            props = torch.cuda.get_device_properties(dev)
            # props.clock_rate is in kHz (cycles/ms), so cycles ~= ms * clock_rate.
            cycles = int(sim_ms * float(props.clock_rate))
            cycles = max(1, cycles)
            if _TIMELINE_VERBOSE:
                tlog(f"block.sim_compute.cuda_sleep.begin ms={sim_ms:.3f} cycles={cycles}", layer_id=layer_id)
            with torch.cuda.device(dev.index if dev.index is not None else torch.cuda.current_device()):
                torch.cuda._sleep(cycles)
            if _TIMELINE_VERBOSE:
                tlog("block.sim_compute.cuda_sleep.enqueued", layer_id=layer_id)
        else:
            # Fallback: run a small op. Duration is approximate.
            if _TIMELINE_VERBOSE:
                tlog(f"block.sim_compute.fallback_op.begin ms={sim_ms:.3f}", layer_id=layer_id)
            x = hidden_states
            for _ in range(max(1, int(sim_ms))):
                x = x + 1
            if _TIMELINE_VERBOSE:
                tlog("block.sim_compute.fallback_op.done", layer_id=layer_id)
        return

    # CPU fallback: host-side sleep.
    if _TIMELINE_VERBOSE:
        tlog(f"block.sim_compute.cpu_sleep.begin ms={sim_ms:.3f}", layer_id=layer_id)
    time.sleep(sim_ms / 1000.0)
    if _TIMELINE_VERBOSE:
        tlog("block.sim_compute.cpu_sleep.done", layer_id=layer_id)

class CompressedTokenizer:
    def __init__(
        self,
        tokenizer_name_or_path,
    ):
        self.tokenizer = _load_tokenizer(tokenizer_name_or_path)
        
        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])
        
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

import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=norm_eps) 
            for _ in range(hc_mult)
        ])
        
        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B,L,HC_MULT,D)
        Output: (B,L,HC_MULT,D)
        """
        B, T, G, C = x.shape
        
        assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))
        
        x_norm = torch.cat(normed_chunks, dim=-1)
        x_bct = x_norm.transpose(1, 2)    
        # with torch.no_grad(),  torch.autocast('cpu', dtype=torch.float16):    
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
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
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
                dtype=np.int64
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
                    found_prime = find_next_prime(
                        current_prime_search_start, 
                        seen_primes
                    )
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
            if k == 0: return x
            shifted = np.pad(x, ((0, 0), (k, 0)),
                                mode='constant', constant_values=self.pad_id)[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []
        
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = (tokens[0] * multipliers[0])
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


class EngramCpuRetriever:
    """CPU-side deterministic retrieval: hash (NumPy) + embedding lookup (Torch on CPU).

    This is the "conditional memory" part intended to be offloaded (paper Figure 2).
    """

    def __init__(
        self,
        layer_id: int,
        *,
        cpu_dtype: torch.dtype = torch.float16,
        pin_memory: bool = True,
    ):
        self.layer_id = int(layer_id)
        self.cpu_dtype = cpu_dtype
        self.pin_memory = pin_memory

        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size=engram_cfg.max_ngram_size,
            n_embed_per_ngram=engram_cfg.n_embed_per_ngram,
            n_head_per_ngram=engram_cfg.n_head_per_ngram,
            layer_ids=engram_cfg.layer_ids,
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            pad_id=engram_cfg.pad_id,
            seed=engram_cfg.seed,
        )

        list_of_N = [x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y]
        self.engram_hidden_size = (engram_cfg.max_ngram_size - 1) * engram_cfg.n_embed_per_ngram
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N=list_of_N,
            D=engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram,
        )
        self.multi_head_embedding.eval()

        total_N = sum(list_of_N)
        D = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram
        print(
            f"[EngramCpuRetriever]: layer={self.layer_id} tables={len(list_of_N)} "
            f"total_N={human_format(total_N)} D={D} params={human_format(total_N * D)}"
        )

    @torch.no_grad()
    def retrieve(self, input_ids_cpu: torch.Tensor) -> torch.Tensor:
        """Return CPU tensor [B,L,engram_hidden_size].

        input_ids_cpu must be CPU (NumPy hashing).
        """
        t0_ns = time.perf_counter_ns()
        tlog(f"cpu.retrieve.begin shape={tuple(input_ids_cpu.shape)}", layer_id=self.layer_id)
        if input_ids_cpu.device.type != "cpu":
            input_ids_cpu = input_ids_cpu.cpu()

        t_hash0 = time.perf_counter_ns()
        hash_np = self.hash_mapping.hash(input_ids_cpu)[self.layer_id]  # [B,L,H]
        t_hash1 = time.perf_counter_ns()

        hash_t = torch.from_numpy(hash_np)  # CPU long

        t_emb0 = time.perf_counter_ns()
        emb = self.multi_head_embedding(hash_t).flatten(start_dim=-2)  # [B,L,engram_hidden_size]
        t_emb1 = time.perf_counter_ns()

        emb = emb.to(dtype=self.cpu_dtype)
        if self.pin_memory:
            try:
                emb = emb.pin_memory()
            except RuntimeError:
                # Pinning can fail on some setups; fall back to regular CPU memory.
                pass

        t1_ns = time.perf_counter_ns()
        tot_ms = (t1_ns - t0_ns) / 1e6
        record_retrieve_tot_ms(self.layer_id, tot_ms)
        tlog(
            "cpu.retrieve.done "
            f"hash={(t_hash1 - t_hash0) / 1e6:.3f}ms "
            f"emb={(t_emb1 - t_emb0) / 1e6:.3f}ms "
            f"tot={tot_ms:.3f}ms "
            f"out={tuple(emb.shape)} pin={getattr(emb, 'is_pinned', lambda: False)()}",
            layer_id=self.layer_id,
        )
        return emb


class EngramGpuFusion(nn.Module):
    """GPU-side fusion: gating + value projection + ShortConv.

    Expects precomputed embeddings from the CPU retriever.
    """

    def __init__(self):
        super().__init__()
        engram_hidden_size = (engram_cfg.max_ngram_size - 1) * engram_cfg.n_embed_per_ngram

        self.short_conv = ShortConv(
            hidden_size=backbone_config.hidden_size,
            kernel_size=engram_cfg.kernel_size,
            dilation=engram_cfg.max_ngram_size,
            hc_mult=backbone_config.hc_mult,
        )

        self.value_proj = nn.Linear(engram_hidden_size, backbone_config.hidden_size)
        self.key_projs = nn.ModuleList(
            [nn.Linear(engram_hidden_size, backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)]
        )
        self.norm1 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])

    def forward(self, hidden_states: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """hidden_states: [B,L,HC,D] on GPU; embeddings: [B,L,engram_hidden_size] on GPU."""
        if embeddings.device != hidden_states.device:
            raise RuntimeError(
                f"EngramGpuFusion device mismatch: hidden_states={hidden_states.device} embeddings={embeddings.device}"
            )
        if embeddings.dtype != hidden_states.dtype:
            embeddings = embeddings.to(dtype=hidden_states.dtype)

        gates = []
        for hc_idx in range(backbone_config.hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            query = hidden_states[:, :, hc_idx, :]
            normed_query = self.norm2[hc_idx](query)
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(backbone_config.hidden_size)
            # Demo-only stabilization nonlinearity (not part of the paper's canonical equation).
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)

        gates = torch.stack(gates, dim=2)  # [B,L,HC,1]
        value = gates * self.value_proj(embeddings).unsqueeze(2)  # [B,L,HC,D]
        output = value + self.short_conv(value)
        return output


class EngramPrefetcher:
    """Asynchronously computes CPU retrieval and transfers to GPU with overlap.

    Pattern:
      - start(input_ids_cpu)
      - while GPU runs backbone, call get(layer_id) when needed
    """

    def __init__(
        self,
        cpu_retrievers: Dict[int, EngramCpuRetriever],
        device: torch.device,
        *,
        max_workers: Optional[int] = None,
    ):
        self.cpu_retrievers = cpu_retrievers
        self.device = device
        self.executor = ThreadPoolExecutor(max_workers=max_workers or max(1, min(8, len(cpu_retrievers))))
        self.futures: Dict[int, Future] = {}
        self.gpu_cache: Dict[int, torch.Tensor] = {}
        self.ready_events: Dict[int, torch.cuda.Event] = {}

        self._h2d_stream: Optional[torch.cuda.Stream] = None
        if self.device.type == "cuda":
            self._h2d_stream = torch.cuda.Stream(device=self.device)

    def start(self, input_ids_cpu: torch.Tensor) -> None:
        tlog(
            f"prefetch.start layers={sorted(list(self.cpu_retrievers.keys()))} "
            f"workers={getattr(self.executor, '_max_workers', 'unknown')} device={self.device}",
        )
        for layer_id, retriever in self.cpu_retrievers.items():
            if layer_id in self.futures:
                continue
            tlog("prefetch.schedule", layer_id=layer_id)
            self.futures[layer_id] = self.executor.submit(retriever.retrieve, input_ids_cpu)

    def get(self, layer_id: int) -> torch.Tensor:
        layer_id = int(layer_id)
        if layer_id in self.gpu_cache:
            tlog("prefetch.cache_hit", layer_id=layer_id)
            evt = self.ready_events.get(layer_id)
            if evt is not None:
                torch.cuda.current_stream(device=self.device).wait_event(evt)
            return self.gpu_cache[layer_id]

        if layer_id not in self.futures:
            raise KeyError(f"No prefetch scheduled for layer_id={layer_id}")

        fut = self.futures[layer_id]
        waited = not fut.done()
        if waited:
            tlog("prefetch.wait_cpu_future.begin", layer_id=layer_id)
            t_wait0 = time.perf_counter_ns()
        emb_cpu: torch.Tensor = fut.result()
        if waited:
            wait_ms = (time.perf_counter_ns() - t_wait0) / 1e6
            record_wait_cpu_future_ms(layer_id, wait_ms)
            tlog(f"prefetch.wait_cpu_future.done wait_ms={wait_ms:.3f}", layer_id=layer_id)
        else:
            # Do not record anything: missing key => zero blocking wait for this layer.
            pass

        nbytes = int(emb_cpu.numel() * emb_cpu.element_size())
        tlog(
            "prefetch.cpu_embedding "
            f"shape={tuple(emb_cpu.shape)} "
            f"dtype={str(emb_cpu.dtype).replace('torch.', '')} "
            f"bytes={_format_bytes(nbytes)}",
            layer_id=layer_id,
        )

        if self.device.type != "cuda":
            # CPU-only fallback: just return CPU embeddings.
            tlog(
                f"prefetch.cpu_only.return bytes={_format_bytes(nbytes)} shape={tuple(emb_cpu.shape)}",
                layer_id=layer_id,
            )
            self.gpu_cache[layer_id] = emb_cpu
            return emb_cpu

        assert self._h2d_stream is not None
        tlog(
            f"prefetch.h2d.begin bytes={_format_bytes(nbytes)} shape={tuple(emb_cpu.shape)}",
            layer_id=layer_id,
        )
        with torch.cuda.stream(self._h2d_stream):
            emb_gpu = emb_cpu.to(self.device, non_blocking=True)
            evt = torch.cuda.Event()
            evt.record(self._h2d_stream)

        tlog("prefetch.h2d.enqueued", layer_id=layer_id)

        # Make the current stream wait on H2D completion.
        torch.cuda.current_stream(device=self.device).wait_event(evt)
        tlog("prefetch.h2d.waited", layer_id=layer_id)
        self.gpu_cache[layer_id] = emb_gpu
        self.ready_events[layer_id] = evt
        return emb_gpu

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)
    
class Engram(nn.Module):
    """Convenience wrapper holding both CPU retriever and GPU fusion."""

    def __init__(self, layer_id: int, *, cpu_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.layer_id = int(layer_id)
        self.cpu = EngramCpuRetriever(layer_id=self.layer_id, cpu_dtype=cpu_dtype)
        self.gpu = EngramGpuFusion()

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        embeddings: Optional[torch.Tensor] = None,
        input_ids_cpu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if embeddings is None:
            if input_ids_cpu is None:
                raise ValueError("Must provide either embeddings (preferred) or input_ids_cpu")
            embeddings_cpu = self.cpu.retrieve(input_ids_cpu)
            embeddings = embeddings_cpu.to(hidden_states.device, non_blocking=True)
        return self.gpu(hidden_states, embeddings)

class TransformerBlock(nn.Module):
    def __init__(self,layer_id):
        super().__init__()
        self.attn = lambda x:x
        self.moe  = lambda x:x
        self.engram = None
        if layer_id in engram_cfg.layer_ids:
            self.engram = Engram(layer_id=layer_id)
        self.layer_id = int(layer_id)
        self.non_engram_sim_ms: float = 0.0
    
    def forward(self, input_ids: torch.Tensor, hidden_states: torch.Tensor, *, prefetcher: Optional[EngramPrefetcher] = None, input_ids_cpu: Optional[torch.Tensor] = None):
        if self.engram is not None:
            tlog("block.engram.enter", layer_id=self.layer_id)
            if prefetcher is not None:
                tlog("block.engram.get_embeddings", layer_id=self.layer_id)
                emb = prefetcher.get(self.layer_id)
                tlog(
                    f"block.engram.gpu_fuse.begin emb={tuple(emb.shape)}",
                    layer_id=self.layer_id,
                )
                hidden_states = self.engram.gpu(hidden_states, emb) + hidden_states
                tlog("block.engram.gpu_fuse.done", layer_id=self.layer_id)
            else:
                if input_ids_cpu is None:
                    raise ValueError("input_ids_cpu is required when prefetcher is not provided (CPU hashing).")
                tlog("block.engram.no_prefetch.cpu_retrieve", layer_id=self.layer_id)
                hidden_states = self.engram(hidden_states=hidden_states, input_ids_cpu=input_ids_cpu) + hidden_states
            tlog("block.engram.exit", layer_id=self.layer_id)
        else:
            # Simulate backbone compute cost for blocks without Engram.
            _simulate_block_compute(hidden_states, self.non_engram_sim_ms, layer_id=self.layer_id)

        if _TIMELINE_VERBOSE:
            tlog("block.attn", layer_id=self.layer_id)
        hidden_states = self.attn(hidden_states) + hidden_states

        if _TIMELINE_VERBOSE:
            tlog("block.moe", layer_id=self.layer_id)
        hidden_states = self.moe(hidden_states) + hidden_states
        return hidden_states


class DemoLLM(nn.Module):
    def __init__(
        self,
        *,
        cpu_retrieval_dtype: torch.dtype = torch.float16,
        non_engram_block_sim_ms: float = 0.0,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(backbone_config.vocab_size, backbone_config.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(layer_id=layer_id) for layer_id in range(backbone_config.num_layers)])
        # Ensure Engram CPU retrievers use a stable CPU dtype.
        for blk in self.blocks:
            if getattr(blk, "engram", None) is not None:
                blk.engram.cpu.cpu_dtype = cpu_retrieval_dtype
            else:
                blk.non_engram_sim_ms = float(non_engram_block_sim_ms)
        self.lm_head = nn.Linear(backbone_config.hidden_size, backbone_config.vocab_size)

    def build_prefetcher(self, device: torch.device) -> Optional[EngramPrefetcher]:
        if len(engram_cfg.layer_ids) == 0:
            return None
        cpu_retrievers: Dict[int, EngramCpuRetriever] = {}
        for blk in self.blocks:
            if blk.engram is not None:
                cpu_retrievers[blk.layer_id] = blk.engram.cpu
        if not cpu_retrievers:
            return None
        return EngramPrefetcher(cpu_retrievers=cpu_retrievers, device=device)

    def forward(self, input_ids: torch.Tensor, *, prefetcher: Optional[EngramPrefetcher] = None, input_ids_cpu: Optional[torch.Tensor] = None) -> torch.Tensor:
        tlog(f"model.forward.begin device={input_ids.device} shape={tuple(input_ids.shape)}")
        # Token embedding on device.
        hidden_states = self.tok_emb(input_ids)  # [B,L,D]
        # Mock hyper-connection (M branches).
        hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, backbone_config.hc_mult, -1).contiguous()  # [B,L,HC,D]

        for blk in self.blocks:
            if _TIMELINE_VERBOSE:
                tlog("model.block.begin", layer_id=blk.layer_id)
            hidden_states = blk(input_ids=input_ids, hidden_states=hidden_states, prefetcher=prefetcher, input_ids_cpu=input_ids_cpu)
            if _TIMELINE_VERBOSE:
                tlog("model.block.end", layer_id=blk.layer_id)

        # Mock hyper-connection collapse.
        hidden_states = hidden_states[:, :, 0, :]
        logits = self.lm_head(hidden_states)
        tlog(f"model.forward.done logits_shape={tuple(logits.shape)}")
        return logits

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        default="Only Alexander the Great could tame the horse Bucephalus.",
        help="Text prompt to run (default: demo sentence).",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Also run a random fixed-length batch (token IDs) for benchmarking.",
    )
    parser.add_argument(
        "--only-random",
        action="store_true",
        help="Run only the random batch; skip the fixed text prompt.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--non-engram-block-sim-ms",
        type=float,
        default=_env_float("ENGRAM_NON_ENGRAM_BLOCK_SIM_MS", 0.0),
        help="Simulated compute time (ms) for Transformer blocks that do NOT contain Engram. "
        "(env: ENGRAM_NON_ENGRAM_BLOCK_SIM_MS)",
    )
    args = parser.parse_args()

    tokenizer = _load_tokenizer(engram_cfg.tokenizer_name_or_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tlog(f"main.device {device}")
    if device.type != "cuda":
        print("[WARN] CUDA not available; running everything on CPU (no overlap demo).")

    # Build model: backbone on GPU, Engram retrieval stays on CPU.
    model = DemoLLM(
        cpu_retrieval_dtype=torch.float16,
        non_engram_block_sim_ms=args.non_engram_block_sim_ms,
    )
    model = model.to(device)
    model.eval()

    def run_case(case_name: str, input_ids_cpu: torch.Tensor, *, meta: str) -> None:
        reset_retrieve_stats()
        reset_wait_stats()
        print(f"\n=== CASE: {case_name} ===\n{meta}", flush=True)
        print(f"input_ids_cpu.shape={tuple(input_ids_cpu.shape)} dtype={input_ids_cpu.dtype}", flush=True)

        # Schedule CPU retrieval in background threads (paper-like deterministic prefetch).
        prefetcher = model.build_prefetcher(device=device)
        try:
            if prefetcher is not None:
                tlog("main.prefetcher.start")
                prefetcher.start(input_ids_cpu)

            input_ids = input_ids_cpu.to(device)

            with torch.no_grad():
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t_fwd0_ns = time.perf_counter_ns()
                logits = model(input_ids, prefetcher=prefetcher, input_ids_cpu=input_ids_cpu)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t_fwd1_ns = time.perf_counter_ns()
        finally:
            if prefetcher is not None:
                tlog("main.prefetcher.shutdown")
                prefetcher.shutdown()

        forward_ms = (t_fwd1_ns - t_fwd0_ns) / 1e6
        per_layer_retrieve = get_retrieve_tot_ms()
        per_layer_wait = get_wait_cpu_future_ms()

        # Only count *blocking* wait in the sum/pct, because non-waited layers are overlapped.
        sum_wait_ms = float(sum(per_layer_wait.values()))
        pct = (sum_wait_ms / forward_ms * 100.0) if forward_ms > 0 else float('nan')

        retrieve_parts = " ".join(
            [f"layer{lid}={per_layer_retrieve[lid]:.3f}ms" for lid in sorted(per_layer_retrieve.keys())]
        )
        wait_parts = " ".join([f"layer{lid}={per_layer_wait[lid]:.3f}ms" for lid in sorted(per_layer_wait.keys())])
        print(
            f"[TIME] forward_ms={forward_ms:.3f} cpu_retrieve_tot=({retrieve_parts}) "
            f"cpu_wait_cpu_future_ms=({wait_parts}) cpu_wait_sum_ms={sum_wait_ms:.3f} cpu_wait_sum_pct={pct:.2f}%",
            flush=True,
        )
        print("✅ Forward Complete!", flush=True)
        print(f"input_ids.shape={tuple(input_ids.shape)} logits.shape={tuple(logits.shape)}", flush=True)

    if not args.only_random:
        text = args.text
        input_ids_cpu_text = tokenizer(text, return_tensors="pt").input_ids
        run_case(
            "text",
            input_ids_cpu_text,
            meta=f"text={text!r}",
        )

    if args.random or args.only_random:
        input_ids_cpu_rand = _make_random_input_ids(
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            seed=args.seed,
        )
        preview = input_ids_cpu_rand[0, : min(16, input_ids_cpu_rand.shape[1])].tolist()
        run_case(
            "random",
            input_ids_cpu_rand,
            meta=(
                f"batch_size={args.batch_size} seq_len={args.seq_len} seed={args.seed} "
                f"vocab_limit={int(min(backbone_config.vocab_size, len(tokenizer)))} first_ids={preview}"
            ),
        )
            