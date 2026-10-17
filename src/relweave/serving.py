"""How relweave runs the generator on vLLM: weight format, memory share and parallelism, chosen from the machine.

One rule set, used by `relweave run --engine vllm` and Extractor(engine="vllm"): the most capable format that fits
with room for a useful key/value cache, the most parallel settings that fit, each overridable.

Formats (all score the same within noise; measured 2026-10-17, whitepaper Appendix C):
  bf16  the merged generator at the weights' root, full precision: needs about 8 GB for the 4B weights
  fp8   the same merged weights quantized by vLLM at load (weight-only fp8 on Ampere, native on Ada and newer)
  4bit  the 4-bit base the adapters were trained on, with the generator's adapter as a LoRA (vllm-bnb-plugin):
        the training format exactly, the smallest weights, the most room for parallel sequences on a small card
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_ENGINE = "auto"  # vLLM when it is installed (same results, about 3x faster end to end), else transformers

# GiB of GPU memory per format for the 4B generator: weights, then vLLM's own overhead (CUDA graphs, activations,
# sampler) at full parallelism, then the least key/value cache worth running with (about 10 full-length sequences)
WEIGHTS_GB = {"bf16": 8.1, "fp8": 4.4, "4bit": 2.7}
OVERHEAD_GB = 1.5
MIN_KV_GB = 1.5
FORMATS = ("bf16", "fp8", "4bit")
HEAD_GB = 5.5  # the pair head and tagger next to vLLM: the 4-bit base, two adapters, probe activations
MAX_NUM_SEQS = 256  # vLLM's default: as parallel as the cache allows
SMALL_CARD_NUM_SEQS = 32  # fewer CUDA graphs and sampler buffers when memory is tight
# per parallel sequence, memory vLLM's start-up profile does not count: float32 logits and grammar bitmasks over the
# 152k-token vocabulary at sampling (256 sequences at 0.95 of a 24 GB card ran out by 0.3 GB, 2026-10-17)
SAMPLER_GB_PER_SEQ = 0.003
MAX_SHARE = 0.92


@dataclass
class VllmPlan:
    format: str
    gpu_memory_utilization: float
    max_num_seqs: int
    max_model_len: int
    reason: str


def plan(free_gb: float, total_gb: float, fmt: str = "auto", memory: float | None = None,
         max_num_seqs: int | None = None, max_model_len: int = 3072, reserve_gb: float = 0.5) -> VllmPlan:
    """The vLLM settings for a card with free_gb of total_gb free. fmt "auto": the first of bf16, fp8, 4bit whose
    weights, overhead and minimal cache fit; memory: the share of the card vLLM may take (default: all that is free
    but reserve_gb); max_num_seqs: parallel sequences (default: vLLM's 256, or 32 on a tight card)."""
    budget = (memory * total_gb) if memory else max(free_gb - reserve_gb, 0.0)
    if fmt == "auto":
        fits = [f for f in FORMATS if WEIGHTS_GB[f] + OVERHEAD_GB + MIN_KV_GB <= budget]
        if not fits:
            raise RuntimeError(f"{budget:.1f} GiB for vLLM: too little even for the 4-bit generator "
                               f"({WEIGHTS_GB['4bit'] + OVERHEAD_GB + MIN_KV_GB:.1f} GiB); use engine='transformers'")
        fmt = fits[0]
        reason = f"{fmt}: the most precise format that fits {budget:.1f} GiB with room for a cache"
    elif fmt in FORMATS:
        reason = f"{fmt}: as asked"
    else:
        raise ValueError(f"vLLM format {fmt!r}: one of auto, {', '.join(FORMATS)}")
    kv = budget - WEIGHTS_GB[fmt] - OVERHEAD_GB
    seqs = max_num_seqs or (MAX_NUM_SEQS if kv >= 4 else SMALL_CARD_NUM_SEQS)
    share = (budget - seqs * SAMPLER_GB_PER_SEQ) / total_gb
    return VllmPlan(fmt, round(min(share, MAX_SHARE), 2), seqs, max_model_len, reason)


def resident_share(free_gb: float, total_gb: float) -> float | None:
    """vLLM's memory share when the pair head can stay loaded next to it (a big card: generate and score without
    unloading), or None when it cannot (generate everything, stop vLLM, then score)."""
    budget = free_gb - HEAD_GB - 0.5
    if budget < WEIGHTS_GB["4bit"] + OVERHEAD_GB + MIN_KV_GB:
        return None
    return round(budget / total_gb, 2)


def gpu_memory_gb() -> tuple[float, float]:
    """(free, total) GiB of the current CUDA device."""
    import torch
    free, total = torch.cuda.mem_get_info()
    return free / 2**30, total / 2**30


def merged_model_dir(weights_dir):
    """The merged generator (a plain model at the weights' root, as published), or None."""
    from pathlib import Path
    d = Path(weights_dir)
    return d if (d / "config.json").exists() and any(d.glob("*.safetensors")) else None
