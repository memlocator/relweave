"""Generator throughput on one text file: chunks per minute and peak GPU memory at several batch sizes.

Needs a CUDA GPU; not a test. Usage:
    uv run python scripts/bench_throughput.py document.txt [--sizes 1,2,4] [--weights DIR_OR_HUB_ID] [--max-words 200]
"""
import argparse
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("text", type=Path)
    ap.add_argument("--sizes", default="1,2,4")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--max-words", type=int, default=200)
    a = ap.parse_args()

    import torch

    from relweave import Extractor
    from relweave.chunk import chunk

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    ex = Extractor(weights=a.weights, sequential=True, max_words=a.max_words)
    chunks = chunk(a.text.read_text(), a.max_words)
    items = [(f"0:{c.index}", c.text) for c in chunks]
    ce = ex._ce
    ce.load_generator()
    print(f"{len(items)} chunks of at most {a.max_words} words")
    for size in (int(s) for s in a.sizes.split(",")):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = ce.generate_many(items, batch_size=size)
        minutes = (time.perf_counter() - t0) / 60
        failed = sum(isinstance(r, Exception) for r in out.values())
        print(f"batch {size} (ran at {ce.last_batch_size}): {len(items) / minutes:.1f} chunks/min, "
              f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MB, {failed} failed", flush=True)
    ce.unload_generator()


if __name__ == "__main__":
    main()
