"""vLLM generator on a chunk file: results in the same format as the transformers backend, for an F1 comparison, and
the wall time (runs in an environment with vLLM installed; not a test).

Usage: python scripts/vllm_check.py WEIGHTS_DIR chunks.jsonl out/results.jsonl --model Qwen/Qwen3-1.7B
       [--quantization fp8] [--gpu-mem 0.8]
"""
import argparse
import json
import time
from pathlib import Path

from relweave.extract import generator_settings
from relweave.formats import save_results
from relweave.schema import registered_schema
from relweave.vllm_generator import VllmExtractor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("weights", type=Path)
    ap.add_argument("chunks", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--gpu-mem", type=float, default=0.8)
    ap.add_argument("--model", required=True, help="full-precision base")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-batched-tokens", type=int, default=None, help="prefill step size (less activation memory)")
    ap.add_argument("--merged", action="store_true", help="--model is a merged model (merge_for_vllm.py)")
    a = ap.parse_args()
    s = generator_settings(a.weights / "generator")
    items = [(r["chunk_id"], r["text"]) for r in map(json.loads, a.chunks.read_text().splitlines()) if r]
    t0 = time.perf_counter()
    ex = VllmExtractor(a.model, None if a.merged else a.weights / "generator", s.get("max_new_tokens", 2500), s["format"],
                       s["compact_prompt"], registered_schema(s["schema"]), s["conditioned"],
                       gpu_memory_utilization=a.gpu_mem, quantization=a.quantization,
                       max_model_len=a.max_model_len, max_num_batched_tokens=a.max_batched_tokens)
    t1 = time.perf_counter()
    res = ex.extract_batch(items)
    t2 = time.perf_counter()
    save_results(res, a.out)
    print(f"load {t1 - t0:.0f} s; {len(items)} chunks in {t2 - t1:.0f} s = {len(items) / (t2 - t1) * 60:.1f} chunks/min; "
          f"{sum(r.output is None for r in res)} failed")


if __name__ == "__main__":
    main()
