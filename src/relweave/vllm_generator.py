"""The generator on vLLM: the same prompt, grammar and output as relweave.generator.QwenExtractor, with vLLM's
continuous batching and paged attention in place of transformers' generate (optional: pip install relweave[vllm]).

Differences from the transformers backend:
- stop on repeat is applied after decoding: once a relation line repeats one already written in the chunk, the rest
  of that sentence block is dropped (the transformers backend forbids those lines while decoding; the relations kept
  are the same, a loop only costs tokens up to the per-block cap of the grammar, and later blocks are conditioned on
  the loop);
- vLLM has no bitsandbytes 4-bit. Give it the merged model from scripts/merge_for_vllm.py (the dequantized 4-bit
  base with the adapter merged; adapter=None), or the adapter on the original full-precision base, which scores
  lower because the adapter learned against the 4-bit weights; quantization (e.g. "fp8") is applied by vLLM at load.
"""

from __future__ import annotations

import time
from pathlib import Path

from relweave.formats import FORMATS, STUDENT_FORMAT, ExtractionResult, chat_messages, finish, generation_prompt
from relweave.schema import BUSINESS, Schema


def drop_repeats(raw: str) -> str:
    """Stop on repeat after the fact: within a sentence block, lines after the first relation line that repeats one
    already written in the answer are dropped, up to the next block header."""
    seen, out, skipping = set(), [], False
    for line in raw.split("\n"):
        if line.startswith("S"):
            skipping = False
        elif skipping:
            continue
        elif line.startswith("R "):
            key = tuple(line.split()[1:4])
            skipping = key in seen
            seen.add(key)
        out.append(line)
    return "\n".join(out)


class VllmExtractor:
    """extract_batch(items) like QwenExtractor; vLLM schedules the batch itself (batch_size is ignored)."""

    def __init__(self, model_id: str, adapter: Path | None, max_new_tokens: int = 2500, fmt: str = STUDENT_FORMAT,
                 compact: bool = False, schema: Schema | None = None, conditioned: bool = False,
                 stop_on_repeat: bool = True, gpu_memory_utilization: float = 0.8, max_model_len: int = 8192,
                 lora_rank: int = 32, quantization: str | None = None):
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self.fmt, self.schema, self.compact, self.conditioned = fmt, schema or BUSINESS, compact, conditioned
        self.format = FORMATS[fmt].with_schema(self.schema)
        self.max_new_tokens, self.stop_on_repeat = max_new_tokens, stop_on_repeat
        self.name = f"qwen_finetuned_{fmt}_vllm"
        tok_source = adapter if adapter is not None and (Path(adapter) / "tokenizer_config.json").exists() else model_id
        self.tokenizer = AutoTokenizer.from_pretrained(str(tok_source))
        self.llm = LLM(model=model_id, quantization=quantization, enable_lora=adapter is not None, max_lora_rank=lora_rank,
                       gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                       enable_prefix_caching=True)
        self.lora = LoRARequest("generator", 1, str(adapter)) if adapter is not None else None

    def _params(self, text: str):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        grammar = self.format.ebnf_for(self.format.grammar_key(text))
        return SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens,
                              structured_outputs=StructuredOutputsParams(grammar=grammar))

    def extract_batch(self, items: list[tuple[str, str]], batch_size: int = 0) -> list[ExtractionResult]:
        prompts = [generation_prompt(self.tokenizer, chat_messages(text, self.fmt, compact=self.compact,
                                                                   schema=self.schema, conditioned=self.conditioned))
                   for _, text in items]
        t0 = time.perf_counter()
        outs = self.llm.generate(prompts, [self._params(text) for _, text in items], lora_request=self.lora,
                                 use_tqdm=False)
        per_chunk = (time.perf_counter() - t0) / max(len(items), 1)
        results = []
        for (cid, text), o in zip(items, outs):
            raw = o.outputs[0].text
            results.append(finish(cid, text, drop_repeats(raw) if self.stop_on_repeat else raw, per_chunk,
                                  fmt=self.fmt, schema=self.schema))
        print(f"\nprogress: extract {len(items)}/{len(items)} chunks, {per_chunk * len(items) / 60:.1f} min", flush=True)
        return results
