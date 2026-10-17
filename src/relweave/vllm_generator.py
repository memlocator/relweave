"""The generator on vLLM: the same prompt, grammar and output as relweave.generator.QwenExtractor, with vLLM's
continuous batching and paged attention in place of transformers' generate (optional: install vllm and vllm-bnb-plugin in the same environment; vLLM pins its own torch).

Differences from the transformers backend:
- stop on repeat is applied after decoding: once a relation line repeats one already written in the chunk, the rest
  of that sentence block is dropped (the transformers backend forbids those lines while decoding; the relations kept
  are the same, a loop only costs tokens up to the per-block cap of the grammar, and later blocks are conditioned on
  the loop);
- ngram=k drafts k tokens by prompt lookup (speculative decoding); it needs the merged model (no LoRA at run time).
- vLLM's own bitsandbytes 4-bit is not used. Give it the merged model from scripts/merge_for_vllm.py (the dequantized 4-bit
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


def _check_finished(outs) -> None:
    """vLLM reports a request it could not finish (out of memory while sampling, an engine error) as aborted, with
    whatever text it had; that text is a silently truncated answer, so stop instead."""
    bad = [o for o in outs if o.outputs and o.outputs[0].finish_reason not in ("stop", "length")]
    if bad:
        raise RuntimeError(f"vLLM did not finish {len(bad)} of {len(outs)} requests (finish reason "
                           f"{bad[0].outputs[0].finish_reason!r}); lower --vllm-memory or --vllm-max-seqs")


class VllmExtractor:
    """extract_batch(items) like QwenExtractor; vLLM schedules the batch itself (batch_size is ignored)."""

    def __init__(self, model_id: str, adapter: Path | None, max_new_tokens: int = 2500, fmt: str = STUDENT_FORMAT,
                 compact: bool = False, schema: Schema | None = None, conditioned: bool = False,
                 stop_on_repeat: bool = True, gpu_memory_utilization: float = 0.8, max_model_len: int = 8192,
                 lora_rank: int = 32, quantization: str | None = None, max_num_batched_tokens: int | None = None,
                 ngram: int = 0, max_num_seqs: int | None = None, typed_relations: bool = False):
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self.fmt, self.schema, self.compact, self.conditioned = fmt, schema or BUSINESS, compact, conditioned
        self.format = FORMATS[fmt].with_schema(self.schema)
        self.max_new_tokens, self.stop_on_repeat = max_new_tokens, stop_on_repeat
        self.typed_relations = typed_relations
        self.name = f"qwen_finetuned_{fmt}_vllm"
        tok_source = adapter if adapter is not None and (Path(adapter) / "tokenizer_config.json").exists() else model_id
        self.tokenizer = AutoTokenizer.from_pretrained(str(tok_source))
        self.llm = LLM(model=model_id, quantization=quantization, enable_lora=adapter is not None, max_lora_rank=lora_rank,
                       gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                       enable_prefix_caching=True,
                       # fewer parallel sequences: smaller CUDA graphs and sampler buffers (an 8 GB card)
                       **({"max_num_seqs": max_num_seqs} if max_num_seqs else {}),
                       # prompt-lookup speculative decoding: entity lines copy names from the text, so drafts taken
                       # from the prompt are often right; greedy output is unchanged (vLLM checks drafts against the
                       # grammar); not with a LoRA served at run time
                       **({"speculative_config": {"method": "ngram", "num_speculative_tokens": ngram,
                                                  "prompt_lookup_min": 2, "prompt_lookup_max": 4}} if ngram else {}),
                       **({"max_num_batched_tokens": max_num_batched_tokens} if max_num_batched_tokens else {}))
        self.lora = LoRARequest("generator", 1, str(adapter)) if adapter is not None else None

    def shutdown(self) -> None:
        """Stop the engine and give its GPU memory back (the pair head and tagger load next on a small card)."""
        import gc
        try:
            self.llm.llm_engine.engine_core.shutdown()
        except Exception:  # noqa: BLE001  older or newer vLLM: fall back to dropping the object
            pass
        self.llm = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _params(self, text: str, entities_only: bool = False, entity_types: dict[int, str] | None = None,
                max_tokens: int | None = None):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        grammar = self.format.ebnf_for(0 if entities_only else self.format.grammar_key(text), entity_types)
        return SamplingParams(temperature=0.0, max_tokens=max_tokens or self.max_new_tokens,
                              structured_outputs=StructuredOutputsParams(grammar=grammar))

    def _typed_relations(self, items, prompts, first, entities_only):
        """The relation lines of each chunk, written after its entity lines under a grammar compiled for exactly those
        entities (only types whose endpoint types fit, no self-loops), as the transformers backend's
        TypedRelationGrammar switches once the entity lines are complete. The second request continues from the
        first's own tokens (prefix caching keeps that cheap). Returns the full answers."""
        from vllm.inputs import TokensPrompt
        from relweave.generator import ENTITY_LINE
        texts = [o.outputs[0].text for o in first]
        # the entities-only request ends with the end-of-turn token; the continuation must not include it, or the
        # model would write its relation lines after its own turn has ended
        stop = {i for i in (self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
                if isinstance(i, int)}
        todo = [k for k, (cid, _, _) in enumerate(items) if cid not in entities_only]
        reqs, params = [], []
        for k in todo:
            types = {int(m.group(1)): m.group(2) for line in texts[k].split("\n") if (m := ENTITY_LINE.match(line))}
            out = list(first[k].outputs[0].token_ids)
            while out and out[-1] in stop:
                out.pop()
            ids = list(first[k].prompt_token_ids) + out
            reqs.append(TokensPrompt(prompt_token_ids=ids))
            params.append(self._params(items[k][1], entity_types=types,
                                       max_tokens=max(self.max_new_tokens - len(first[k].outputs[0].token_ids), 1)))
        outs = self.llm.generate(reqs, params, lora_request=self.lora, use_tqdm=False) if reqs else []
        _check_finished(outs)
        for k, o in zip(todo, outs):
            texts[k] = texts[k] + o.outputs[0].text
        return texts

    def extract_batch(self, items: list[tuple[str, str]], batch_size: int = 0,
                      entities_only: frozenset = frozenset()) -> list[ExtractionResult]:
        items = [(*it, None)[:3] for it in items]  # (key, text) or (key, text, known entity lines)
        prompts = [generation_prompt(self.tokenizer, chat_messages(text, self.fmt, compact=self.compact,
                                                                   schema=self.schema, conditioned=self.conditioned,
                                                                   known=known))
                   for _, text, known in items]
        t0 = time.perf_counter()
        typed = self.typed_relations
        outs = self.llm.generate(prompts, [self._params(text, typed or cid in entities_only) for cid, text, _ in items],
                                 lora_request=self.lora,
                                 use_tqdm=False)
        _check_finished(outs)
        raws = self._typed_relations(items, prompts, outs, entities_only) if typed else [o.outputs[0].text for o in outs]
        per_chunk = (time.perf_counter() - t0) / max(len(items), 1)
        results = []
        for (cid, text, _), raw in zip(items, raws):
            results.append(finish(cid, text, drop_repeats(raw) if self.stop_on_repeat else raw, per_chunk,
                                  fmt=self.fmt, schema=self.schema))
        print(f"\nprogress: extract {len(items)}/{len(items)} chunks, {per_chunk * len(items) / 60:.1f} min", flush=True)
        return results
