"""The generator: a Qwen3 model (base plus a LoRA adapter) writing the `sentences` format under a decoding grammar.

QwenExtractor loads the model and generates per chunk or in batches; TypedRelationGrammar is the logits processor
for controlled constrained decoding of the `sentences` format: type-aware relations, stop on repeat, and a stop
bias.

Generation starts under the chunk's normal grammar. When a row writes its first sentence
header ("S<n>"), the entity lines are complete, so their types are known: the row switches to
a grammar compiled for exactly those entities, in which a relation line may only use an edge
whose endpoint types fit (relweave.formats.SentenceFormat.ebnf_for with entity_types). Relations
that would be dropped as illegal on parsing can no longer be written.

stop_on_repeat: once a relation line repeats one already written in the chunk, the next line may
only open a new sentence block or end the answer (greedy decoding otherwise loops on a line
until the per-sentence cap). stop_bias: added to the logits of "new block" and "end" at every
point where the model chooses between another relation line and moving on.

grounded: the typed grammar additionally restricts block S<k> to relations between entities the model
listed a surface for that occurs in sentence k (relweave.formats.SentenceFormat.ebnf_for with grounded):
a relation can only be written under a sentence that mentions both its entities.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from relweave.formats import FORMATS, STUDENT_FORMAT, ExtractionResult, chat_messages, finish, generation_prompt
from relweave.schema import BUSINESS, Schema

ENTITY_LINE = re.compile(r"^E(\d+) (\w+): ")
ENTITY_SURFACES = re.compile(r"^E(\d+) \w+: (.*)$")


def mentions_by_sentence(answer: str, chunk_text: str, fmt) -> dict[int, set[int]]:
    """Sentence number (1-based) -> numbers of the entities whose listed surfaces occur in that sentence,
    from the entity lines written so far."""
    from relweave.formats import find_all
    bounds = fmt.sentences(chunk_text)
    out: dict[int, set[int]] = {}
    for line in answer.split("\n"):
        if m := ENTITY_SURFACES.match(line):
            n = int(m.group(1))
            for surface in (x.strip() for x in m.group(2).split("|")):
                for a, b in find_all(chunk_text, surface) if surface else []:
                    for k, (s0, s1) in enumerate(bounds, 1):
                        if s0 <= a and b <= s1:
                            out.setdefault(k, set()).add(n)
    return out


class TypedRelationGrammar:
    """Logits processor for a batch: one grammar matcher per row, switched to a typed relation
    grammar once that row's entity lines are complete."""

    def __init__(self, compiler, fmt, texts: list[str], tokenizer, vocab_size: int, typed: bool = True,
                 stop_on_repeat: bool = False, stop_bias: float = 0.0, grounded: bool = False):
        import xgrammar as xgr
        self.xgr, self.compiler, self.fmt, self.tok = xgr, compiler, fmt, tokenizer
        self.typed, self.stop_on_repeat, self.stop_bias, self.grounded = typed, stop_on_repeat, stop_bias, grounded
        self.stop_ids = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer("S", add_special_tokens=False)["input_ids"][0]]
        self.rows = [{"m": xgr.GrammarMatcher(fmt.compile(compiler, t)), "ids": [], "typed": not typed,
                      "n": fmt.grammar_key(t), "all_ids": [], "seen": set(), "force_stop": False, "text": t}
                     for t in texts]
        self.bitmask = xgr.allocate_token_bitmask(len(texts), vocab_size)
        self.first = True
        self.switched = self.fallbacks = 0

    def _switch(self, row: dict) -> None:
        text = self.tok.decode(row["ids"])
        types = {int(m.group(1)): m.group(2) for line in text.split("\n") if (m := ENTITY_LINE.match(line))}
        grounded = mentions_by_sentence(text, row["text"], self.fmt) if self.grounded else None
        matcher = self.xgr.GrammarMatcher(self.compiler.compile_grammar(self.fmt.ebnf_for(row["n"], types, grounded)))
        if matcher.accept_string(text):
            row["m"] = matcher
            self.switched += 1
        else:  # should not happen; keep the untyped grammar rather than fail the row
            self.fallbacks += 1
        row["typed"] = True

    def __call__(self, input_ids, scores):
        if not self.first:
            for i, row in enumerate(self.rows):
                if row["m"].is_terminated():
                    continue
                token = int(input_ids[i, -1])
                row["m"].accept_token(token)
                if self.stop_on_repeat or self.stop_bias:
                    self._track(row, token)
                if not row["typed"]:
                    row["ids"].append(token)
                    text = self.tok.decode(row["ids"])
                    if text.startswith("S") or "\nS" in text:
                        self._switch(row)
        self.first = False
        for i, row in enumerate(self.rows):
            if not row["m"].is_terminated():  # a finished row keeps its last mask (stop token only)
                row["m"].fill_next_token_bitmask(self.bitmask, i)
        self.xgr.apply_token_bitmask_inplace(scores, self.bitmask.to(scores.device))
        for i, row in enumerate(self.rows):
            if row["m"].is_terminated() or not row.get("after_relation"):
                continue
            if row["force_stop"]:  # only a new block or the end may follow
                keep = scores[i, self.stop_ids].clone()
                scores[i, :] = float("-inf")
                scores[i, self.stop_ids] = keep
            elif self.stop_bias:
                scores[i, self.stop_ids] += self.stop_bias
        return scores

    def _track(self, row: dict, token: int) -> None:
        """Follow the row's lines: note completed relation lines and repeats."""
        row["all_ids"].append(token)
        row["after_relation"] = False
        text = self.tok.decode(row["all_ids"][-80:])
        if not text.endswith("\n"):
            return
        line = text.rstrip("\n").rsplit("\n", 1)[-1]
        if line.startswith("R "):
            key = tuple(line.split()[1:4])
            if key in row["seen"] and self.stop_on_repeat:
                row["force_stop"] = True
            row["seen"].add(key)
            row["after_relation"] = True
        elif line.startswith("S"):
            row["force_stop"] = False


DEFAULT_MODEL = "Qwen/Qwen3-1.7B"


class QwenExtractor:
    def __init__(self, model_id: str = DEFAULT_MODEL, adapter: Path | None = None,
                 constrained: bool = True, max_new_tokens: int = 2500, load_in_4bit: bool = False,
                 repetition_penalty: float = 1.0, fmt: str = STUDENT_FORMAT, compact: bool = False,
                 typed_relations: bool = False, stop_on_repeat: bool = False, stop_bias: float = 0.0,
                 grounded: bool = False, temperature: float = 0.0,
                 schema: Schema | None = None, conditioned: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.fmt = fmt
        self.schema = schema or BUSINESS  # must match how the adapter was trained (train_summary.json "schema")
        self.conditioned = conditioned  # the schema is rendered into the prompt (train_summary.json "conditioned")
        self.format = FORMATS[fmt].with_schema(self.schema)
        self.temperature = temperature  # 0: greedy; > 0: sampling (top-p 0.95), e.g. for merging several entity lists
        self.compact = compact  # must match how the adapter was trained (see train_summary.json)
        self.typed_relations = typed_relations and fmt in ("sentences", "counted")  # see TypedRelationGrammar
        self.stop_on_repeat, self.stop_bias = stop_on_repeat, stop_bias
        self.grounded = grounded and fmt == "sentences"  # relations only between entities mentioned in their sentence
        self.typed_relations = self.typed_relations or self.grounded  # grounding is part of the typed grammar
        self.name = ("qwen_finetuned" if adapter else "qwen_zeroshot") + ("" if fmt == "json" else f"_{fmt}")
        # a fine-tuned model is prompted with the tokenizer (and chat template) it was trained with,
        # saved next to the adapter; the base model's own template can differ
        tok_source = adapter if adapter is not None and (Path(adapter) / "tokenizer_config.json").exists() else model_id
        self.tokenizer = AutoTokenizer.from_pretrained(str(tok_source))
        self.tokenizer.padding_side = "left"  # batched generation appends to the right
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs: dict = {"dtype": torch.float16, "device_map": "cuda"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if adapter is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter))
            # merging into 4-bit weights loses most of the adapter; keep it separate on quantized bases
            if not getattr(self.model.config, "quantization_config", None):
                self.model = self.model.merge_and_unload()
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.constrained = constrained
        self._compiler = None
        self._grammars: dict = {}
        if constrained:
            import xgrammar as xgr
            vocab_size = getattr(self.model.config, "vocab_size", None) or len(self.tokenizer)
            info = xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=vocab_size)
            self._compiler = xgr.GrammarCompiler(info)

    def _grammar_for(self, chunk_text: str):
        """Compiled grammar for this chunk; formats with per-chunk grammars are cached by key."""
        if self._compiler is None:
            return None
        fmt = self.format
        key = fmt.grammar_key(chunk_text)
        if key not in self._grammars:
            self._grammars[key] = fmt.compile(self._compiler, chunk_text)
        return self._grammars[key]

    def _messages(self, text: str) -> list[dict[str, str]]:
        return chat_messages(text, self.fmt, compact=self.compact, schema=self.schema, conditioned=self.conditioned)

    def extract(self, chunk_id: str, chunk_text: str) -> ExtractionResult:
        import torch

        prompt = generation_prompt(self.tokenizer, self._messages(chunk_text))
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=False, temperature=None,
                          top_p=None, top_k=None, pad_token_id=self.tokenizer.eos_token_id,
                          repetition_penalty=self.repetition_penalty)
        if self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature, top_p=0.95)
        grammar = self._grammar_for(chunk_text)
        if grammar is not None:
            import xgrammar as xgr
            gen_kwargs["logits_processor"] = [xgr.contrib.hf.LogitsProcessor(grammar)]
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        seconds = time.perf_counter() - t0
        raw = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return finish(chunk_id, chunk_text, raw, seconds, fmt=self.fmt, schema=self.schema)

    def extract_batch(self, items: list[tuple[str, str]], batch_size: int = 8) -> list[ExtractionResult]:
        """Extract many chunks with batched, grammar-constrained generation.

        Chunks are grouped by prompt length to limit padding; each row gets its own
        compiled grammar. Results come back in the input order. Per-chunk `seconds` is
        the batch time divided by the batch size.
        """
        import torch

        prompts = {cid: generation_prompt(self.tokenizer, self._messages(text))
                   for cid, text in items}
        order = sorted(items, key=lambda it: len(prompts[it[0]]))
        t_start = time.perf_counter()
        results: dict[str, ExtractionResult] = {}
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            enc = self.tokenizer([prompts[cid] for cid, _ in batch], return_tensors="pt", padding=True).to(self.model.device)
            gen_kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=False, temperature=None, top_p=None,
                              top_k=None, pad_token_id=self.tokenizer.pad_token_id,
                              repetition_penalty=self.repetition_penalty)
            if self.temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=self.temperature, top_p=0.95)
            grammars = [self._grammar_for(text) for _, text in batch]
            controlled = self.typed_relations or self.stop_on_repeat or self.stop_bias
            if controlled and self.fmt in ("sentences", "counted") and self._compiler is not None:
                gen_kwargs["logits_processor"] = [TypedRelationGrammar(
                    self._compiler, self.format, [text for _, text in batch], self.tokenizer,
                    self.model.config.vocab_size, typed=self.typed_relations,
                    stop_on_repeat=self.stop_on_repeat, stop_bias=self.stop_bias, grounded=self.grounded)]
            elif all(g is not None for g in grammars):
                import xgrammar as xgr
                gen_kwargs["logits_processor"] = [xgr.contrib.hf.LogitsProcessor(grammars)]
            t0 = time.perf_counter()
            with torch.no_grad():
                out = self.model.generate(**enc, **gen_kwargs)
            per_chunk = (time.perf_counter() - t0) / len(batch)
            prompt_len = enc["input_ids"].shape[1]
            for row, (cid, text) in enumerate(batch):
                raw = self.tokenizer.decode(out[row][prompt_len:], skip_special_tokens=True)
                results[cid] = finish(cid, text, raw, per_chunk, fmt=self.fmt, schema=self.schema)
            # one flushed line per batch: remote jobs have no progress bar, and a silent job
            # cannot be told apart from a hung one
            print(f"\nprogress: extract {len(results)}/{len(items)} chunks, "
                  f"{(time.perf_counter() - t_start) / 60:.1f} min", flush=True)
        return [results[cid] for cid, _ in items]
