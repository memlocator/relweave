"""One chunk to a ChunkGraph with the model path: the generator (base model + LoRA adapter, relweave.generator), the
pair head over the generator's own entity lines (relweave.head), and the union (relweave.union: keep a written
relation unless the head scores it at or below cut, add head relations above margin).

Weights: a Hugging Face Hub repository or a local directory with generator/ and head/ (resolve_weights); either
part can be overridden by a local directory (a training run with adapter/ and train_summary.json works too).
The generator and the head are two LoRA adapters on the same 4-bit base, loaded as two models (the generator fp16,
the head bf16 compute), one after the other when `sequential` or both resident.
"""

from __future__ import annotations

import gc
import json
import re
from itertools import combinations
from pathlib import Path

from relweave.chunk import Chunk
from relweave.records import ChunkEntity, ChunkGraph, ChunkRelation, Mention
from relweave.schema import BUSINESS, MAX_ENTITIES_PER_CHUNK, MAX_RELATIONS_PER_CHUNK, Schema, load_schema, registered_schema

DEFAULT_WEIGHTS = "chrullis/relweave-4b-base"
DEFAULT_MODEL = "unsloth/qwen3-4b-unsloth-bnb-4bit"
BATCH_CHOICES = (8, 4, 2, 1)


def resolve_weights(spec: str | Path) -> Path:
    """A local directory as it is; anything else is a Hub repository id, downloaded (cached) with
    huggingface_hub.snapshot_download (only the parts relweave reads: generator/, head/, schema.json; the merged
    model at a repository's root is for HTTP servers such as vLLM)."""
    if Path(spec).is_dir():
        return Path(spec)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(spec), allow_patterns=["generator/*", "head/*", "schema.json", "README.md"]))


def generator_settings(generator_dir: str | Path) -> dict:
    """The generator's prompt and schema settings: relweave_config.json in the directory (exported layout), else a
    training run's train_summary.json (next to adapter/, or beside the adapter files)."""
    d = Path(generator_dir)
    if (d / "relweave_config.json").exists():
        return json.loads((d / "relweave_config.json").read_text())
    for p in (d / "train_summary.json", d.parent / "train_summary.json"):
        if p.exists():
            s = json.loads(p.read_text())
            return {"schema": s.get("schema", BUSINESS.name), "format": s.get("format", "sentences"),
                    "compact_prompt": s.get("compact_prompt", False), "conditioned": s.get("conditioned", False)}
    raise FileNotFoundError(f"{d}: no relweave_config.json or train_summary.json saying how the generator was trained")


def kv_bytes_per_token(config) -> int:
    """Bytes of key/value cache one token costs across all layers (fp16)."""
    heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return config.num_hidden_layers * 2 * heads * head_dim * 2


def pick_batch(free_bytes: int, row_bytes: int, choices=BATCH_CHOICES, safety: float = 2.0) -> int:
    """The largest batch size in choices whose rows (row_bytes each, times a safety factor for activations and the
    allocator) fit in free_bytes; 1 when none does."""
    return next((b for b in sorted(choices, reverse=True) if b * row_bytes * safety <= free_bytes), 1)


HEADING_WORDS = 12
BULLET = re.compile(r"\s*([-*\u2022\u2013]|\d+[.)]|[a-z][.)])\s")


def heading_spans(text: str) -> list[tuple[int, int]]:
    """Lines that are headings, not text: one short line without terminal punctuation, not bulleted or numbered,
    not next to another short line (a list, a run of captions), and followed by more text."""
    lines = [(m.start(), m.end()) for m in re.finditer(r"[^\n]+", text)]
    short = lambda a, b: 0 < len(text[a:b].split()) <= HEADING_WORDS  # noqa: E731
    adjacent = lambda i, j: 0 <= j < len(lines) and text[lines[min(i, j)][1]:lines[max(i, j)][0]] == "\n"  # noqa: E731
    out = []
    for k, (a, b) in enumerate(lines[:-1]):
        if (short(a, b) and not text[a:b].rstrip().endswith((".", "!", "?", ":", ";", ",", '"', "'", ")", "\u201d"))
                and not BULLET.match(text[a:b])
                and not any(adjacent(k, j) and short(*lines[j]) for j in (k - 1, k + 1))):
            out.append((a, b))
    return out


def locate(chunk: Chunk, mention: str) -> list[Mention]:
    """Every whole-word occurrence of `mention` in the chunk's text, at document offsets, except occurrences on a
    heading line when the mention also occurs elsewhere (a heading is not a place the sentence-level model reads
    as text); a mention that occurs only on headings keeps those offsets. One Mention with start None when the
    string does not occur at all (the generator wrote it verbatim, so this is rare)."""
    hits = [m.start() for m in re.finditer(r"(?<!\w)" + re.escape(mention) + r"(?!\w)", chunk.text)]
    heads = heading_spans(chunk.text)
    found = [h for h in hits if not any(a <= h < b for a, b in heads)] or hits
    return [Mention(mention, chunk.start + h) for h in found] or [Mention(mention, None)]


def _place(chunk: Chunk, e) -> list[Mention]:
    """The entity's mentions with offsets. A name occurs wherever it is written; a pronoun or description is pinned
    to the one occurrence that follows the entity's names most closely (another entity's "It" must not land on this one)."""
    from relweave.merge import is_generic
    names = [m for t in e.mentions if not is_generic(t) for m in locate(chunk, t)]
    anchors = [m.start for m in names if m.start is not None]
    out = list(names)
    for t in e.mentions:
        if is_generic(t):
            hits = [m for m in locate(chunk, t) if m.start is not None]
            if len(hits) > 1:
                hits = [min(hits, key=lambda m: min((m.start - a if m.start >= a else 10**9 + a - m.start for a in anchors),
                                                    default=m.start))]  # the first one after a name
            out += hits or locate(chunk, t)
    order = {t: i for i, t in enumerate(e.mentions)}
    return sorted(out, key=lambda m: (order[m.text], m.start if m.start is not None else -1))


def to_chunk_graph(chunk: Chunk, output, written: set, scores: dict[str, float]) -> ChunkGraph:
    """The union result `output` (an ExtractionOutput) as a ChunkGraph. `written`: (type, source, target) the
    generator wrote; `scores`: head margins keyed "TYPE i j" (entity numbers)."""
    g = ChunkGraph(chunk.index, start=chunk.start, end=chunk.end)
    for e in output.entities:
        mentions = _place(chunk, e)
        if mentions:
            g.entities.append(ChunkEntity(e.id, str(e.type), e.name, mentions, dict(e.attributes)))
    kept = {e.id for e in g.entities}
    for r in output.relations:
        if r.source not in kept or r.target not in kept:
            continue
        s = scores.get(f"{r.type} {r.source[1:]} {r.target[1:]}")
        at = chunk.text.find(r.evidence) if r.evidence else -1  # the generator's evidence sentence, verbatim
        g.relations.append(ChunkRelation(r.type, r.source, r.target, str(r.modality), dict(r.attributes),
                                         None if s is None else round(s, 2),
                                         "generator" if (r.type, r.source, r.target) in written else "head",
                                         (chunk.start + at, chunk.start + at + len(r.evidence)) if at >= 0 else None))
    if len(output.entities) >= MAX_ENTITIES_PER_CHUNK:
        g.warnings.append(f"chunk {chunk.index}: hit the {MAX_ENTITIES_PER_CHUNK}-entity cap, entities may be missing")
    return g


def grouped(items: list, group: int, probe, on_oom) -> list:
    """probe(items[k:k+group]) over all items; on CUDA OOM, on_oom() and the group size is halved (down to 1,
    then the error propagates). Returns the probe results in order."""
    import torch
    parts, k = [], 0
    while k < len(items):
        oom = False
        try:
            parts.append(probe(items[k:k + group]))
        except torch.OutOfMemoryError:
            if group == 1:
                raise
            oom = True  # free memory after the except block: while it runs, the traceback still holds the failed pass
        if oom:
            on_oom()
            group //= 2
            continue
        k += group
    return parts


_CAP: float | None = None  # the process-wide memory fraction set so far


def fraction(free: int, total: int, reserve_mb: int, previous: float | None, own: int = 0) -> float:
    """The per-process memory fraction that leaves reserve_mb free; `own` is what this process already holds (its
    torch reserved memory, counted again as available to it). Never above one set earlier in this process."""
    if free < reserve_mb * 2**20:
        raise RuntimeError(f"only {free // 2**20} MB of GPU memory free, less than the {reserve_mb} MB reserve")
    f = (free + own - reserve_mb * 2**20) / total
    return f if previous is None else min(previous, f)


def cap_memory(reserve_mb: int) -> None:
    global _CAP
    import torch
    free, _ = torch.cuda.mem_get_info()
    new = fraction(free, torch.cuda.get_device_properties(0).total_memory, reserve_mb, _CAP, torch.cuda.memory_reserved())
    if _CAP is None or new < _CAP:
        torch.cuda.set_per_process_memory_fraction(new)
        _CAP = new


def _free_gpu() -> None:
    import torch
    gc.collect()
    torch.cuda.empty_cache()


class ChunkExtractor:
    """run(chunk) -> ChunkGraph. generate_many()/pair_scores()/combine() are the phases, so all chunks can be
    generated, the generator unloaded, and then scored (Extractor does this when sequential).

    generator, head: directories (exported layout or training runs); head None: generation only. schema: a Schema, a registered name or
    "path.py:NAME"; None takes the schema the generator was trained for. cut, margin: union settings; None takes
    them from the head's config (else relweave.union.CUT and MARGIN). generator_url: generate on an HTTP server (vLLM
serving the merged generator, relweave.remote) instead of loading the generator; the generator directory then only
supplies the prompt settings and tokenizer."""

    generator_url: str | None = None

    def __init__(self, generator: str | Path, head: str | Path | None, model: str | None = None, schema=None,
                 device: str = "cuda", reserve_mb: int = 500, group: int = 20, cut: float | None = None,
                 margin: float | None = None, max_new_tokens: int | None = None, generator_url: str | None = None):
        from relweave import union as un
        if device != "cuda":
            raise ValueError("relweave needs a CUDA device ('cuda'): the base model is 4-bit")
        self.generator_dir, self.head_dir = Path(generator), Path(head) if head is not None else None
        settings = generator_settings(self.generator_dir)
        trained = settings.get("schema", BUSINESS.name)
        # a name from a config file only resolves against registered schemas; code is loaded only from the caller's spec
        self.schema: Schema = registered_schema(trained) if schema is None else load_schema(schema)
        if trained != self.schema.name:
            raise ValueError(f"generator was trained on schema {trained!r}, not {self.schema.name!r}")
        self.head_config = self._read_head_config()
        if self.head_config.get("schema", BUSINESS.name) != self.schema.name:  # the head's label set is its schema
            raise ValueError(f"pair head was trained on schema {self.head_config.get('schema')!r}, not {self.schema.name!r}")
        union_cfg = self.head_config.get("union", {})
        self.cut = cut if cut is not None else union_cfg.get("cut", un.CUT)
        self.margin = margin if margin is not None else union_cfg.get("margin", un.MARGIN)
        # the base the adapters were trained on (exported config), unless the caller names one
        self.model, self.group = model or settings.get("base_model", DEFAULT_MODEL), group
        self.generator_url = generator_url
        self.compact, self.conditioned = settings.get("compact_prompt", False), settings.get("conditioned", False)
        self.max_new_tokens = max_new_tokens or settings.get("max_new_tokens", 2500)
        cap_memory(reserve_mb)
        self._gen = self._head = None

    def _read_head_config(self) -> dict:
        d = self.head_dir
        if d is None:  # generation only
            return {"schema": self.schema.name}
        if (d / "head_config.json").exists():
            return json.loads((d / "head_config.json").read_text())
        summary = d / "train_summary.json"
        return {"schema": json.loads(summary.read_text()).get("schema", BUSINESS.name) if summary.exists() else BUSINESS.name}

    # -- loading -----------------------------------------------------------------------------------------------

    def load_generator(self) -> None:
        if self._gen is None and self.generator_url:
            from relweave.remote import RemoteExtractor
            self._gen = RemoteExtractor(self.generator_url, self._tokenizer_dir(), fmt="sentences", compact=self.compact,
                                        schema=self.schema, conditioned=self.conditioned,
                                        max_new_tokens=self.max_new_tokens)
        if self._gen is None:
            from relweave.generator import QwenExtractor
            from relweave.head import adapter_dir
            self._gen = QwenExtractor(model_id=self.model, adapter=adapter_dir(self.generator_dir), constrained=True,
                                      fmt="sentences", compact=self.compact, stop_on_repeat=True, schema=self.schema,
                                      conditioned=self.conditioned, max_new_tokens=self.max_new_tokens)

    def _tokenizer_dir(self) -> Path:
        from relweave.head import adapter_dir
        d = adapter_dir(self.generator_dir)
        return d if (d / "tokenizer_config.json").exists() else Path(self.model)

    def unload_generator(self) -> None:
        self._gen = None
        _free_gpu()

    def load_head(self) -> None:
        if self.head_dir is None:
            raise ValueError("no pair head directory given")
        if self._head is None:
            from relweave.head import LAYERS, adapter_dir, labels, load_head, load_model
            head, config = load_head(self.head_dir)
            layers = tuple(config.get("layers", LAYERS))
            tok, model = load_model(self.model, adapter_dir(self.head_dir))
            if head.net[1].in_features != len(layers) * model.config.hidden_size:
                raise ValueError("the head does not match the base model's hidden size and the layers read")
            if head.net[4].out_features != len(labels(self.schema)) + 1:
                raise ValueError(f"the head has {head.net[4].out_features - 1} labels, schema {self.schema.name} "
                                 f"has {len(labels(self.schema))}")
            self._head = (tok, model, head, layers)

    def unload_head(self) -> None:
        self._head = None
        _free_gpu()

    # -- phases ------------------------------------------------------------------------------------------------

    def generate(self, chunk: Chunk):
        """The generator's ExtractionResult for one chunk (a batch of one)."""
        return self.generate_many([(f"chunk{chunk.index}", chunk.text)], batch_size=1)[f"chunk{chunk.index}"]

    def auto_batch(self, items: list[tuple[str, str]]) -> int:
        """The largest of 1/2/4/8 whose key/value cache fits the free GPU memory, estimated from the longest prompt
        (in tokens) plus an answer as long again (capped at max_new_tokens)."""
        import torch
        from relweave.formats import generation_prompt
        if not items:
            return 1
        gen = self._gen
        longest = max(len(gen.tokenizer(generation_prompt(gen.tokenizer, gen._messages(t)))["input_ids"])
                      for _, t in items)
        row = (longest + min(longest, gen.max_new_tokens)) * kv_bytes_per_token(gen.model.config)
        free = torch.cuda.mem_get_info()[0]
        if _CAP is not None:
            budget = _CAP * torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()
            free = min(free, int(budget))
        return pick_batch(free, row)

    def generate_many(self, items: list[tuple[str, str]], batch_size: int | str = "auto") -> dict:
        """key -> ExtractionResult (or the exception of a chunk that failed) for (key, text) items, in batches of
        similar length. On CUDA OOM the batch is retried at half the size (down to 1); a chunk that fails alone keeps
        its exception; no chunk is lost."""
        import torch
        self.load_generator()
        if self.generator_url:  # the server batches; one call per 256 chunks keeps a failure's retry small
            b = 256
        else:
            b = self.auto_batch(items) if batch_size == "auto" else int(batch_size)
        order = sorted(items, key=lambda kv: len(kv[1]))
        out: dict = {}
        k = 0
        while k < len(order):
            batch = order[k:k + b]
            try:
                results = self._gen.extract_batch(batch, batch_size=len(batch))
            except Exception as e:  # noqa: BLE001  one bad chunk must not lose the others (OOM is one too)
                err = e
            else:
                err = None
            if err is None:
                out.update({key: r for (key, _), r in zip(batch, results)})
                k += len(batch)
            elif isinstance(err, torch.OutOfMemoryError) and b > 1:
                err = None  # free after the except block: the traceback holds the failed pass
                _free_gpu()
                b //= 2
            elif len(batch) == 1:
                out[batch[0][0]] = err
                k += 1
            else:  # another error in a batch: find the failing chunk by running them one at a time
                for key, text in batch:
                    try:
                        out[key] = self._gen.extract_batch([(key, text)], batch_size=1)[0]
                    except Exception as e:  # noqa: BLE001
                        out[key] = e
                k += len(batch)
        self.last_batch_size = b
        return out

    def pair_scores(self, chunk: Chunk, result) -> dict[str, float]:
        """Head margins "TYPE i j" -> score for every entity pair of the generator's entity lines."""
        import torch
        from relweave.formats import chat_messages, generation_prompt
        from relweave.head import entity_lines, labels, probe_states
        self.load_head()
        tok, model, head, layers = self._head
        labs = labels(self.schema)
        sym = {t for t, _ in labs} - {t for t, d in labs if d == "<"}
        ents, types = entity_lines(result.raw)
        if len(types) < 2:
            return {}
        prompt = self.head_config.get("prompt", {})
        pre = generation_prompt(tok, chat_messages(chunk.text, "sentences", compact=prompt.get("compact_prompt", True),
                                                  schema=self.schema,
                                                  conditioned=prompt.get("conditioned", self.conditioned))) + ents
        every = list(combinations(sorted(types), 2))
        parts = grouped(every, self.group, lambda ps: probe_states(model, tok, pre, types, layers, pairs=ps)[1],
                        on_oom=torch.cuda.empty_cache)
        with torch.no_grad():
            m = (lambda lg: (lg[:, :-1] - lg[:, -1:]).tolist())(head(torch.cat(parts).float()))
        d = {}
        for p, (i, j) in enumerate(every):
            for li, (typ, direction) in enumerate(labs):
                s, t = (i, j) if direction == ">" else (j, i)
                d[f"{typ} {s} {t}"] = m[p][li]
                if typ in sym:
                    d[f"{typ} {j} {i}"] = m[p][li]
        return d

    def combine(self, chunk: Chunk, result, scores: dict[str, float]) -> ChunkGraph:
        """Union of the generator's relations and the head's scores, as a ChunkGraph."""
        from relweave.union import union
        final = union({result.chunk_id: result}, {result.chunk_id: scores}, self.cut, self.margin,
                      schema=self.schema)[result.chunk_id]
        written = {(x.type, x.source, x.target) for x in result.output.relations}
        g = to_chunk_graph(chunk, final.output, written, scores)
        if len(result.output.relations) >= MAX_RELATIONS_PER_CHUNK:
            g.warnings.append(f"chunk {chunk.index}: the generator hit the {MAX_RELATIONS_PER_CHUNK}-relation cap")
        return g

    def run(self, chunk: Chunk) -> ChunkGraph:
        result = self.generate(chunk)
        if isinstance(result, Exception):
            raise result
        if result.output is None:
            return ChunkGraph(chunk.index, warnings=[f"chunk {chunk.index}: unparseable generator output ({result.error})"])
        return self.combine(chunk, result, self.pair_scores(chunk, result))
