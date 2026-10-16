"""Extractor: text of any length -> Graph (chunk, extract each chunk, merge)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path

from relweave.chunk import DENSE_WORDS, MAX_WORDS, chunk
from relweave.graph import Graph
from relweave.merge import merge
from relweave.records import ChunkGraph


RESIDENT_MB = 9000  # free GPU memory needed to keep the generator and the head (or zero-shot adapter) loaded together


class Extractor:
    """Wraps the model.

    weights: a Hugging Face Hub repository id (default chrullis/relweave-4b-base) or a local directory with
    generator/ and head/; generator, head: local directories overriding either part (a training run works).
    model: the base model (hub id or path). schema: a Schema, a registered name or "path/to/module.py:NAME"; None
    takes the schema the generator was trained for. cut, margin: union settings (default from the head's config:
    -2.0 and +0.5 for the published model). generator_url: generate on an HTTP server instead of locally, e.g.
    "http://localhost:8000" for `vllm serve chrullis/relweave-4b-base` (the merged generator at the repository root);
    the pair head still runs locally.

    zeroshot: relation types defined at run time. `schema` then is your own Schema (its entity types must be among
    the generator's: Person, Org, Place, Object, Event, Coordinate for the published models); the generator finds
    the entities, and every relation type of your schema is scored on every fitting entity pair by the zero-shot
    adapter (zeroshot_weights, default chrullis/relweave-4b-zeroshot) instead of the trained head. threshold: one
    number or {type: number} on density-adjusted scores (default relweave.zeroshot.DEFAULT_THRESHOLD; scores are
    adjusted for the number of entity pairs each type is asked on per chunk, see relweave.zeroshot), or per type from relweave.zeroshot.calibrate
    with a few labelled examples. Relations carry the adjusted score and origin "zeroshot".

    batch_size: chunks generated together (from all documents of a run_many/iter_run call, sorted by length);
    "auto" picks the largest of 1/2/4/8 that fits the free GPU memory after the generator has loaded, and any batch
    that runs out of memory is retried at half the size.

    sequential: True loads one model at a time (generate every chunk, unload the generator, then score with the pair
    head and unload it); False keeps both resident across calls (about 2 x 3.6 GB); None (default) keeps both
    resident when at least RESIDENT_MB (9 GB) of GPU memory is free at start and goes sequential otherwise.

    A model that cannot be loaded, or a generator that fails on every chunk, raises RuntimeError. A single chunk that
    fails (unparseable output, CUDA OOM at batch 1) becomes a warning in Graph.warnings and the document continues;
    a chunk the head could not score keeps the generator's relations unfiltered."""

    _zs = None  # the zero-shot adapter (zeroshot=True)
    carry = False  # carry-forward generation (earlier chunks' entities in the prompt)

    def __init__(self, weights: str | Path | None = None, schema=None, model: str | None = None,
                 generator: str | Path | None = None, head: str | Path | None = None, device: str = "cuda",
                 max_words: int = MAX_WORDS, dense_words: int | None = None, entity_tagger: str | Path | bool | None = None, sequential: bool | None = None, reserve_mb: int = 500, group: int = 20,
                 cut: float | None = None, margin: float | None = None, batch_size: int | str = "auto",
                 generator_url: str | None = None, zeroshot: bool = False, zeroshot_weights: str | Path | None = None,
                 threshold: float | dict | None = None, carry: bool | None = None, keep_unnamed: bool = False):
        from relweave import extract as ex
        self.max_words, self.batch_size = max_words, batch_size
        self._tagger = None
        self.keep_unnamed = keep_unnamed
        if head is not None and not zeroshot:
            head = ex.resolve_head(head)  # a local directory or a Hub repository id
        root = None
        if generator is None or head is None:
            root = ex.resolve_weights(weights or ex.DEFAULT_WEIGHTS)
            generator, head = generator or root / "generator", head or root / "head"
        # entity tagger (relweave.tagger): a directory, False for none, or None for the weights' own tagger/ when present
        if entity_tagger is None and root is not None and (root / "tagger" / "tagger.pt").exists() and not zeroshot:
            entity_tagger = root / "tagger"
        if entity_tagger:
            from relweave.tagger import EntityTagger
            self._tagger = EntityTagger(entity_tagger)
        self._zs = None
        if zeroshot:
            from relweave.schema import load_schema
            from relweave.zeroshot import DEFAULT_THRESHOLD, DEFAULT_WEIGHTS, ZeroShot
            threshold = DEFAULT_THRESHOLD if threshold is None else threshold
            if schema is None:
                raise ValueError("zeroshot needs your own schema (relation types with definitions)")
            user = load_schema(schema)
            self._ce = ex.ChunkExtractor(generator, None, model, None, device, reserve_mb, group,
                                         generator_url=generator_url)  # entities only, in the generator's own schema
            known = set(self._ce.schema.node_names())
            unknown = sorted(set(user.node_names()) - known)
            if unknown:
                raise ValueError(f"entity types {unknown} are not found by the generator (it knows {sorted(known)}); "
                                 "zero-shot covers relation types only")
            self._zs, self.threshold, self.schema = ZeroShot(zeroshot_weights or DEFAULT_WEIGHTS), threshold, user
        else:
            self._ce = ex.ChunkExtractor(generator, head, model, schema, device, reserve_mb, group, cut, margin,
                                         generator_url=generator_url)
            self.schema = self._ce.schema
        if dense_words is None:  # the head says which chunk size it was trained to read tables and lists in
            cfg = getattr(self._ce, "head_config", None) or {}
            dense_words = cfg.get("chunking", {}).get("dense_words", DENSE_WORDS)
        self.dense_words = dense_words or None  # 0: tables and lists chunked as prose
        if sequential is None:
            import torch
            # both parts resident: generator about 3.6 GB plus head or zero-shot adapter about 3.4 GB, plus activations
            sequential = torch.cuda.mem_get_info()[0] < RESIDENT_MB * 2**20
        self.sequential = sequential
        self.carry = self._ce.carry if carry is None else carry
        self.model_info = {"base": str(self._ce.model), "weights": str(weights or ex.DEFAULT_WEIGHTS),
                           "generator": str(generator), "head": str(head),
                           "cut": self._ce.cut, "margin": self._ce.margin,
                           "entity_tagger": str(entity_tagger) if entity_tagger else None}
        if self._zs is not None:
            self.model_info.update(head=None, cut=None, margin=None, zeroshot=str(self._zs.dir), threshold=threshold)

    def run(self, text: str, doc_id: str = "document") -> Graph:
        """One document. In sequential mode this loads and unloads both models on every call; for several
        documents use run_many or iter_run (each model loads once)."""
        return self.run_many([text], [doc_id])[0]

    def run_many(self, texts: list[str], doc_ids: list[str] | None = None) -> list[Graph]:
        """Graphs for several documents, in order (iter_run collected)."""
        return [g for _, g in self.iter_run(texts, doc_ids)]

    def iter_run(self, texts: list[str], doc_ids: list[str] | None = None) -> Iterator[tuple[int, Graph]]:
        """(index, Graph) for each document as it finishes. All chunks of all documents are generated first (batched
        by length), then each document is scored and merged and yielded, so a sequential extractor loads each model
        once per call."""
        ce = self._ce
        doc_ids = doc_ids or [f"document{k}" if len(texts) > 1 else "document" for k in range(len(texts))]
        chunked = [chunk(t, self.max_words, getattr(self, "dense_words", DENSE_WORDS)) for t in texts]
        items = [(f"{d}:{c.index}", c.model_text) for d, chunks in enumerate(chunked) for c in chunks]
        try:
            _load(ce.load_generator, "generator")
            generated = self._generate_carried(chunked) if self.carry else (
                ce.generate_many(items, self.batch_size) if items else {})
            tagger = getattr(self, "_tagger", None)
            self._tagger_skipped = False
            if tagger is not None and tagger.reads == "generator":
                if hasattr(ce._gen, "model"):  # a local generator: tag on its states while it is loaded
                    texts_by_key = dict(items)
                    for k, r in generated.items():
                        if not isinstance(r, Exception) and r.output is not None:
                            generated[k] = tagger.augment(ce._gen.model, ce._gen.tokenizer, texts_by_key[k], r)
                else:  # a served generator (vLLM): its hidden states are not available here
                    self._tagger_skipped = True
            failures = [r for r in generated.values() if isinstance(r, Exception)]
            if items and len(failures) == len(items):
                raise RuntimeError(f"generation failed on every chunk ({_why(failures[0])}){_memory_note()}") \
                    from failures[0]
            if self.sequential:
                ce.unload_generator()
            if any(not isinstance(r, Exception) and r.output is not None for r in generated.values()):
                _load(self._zs.load if self._zs is not None else ce.load_head,
                      "zero-shot adapter" if self._zs is not None else "pair head")
            for d, chunks in enumerate(chunked):
                graphs = [self._chunk_graph(c, generated[f"{d}:{c.index}"]) for c in chunks]
                yield d, merge(graphs, texts[d], self.schema, doc_ids[d], self.model_info, version("relweave"),
                               carried=self.carry, keep_unnamed=getattr(self, "keep_unnamed", False))
        finally:
            if self.sequential:
                ce.unload_generator()
                ce.unload_head()
                if self._zs is not None:
                    self._zs.unload()

    def _chunk_graph(self, c, r) -> ChunkGraph:
        from relweave.extract import to_chunk_graph
        if isinstance(r, Exception):
            return ChunkGraph(c.index, warnings=[f"chunk {c.index}: generation failed ({_why(r)})"], start=c.start, end=c.end)
        if r.output is None:
            return ChunkGraph(c.index, warnings=[f"chunk {c.index}: unparseable generator output ({r.error})"],
                              start=c.start, end=c.end)
        if self._zs is not None:
            return self._zeroshot_graph(c, r)
        tagger = getattr(self, "_tagger", None)
        if tagger is not None and tagger.reads == "head" and self._ce._head is not None:
            tok, model = self._ce._head[0], self._ce._head[1]  # the head's own loaded model: works with a served generator
            r = tagger.augment(model, tok, c.model_text, r)
        try:
            g = self._ce.combine(c, r, self._ce.pair_scores(c, r))
            if getattr(self, "_tagger_skipped", False):
                g.warnings.append(f"chunk {c.index}: entity tagger skipped: it reads the generator's states, which a "
                                  "served generator does not provide (use a tagger trained on the head's states)")
            return g
        except Exception as e:  # noqa: BLE001  one bad chunk must not lose the document
            g = to_chunk_graph(c, r.output, {(x.type, x.source, x.target) for x in r.output.relations}, {})
            g.warnings.append(f"chunk {c.index}: head scoring failed ({_why(e)}); generator relations kept unfiltered")
            return g


    def _generate_carried(self, chunked) -> dict:
        """Carry-forward generation in rounds: chunk k of every document together, each with the entities its own
        earlier chunks produced (relweave.carry.known_lines, the list the model was trained on)."""
        from relweave.carry import Known, known_lines
        known = [{} for _ in chunked]  # per document: number -> Known
        generated = {}
        for k in range(max((len(c) for c in chunked), default=0)):
            items = [(f"{d}:{cs[k].index}", cs[k].text, known_lines(list(known[d].values())))
                     for d, cs in enumerate(chunked) if k < len(cs)]
            got = self._ce.generate_many(items, self.batch_size)
            generated.update(got)
            for d, cs in enumerate(chunked):
                r = got.get(f"{d}:{cs[k].index}") if k < len(cs) else None
                if r is None or isinstance(r, Exception) or r.output is None:
                    continue
                for e in r.output.entities:
                    n = int(e.id[1:]) if e.id[1:].isdigit() else None
                    if n is None:
                        continue
                    entry = known[d].setdefault(n, Known(n, str(e.type)))
                    entry.surfaces.extend(m for m in e.mentions if m not in entry.surfaces)
                    entry.last_seen = k
        return generated

    def _zeroshot_graph(self, c, r) -> ChunkGraph:
        """The generator's entities, relations scored by the zero-shot adapter for the user's schema."""
        from relweave.extract import to_chunk_graph
        from relweave.records import ChunkRelation
        from relweave.zeroshot import adjust, threshold_for
        g = to_chunk_graph(c, r.output.model_copy(update={"relations": []}), set(), {})
        by_id = {e.id: e for e in r.output.entities}
        ents = [(e.name, e.type, list(dict.fromkeys([e.name] + list(by_id[e.id].mentions)))) for e in g.entities]
        try:
            scored = self._zs.score_chunk(c.model_text, ents, self.schema)
        except Exception as e:  # noqa: BLE001  one bad chunk must not lose the document
            g.warnings.append(f"chunk {c.index}: zero-shot scoring failed ({_why(e)}); no relations")
            return g
        per_type = Counter(edge for edge, _, _, _ in scored)
        for edge, i, j, p in scored:
            q = adjust(p, per_type[edge])  # density calibration: the more pairs a type is asked on, the stricter
            if q > threshold_for(self.threshold, edge):
                g.relations.append(ChunkRelation(edge, g.entities[i].id, g.entities[j].id, "asserted", {}, round(q, 3),
                                                 "zeroshot"))
        return g


def _load(load, what: str) -> None:
    """Run a model loader; any failure becomes a RuntimeError naming the part and the GPU memory situation."""
    try:
        load()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"could not load the {what}: {_why(e)}{_memory_note()}") from e


def _memory_note() -> str:
    """'; N MB of M MB GPU memory free (the 4-bit model needs about 3.6 GB per part plus generation memory)'."""
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001  no CUDA: nothing to add
        return ""
    return (f"; {free // 2**20} MB of {total // 2**20} MB GPU memory free (each model part needs about 3600 MB "
            "plus memory to generate)")


def _why(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ''}"
