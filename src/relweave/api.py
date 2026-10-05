"""Extractor: text of any length -> Graph (chunk, extract each chunk, merge)."""

from __future__ import annotations

from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path

from relweave.chunk import chunk
from relweave.graph import Graph
from relweave.merge import merge
from relweave.records import ChunkGraph


class Extractor:
    """Wraps the model.

    weights: a Hugging Face Hub repository id (default chrullis/relweave-4b-base) or a local directory with
    generator/ and head/; generator, head: local directories overriding either part (a training run works).
    model: the base model (hub id or path). schema: a Schema, a registered name or "path/to/module.py:NAME"; None
    takes the schema the generator was trained for. cut, margin: union settings (default from the head's config:
    -2.0 and +0.5 for the published model).

    batch_size: chunks generated together (from all documents of a run_many/iter_run call, sorted by length);
    "auto" picks the largest of 1/2/4/8 that fits the free GPU memory after the generator has loaded, and any batch
    that runs out of memory is retried at half the size.

    sequential: True loads one model at a time (generate every chunk, unload the generator, then score with the pair
    head and unload it); False keeps both resident across calls (about 2 x 2.7 GB); None (default) keeps both
    resident when at least 7 GB of GPU memory is free at start and goes sequential otherwise.

    A model that cannot be loaded, or a generator that fails on every chunk, raises RuntimeError. A single chunk that
    fails (unparseable output, CUDA OOM at batch 1) becomes a warning in Graph.warnings and the document continues;
    a chunk the head could not score keeps the generator's relations unfiltered."""

    def __init__(self, weights: str | Path | None = None, schema=None, model: str | None = None,
                 generator: str | Path | None = None, head: str | Path | None = None, device: str = "cuda",
                 max_words: int = 200, sequential: bool | None = None, reserve_mb: int = 500, group: int = 20,
                 cut: float | None = None, margin: float | None = None, batch_size: int | str = "auto"):
        from relweave import extract as ex
        self.max_words, self.batch_size = max_words, batch_size
        if generator is None or head is None:
            root = ex.resolve_weights(weights or ex.DEFAULT_WEIGHTS)
            generator, head = generator or root / "generator", head or root / "head"
        model = model or ex.DEFAULT_MODEL
        self._ce = ex.ChunkExtractor(generator, head, model, schema, device, reserve_mb, group, cut, margin)
        self.schema = self._ce.schema
        if sequential is None:
            import torch
            sequential = torch.cuda.mem_get_info()[0] < 7000 * 2**20
        self.sequential = sequential
        self.model_info = {"base": str(model), "weights": str(weights or ex.DEFAULT_WEIGHTS),
                           "generator": str(generator), "head": str(head),
                           "cut": self._ce.cut, "margin": self._ce.margin}

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
        chunked = [chunk(t, self.max_words) for t in texts]
        items = [(f"{d}:{c.index}", c.text) for d, chunks in enumerate(chunked) for c in chunks]
        try:
            _load(ce.load_generator, "generator")
            generated = ce.generate_many(items, self.batch_size) if items else {}
            failures = [r for r in generated.values() if isinstance(r, Exception)]
            if items and len(failures) == len(items):
                raise RuntimeError(f"generation failed on every chunk ({_why(failures[0])}){_memory_note()}") \
                    from failures[0]
            if self.sequential:
                ce.unload_generator()
            if any(not isinstance(r, Exception) and r.output is not None for r in generated.values()):
                _load(ce.load_head, "pair head")
            for d, chunks in enumerate(chunked):
                graphs = [self._chunk_graph(c, generated[f"{d}:{c.index}"]) for c in chunks]
                yield d, merge(graphs, texts[d], self.schema, doc_ids[d], self.model_info, version("relweave"))
        finally:
            if self.sequential:
                ce.unload_generator()
                ce.unload_head()

    def _chunk_graph(self, c, r) -> ChunkGraph:
        from relweave.extract import to_chunk_graph
        if isinstance(r, Exception):
            return ChunkGraph(c.index, warnings=[f"chunk {c.index}: generation failed ({_why(r)})"], start=c.start, end=c.end)
        if r.output is None:
            return ChunkGraph(c.index, warnings=[f"chunk {c.index}: unparseable generator output ({r.error})"],
                              start=c.start, end=c.end)
        try:
            return self._ce.combine(c, r, self._ce.pair_scores(c, r))
        except Exception as e:  # noqa: BLE001  one bad chunk must not lose the document
            g = to_chunk_graph(c, r.output, {(x.type, x.source, x.target) for x in r.output.relations}, {})
            g.warnings.append(f"chunk {c.index}: head scoring failed ({_why(e)}); generator relations kept unfiltered")
            return g


def _load(load, what: str) -> None:
    """Run a model loader; any failure becomes a RuntimeError naming the part and the GPU memory situation."""
    try:
        load()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"could not load the {what}: {_why(e)}{_memory_note()}") from e


def _memory_note() -> str:
    """'; N MB of M MB GPU memory free (the 4-bit model needs about 2.7 GB per part plus generation memory)'."""
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001  no CUDA: nothing to add
        return ""
    return (f"; {free // 2**20} MB of {total // 2**20} MB GPU memory free (each model part needs about 2700 MB "
            "plus memory to generate)")


def _why(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ''}"
