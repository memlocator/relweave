"""Run the models over a labelled chunk file (for evaluation, head training and drafts): the generator's results
(relweave.formats.save_results JSONL) and the pair head's scores ({chunk id: {"TYPE i j": margin}})."""

from __future__ import annotations

import json
from pathlib import Path

from relweave.chunk import Chunk
from relweave.extract import DEFAULT_MODEL, ChunkExtractor
from relweave.formats import load_results, save_results
from relweave.training.data import load_chunks


def generate(chunks_path: Path, out: Path, generator: Path, model: str = DEFAULT_MODEL, schema=None,
             batch_size: int | str = "auto", reserve_mb: int = 500, limit: int | None = None) -> int:
    """The generator's results on every chunk -> out (JSONL). Returns the number of chunks written; a chunk whose
    generation raised is saved as an error result."""
    from relweave.formats import ExtractionResult
    chunks = load_chunks(chunks_path)[:limit]
    ce = ChunkExtractor(generator, None, model, schema, reserve_mb=reserve_mb)
    got = ce.generate_many([(c.chunk_id, c.text) for c in chunks], batch_size)
    results = [r if not isinstance(r, Exception) else
               ExtractionResult(chunk_id=c.chunk_id, raw="", output=None, error=f"{type(r).__name__}: {r}", fmt="sentences")
               for c, r in ((c, got[c.chunk_id]) for c in chunks)]
    save_results(results, out)
    ce.unload_generator()
    return len(results)


def score(chunks_path: Path, results_path: Path, out: Path, generator: Path, head: Path, model: str = DEFAULT_MODEL,
          schema=None, reserve_mb: int = 500) -> int:
    """The pair head's scores for every entity pair of the generator's results -> out (JSON). Returns the number of
    chunks scored."""
    chunks = {c.chunk_id: c for c in load_chunks(chunks_path)}
    ce = ChunkExtractor(generator, head, model, schema, reserve_mb=reserve_mb)
    res = load_results(results_path, {k: c.text for k, c in chunks.items()}, schema={k: c.schema for k, c in chunks.items()})
    scores = {}
    for r in res:
        if r.output is not None and r.chunk_id in chunks:
            text = chunks[r.chunk_id].text
            scores[r.chunk_id] = ce.pair_scores(Chunk(0, text, 0, len(text)), r)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores))
    ce.unload_head()
    return len(scores)
