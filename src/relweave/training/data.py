"""Labelled chunk files: one JSON record per line with chunk_id, doc_id, text, gold (an ExtractionOutput under the
record's schema, or null for unlabelled text), optional "schema" (a registered schema name, default business) and
"split" (train or test)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from relweave.formats import sentence_windows
from relweave.schema import BUSINESS, ExtractionOutput, Schema, registered_schema


@dataclass
class LabelledChunk:
    chunk_id: str
    doc_id: str
    start: int
    end: int
    text: str
    gold: ExtractionOutput | None
    entity_map: dict[str, str] = field(default_factory=dict)  # local id -> world id (synthetic corpora)
    schema: Schema = BUSINESS  # the schema the gold is written under (scoring: legality, symmetric edges)


def schema_of(record: dict) -> Schema:
    """The schema a record names (default business); a user schema must be loaded first (load_schema)."""
    try:
        return registered_schema(record.get("schema", BUSINESS.name))
    except ValueError as e:
        raise ValueError(f"chunk {record.get('chunk_id')}: {e}") from None


def load_chunks(path: Path) -> list[LabelledChunk]:
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        schema = schema_of(d)
        gold = ExtractionOutput.model_validate(d["gold"], context={"schema": schema}) if d.get("gold") else None
        out.append(LabelledChunk(chunk_id=d["chunk_id"], doc_id=d["doc_id"], start=0, end=len(d["text"]),
                                 text=d["text"], gold=gold, entity_map=d.get("entity_map", {}), schema=schema))
    return out


def project_gold(window_text: str, gold: ExtractionOutput) -> tuple[ExtractionOutput, int]:
    """Restrict a chunk-level gold output to a window of that chunk by string matching.

    Entities keep the surfaces that occur in the window; relations are kept when
    their evidence is inside the window and both endpoints survive. Returns the
    projected output and the number of relations dropped.
    """
    kept = []
    for e in gold.entities:
        surfaces = [m for m in e.mentions if m in window_text]
        if surfaces:
            kept.append(e.model_copy(update={"mentions": surfaces}))
    ids = {e.id for e in kept}
    rels = [r for r in gold.relations if r.evidence in window_text and r.source in ids and r.target in ids]
    # model_copy keeps a non-business gold valid: a constructor would validate under BUSINESS
    return gold.model_copy(update={"entities": kept, "relations": rels}), len(gold.relations) - len(rels)


def windows_for(records: list[dict], max_chars: int) -> list[dict]:
    """Training augmentation: each labelled chunk again as small sentence windows, with its labels projected onto
    each window by string match. Every window keeps its record's schema."""
    extra = []
    for rec in records:
        schema = schema_of(rec)
        gold = ExtractionOutput.model_validate(rec["gold"], context={"schema": schema})
        for i, (a, b) in enumerate(sentence_windows(rec["text"], max_chars, overlap_sentences=1)):
            text = rec["text"][a:b]
            sub, _ = project_gold(text, gold)
            if sub.entities:
                extra.append({"chunk_id": f"{rec['chunk_id']}w{i+1}", "doc_id": rec["doc_id"], "text": text,
                              "gold": sub.model_dump(mode="json"), "entity_map": {},
                              **({"schema": rec["schema"]} if "schema" in rec else {})})
    return extra


def ingest(files: list[Path], out: Path, test_frac: float = 0.2, seed: int = 0, max_chars: int | None = None,
           overlap_sentences: int = 2, max_words: int | None = None) -> tuple[int, int]:
    """Chunk plain-text documents into unlabelled chunk records (gold null) with a train/test split by document. By
    default with relweave.chunk.chunk at MAX_WORDS, the chunking `relweave run` applies, so the model is trained on
    chunks like the ones it reads; MAX_CHARS instead gives sentence windows of that many characters.
    Returns (chunks written, test documents)."""
    from relweave.chunk import MAX_WORDS, chunk
    import random
    rng = random.Random(seed)
    files = sorted(files)
    doc_ids = [f.stem for f in files]
    test_ids = set(rng.sample(doc_ids, max(1, int(round(len(doc_ids) * test_frac))))) if test_frac > 0 else set()
    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for f in files:
            text = f.read_text()
            spans = (sentence_windows(text, max_chars, overlap_sentences) if max_chars
                     else [(c.start, c.end) for c in chunk(text, max_words or MAX_WORDS)])
            for k, (a, b) in enumerate(spans, 1):
                fh.write(json.dumps({"chunk_id": f"{f.stem}#{k}", "doc_id": f.stem, "text": text[a:b], "gold": None,
                                     "entity_map": {}, "split": "test" if f.stem in test_ids else "train"},
                                    ensure_ascii=False) + "\n")
                n += 1
    return n, len(test_ids)


def split(chunks: Path, out_dir: Path, schema: str, val_frac: float = 0.15, test_frac: float = 0.2, seed: int = 0,
          group: str | None = None) -> dict[str, tuple[int, int]]:
    """Split a chunk file into OUT_DIR/{train,val,test}_unlabelled.jsonl by document, every record tagged with the
    schema name. GROUP: a regular expression whose first match in the document id names a group kept on one side (the
    volumes of one report, the years of one company), so no group spans the split. Returns split -> (documents,
    chunks)."""
    import random
    import re
    rows = [json.loads(x) for x in chunks.read_text().splitlines() if x.strip()]
    key = (lambda d: (re.search(group, d) or [d])[0]) if group else (lambda d: d)
    groups = sorted({key(r["doc_id"]) for r in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_test = max(1, round(len(groups) * test_frac))
    n_val = max(1, round(len(groups) * val_frac))
    if n_test + n_val >= len(groups):
        raise ValueError(f"{len(groups)} documents (or groups) are too few for a train, validation and test split")
    side = {g: "test" for g in groups[:n_test]} | {g: "val" for g in groups[n_test:n_test + n_val]}
    out = {"train": [], "val": [], "test": []}
    for r in rows:
        name = side.get(key(r["doc_id"]), "train")
        out[name].append({**r, "schema": schema, "split": "train" if name == "train" else "test"})
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rs in out.items():
        (out_dir / f"{name}_unlabelled.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs))
    return {k: (len({r["doc_id"] for r in v}), len(v)) for k, v in out.items()}
