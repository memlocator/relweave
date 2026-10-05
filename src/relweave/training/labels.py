"""Labels from a frontier model (the teacher): one prompt file per chunk, the teacher's answers back as gold.

write_prompts writes <chunk>.json with chunk_id and messages (system prompt with the schema, for the business
schema one worked example, the chunk; JSON format). Whoever answers them (an agent or an API script) writes ONLY
the JSON answer to <raw dir>/<chunk file stem>.txt; the conventions an agent should follow are in
relweave/data/label_with_teacher.md (conventions()). import_labels parses and span-verifies the answers exactly
like generator output and writes train/test chunk files.
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

from relweave.formats import chat_messages, finish
from relweave.schema import Schema, load_schema
from relweave.training.data import windows_for


def safe_name(chunk_id: str) -> str:
    return chunk_id.replace("#", "_")


def conventions_file() -> str:
    """The labelling guide shipped with relweave (prompt for an agent and the business conventions)."""
    return files("relweave").joinpath("data/label_with_teacher.md").read_text()


def conventions() -> str:
    """The labelling conventions (the text between the --- lines of the guide), without its first paragraph, which
    is about the per-chunk prompt files."""
    block = conventions_file().split("\n---\n")[1]
    return block.split("\n\n", 1)[1].strip()


def write_prompts(chunks: list[tuple[str, str]], prompt_dir: Path, schema: Schema | None = None) -> list[Path]:
    """One prompt file per chunk; the schema text is rendered from `schema` (default: the business schema)."""
    prompt_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for chunk_id, text in chunks:
        p = prompt_dir / f"{safe_name(chunk_id)}.json"
        p.write_text(json.dumps({"chunk_id": chunk_id, "messages": chat_messages(text, "json", schema=schema)},
                                ensure_ascii=False, indent=2))
        paths.append(p)
    return paths


def import_labels(chunks_path: Path, raw_dir: Path, out: Path, schema: str | Schema = "business",
                  train_windows: int = 500, exclude_docs: set[str] = frozenset(), fix=None, echo=print) -> dict:
    """Teacher answers (one <chunk>.txt per chunk) -> gold out/train.jsonl and out/test.jsonl (by each record's
    "split", default train). Unverifiable mentions, evidence and illegal relations are dropped and counted.
    train_windows: also add each training chunk as sentence windows of this many characters (0: none).
    fix(output) -> (output, n_changed): an optional correction applied to every parsed answer.
    Returns counts: missing, dropped, fixed and chunks per split."""
    sch = load_schema(schema)
    recs: dict[str, list] = {"train": [], "test": []}
    missing = dropped = fixed = 0
    for line in Path(chunks_path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["doc_id"] in exclude_docs:
            continue
        raw = Path(raw_dir) / f"{safe_name(d['chunk_id'])}.txt"
        if not raw.exists():
            missing += 1
            continue
        r = finish(d["chunk_id"], d["text"], raw.read_text(), 0.0, schema=sch)
        if r.output is None:
            echo(f"{d['chunk_id']}: label not parseable ({r.error}); skipped")
            continue
        dropped += r.unresolved_mentions + r.unresolved_evidence + r.illegal_relations + r.dropped_relations
        if fix is not None:
            r.output, n = fix(r.output)
            fixed += n
        recs[d.get("split", "train")].append({"chunk_id": d["chunk_id"], "doc_id": d["doc_id"], "text": d["text"],
                                              "gold": r.output.model_dump(mode="json"), "entity_map": {},
                                              "schema": sch.name})
    if train_windows:
        recs["train"] += windows_for(recs["train"], train_windows)
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in recs.items():
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        echo(f"{split}: {len(rows)} chunks -> {out / f'{split}.jsonl'}")
    return {"missing": missing, "dropped": dropped, "fixed": fixed, **{k: len(v) for k, v in recs.items()}}
