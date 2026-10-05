"""Cheap labelling by correcting the model's draft (pre-annotate, then verify).

Instead of writing a full extraction per chunk, an agent reads the schema and conventions once and then, for
batches of chunks, corrects the draft of our best system (the all-4B union): drop or retype drafted relations, add
entities and relations it missed. Its answer is a few short lines per chunk.

  export  N chunks + the system's drafts (generator results and head scores) -> <out>/instructions.md, <out>/batch_NN.txt
  merge   <out>/answers_NN.txt -> <out>/labels.jsonl (chunks with the corrected gold) and, if the chunks already
          have gold, agreement of the corrected labels with it (typed relation F1), next to the draft's own score

Answer format, per chunk (relations not mentioned are kept):
  ## <chunk_id>
  drop R3 R7                    drafted relations that the text does not state
  retype R2 OWNS_STAKE_IN       right pair, wrong type (same direction)
  flip R5                       right pair and type, wrong direction
  entity E14 Org: Surface | another surface     a missing entity (surfaces verbatim from the text)
  mention E3: the firm          another surface of an existing entity
  add ACQUIRED E2 E14           a missing relation (source first)

CLI: relweave label draft (export) and relweave label correct (merge).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from relweave.formats import ExtractionResult, load_results
from relweave.schema import BUSINESS, MAX_ENTITIES_PER_CHUNK, MAX_RELATIONS_PER_CHUNK, ExtractionOutput, schema_description
from relweave.schema.output import Entity, Relation
from relweave.training.data import load_chunks
from relweave.training.eval import summarize
from relweave.training.labels import conventions
from relweave.union import union

CUT, MARGIN = -2.0, 1.0  # the union settings for drafts (all-4B system, chosen on validation): fewer head additions


def render(cid: str, text: str, out) -> str:
    num = {e.id: k for k, e in enumerate(out.entities, 1)}
    lines = [f"## {cid}", "TEXT:", text.strip(), "DRAFT ENTITIES:"]
    for e in out.entities:
        lines.append(f"E{num[e.id]} {str(e.type)}: " + " | ".join(dict.fromkeys([e.name, *e.mentions])))
    lines.append("DRAFT RELATIONS:")
    for k, r in enumerate(out.relations, 1):
        lines.append(f"R{k} {r.type} E{num[r.source]} E{num[r.target]}")
    return "\n".join(lines) + "\n"


INSTRUCTIONS = """# Correct draft extractions

Each batch file holds chunks of text with a DRAFT extraction made by a model: entities (E<n>,
type, surface strings) and relations (R<n> TYPE source target). The draft is about 80% right and misses about a
quarter of the relations the text states, most often items of lists ("acquired A, B, C and D": one relation per
item) and relations inside long sentences. Your job is to make the extraction exactly what the text states.

For each chunk, read the text once, check every drafted relation against it, then look specifically for what the
draft missed: every item of every list, every relation in long sentences. Be as strict with the draft as with your
own additions: keep a drafted relation only if the text states it outright; drop it if it is merely plausible,
inferred from names or world knowledge, or true of a related entity rather than this one. When in doubt, drop. Write only the changes, in this format
(one `## <chunk_id>` header per chunk, even if nothing changes; relations you do not mention are kept):

  drop R3 R7                    the text does not state these
  retype R2 OWNS_STAKE_IN       right entities, wrong type (keeps the direction)
  flip R5                       right entities and type, wrong direction
  entity E14 Org: Surface | another surface     a missing entity; surfaces copied verbatim from the text
  mention E3: the firm          another verbatim surface of an existing entity
  add ACQUIRED E2 E14           a missing relation, source first

New entities continue the numbering after the draft's last entity. Write the answers for all chunks of batch file
<batch> to <answers>, nothing else: no prose, no code fences. Do not read any other files except this one and your
batch file.

## Schema

{schema}

## Conventions

{conventions}
"""


def export(out_dir, chunks_path, results_path, scores_path, set_name=None, n=50, per_batch=25, cut=CUT, margin=MARGIN):
    """scores_path: head scores {chunk id: {"TYPE i j": margin}}, or {set name: {...}} with set_name."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    chunks = {c.chunk_id: c for c in load_chunks(Path(chunks_path))}
    res = {r.chunk_id: r for r in load_results(Path(results_path), {k: c.text for k, c in chunks.items()},
                                               schema={k: c.schema for k, c in chunks.items()})
           if r.output is not None and r.chunk_id in chunks}
    scores = json.loads(Path(scores_path).read_text())
    scores = scores[set_name] if set_name else scores
    ids = sorted(res); random.Random(0).shuffle(ids); ids = sorted(ids[:int(n)])
    schema = chunks[ids[0]].schema  # the chunks' own schema (record field "schema"; business by default)
    business = schema.name == "business"
    drafts = union({k: res[k] for k in ids}, scores, float(cut), float(margin), schema=schema)
    for d in drafts.values():  # a relation from both the generator and the head appears once
        uniq = {(r.type, r.source, r.target): r for r in d.output.relations}
        d.output = d.output.model_copy(update={"relations": list(uniq.values())})
    (out_dir / "instructions.md").write_text(INSTRUCTIONS.format(
        schema=schema_description() if business else schema.render(), conventions=conventions()))
    (out_dir / "drafts.json").write_text(json.dumps({k: drafts[k].output.model_dump(mode="json") for k in ids}))
    for b in range(0, len(ids), int(per_batch)):
        (out_dir / f"batch_{b // int(per_batch):02d}.txt").write_text(
            "\n".join(render(k, chunks[k].text, drafts[k].output) for k in ids[b:b + int(per_batch)]))
    print(f"{len(ids)} chunks in {(len(ids) + int(per_batch) - 1) // int(per_batch)} batches -> {out_dir}")


def apply(draft, text: str, lines: list[str], schema=None):
    """The corrected ExtractionOutput: the draft with an agent's change lines applied; bad lines are skipped.
    Types and legality follow schema (business by default)."""
    schema = schema or BUSINESS
    ents = {f"E{k}": e.model_copy(deep=True) for k, e in enumerate(draft.entities, 1)}
    rels = {f"R{k}": r.model_copy() for k, r in enumerate(draft.relations, 1)}
    by_old = {e.id: f"E{k}" for k, e in enumerate(draft.entities, 1)}
    triples = {k: [r.type, by_old[r.source], by_old[r.target]] for k, r in rels.items()}
    skipped = 0
    for line in lines:
        w = line.split()
        try:
            if w[0] == "drop":
                for r in w[1:]:
                    triples.pop(r, None)
            elif w[0] == "retype":
                triples[w[1]][0] = w[2]
            elif w[0] == "flip":
                t = triples[w[1]]; t[1], t[2] = t[2], t[1]
            elif w[0] == "entity":
                m = re.match(r"entity (E\d+) (\w+): (.+)$", line)
                surfaces = [s.strip() for s in m.group(3).split("|") if s.strip() and s.strip() in text]
                if surfaces and m.group(2) in schema.node_names():
                    ents[m.group(1)] = Entity(id=m.group(1).lower(), type=m.group(2), name=surfaces[0], mentions=surfaces)
                else:
                    skipped += 1
            elif w[0] == "mention":
                m = re.match(r"mention (E\d+): (.+)$", line)
                if m.group(2).strip() in text:
                    ents[m.group(1)].mentions.append(m.group(2).strip())
            elif w[0] == "add":
                triples[f"A{len(triples)}_{w[2]}_{w[3]}"] = [w[1], w[2], w[3]]
            else:
                skipped += 1
        except (IndexError, KeyError, AttributeError, ValueError):
            skipped += 1
    for k, e in ents.items():
        e.id = k.lower()
    types = {k: str(e.type) for k, e in ents.items()}
    out_rels, seen = [], set()
    for typ, s, t in triples.values():
        if s in ents and t in ents and s != t and schema.legal(typ, types[s], types[t]) and (typ, s, t) not in seen:
            seen.add((typ, s, t))
            out_rels.append(Relation(type=typ, source=s.lower(), target=t.lower(), evidence=""))
        else:
            skipped += 1
    out_rels = out_rels[:MAX_RELATIONS_PER_CHUNK]
    entities = list(ents.values())
    if len(entities) > MAX_ENTITIES_PER_CHUNK:  # agents' additions overflowed: drop entities in no relation first
        used = {r.source for r in out_rels} | {r.target for r in out_rels}
        spare = [e for e in reversed(entities) if e.id not in used][:len(entities) - MAX_ENTITIES_PER_CHUNK]
        entities = [e for e in entities if e not in spare]
        skipped += len(spare)
    if len(entities) > MAX_ENTITIES_PER_CHUNK:
        return None, skipped
    return ExtractionOutput.model_validate({"entities": entities, "relations": out_rels}, context={"schema": schema}), skipped


def merge(out_dir, chunks_path):
    out_dir = Path(out_dir)
    chunks = {c.chunk_id: c for c in load_chunks(Path(chunks_path))}
    drafts = {k: ExtractionOutput.model_validate(v, context={"schema": chunks[k].schema})
              for k, v in json.loads((out_dir / "drafts.json").read_text()).items()}
    answers = {}
    for p in sorted(out_dir.glob("answers_*.txt")):
        cid = None
        for line in p.read_text().splitlines():
            if line.startswith("## "):
                cid = line[3:].strip(); answers[cid] = []
            elif cid and line.strip():
                answers[cid].append(line.strip())
    corrected, skipped = {}, 0
    for cid, d in drafts.items():
        if cid in answers:
            out, s = apply(d, chunks[cid].text, answers[cid], chunks[cid].schema); skipped += s
            if out is None:
                print(f"  {cid}: over {len(d.entities)} entities even without unrelated ones; left out")
                continue
            corrected[cid] = out
    with (out_dir / "labels.jsonl").open("w") as f:
        for cid, o in corrected.items():
            c = chunks[cid]  # the labelled chunk record layout (relweave.training.data)
            f.write(json.dumps({"chunk_id": cid, "doc_id": c.doc_id, "text": c.text, "gold": o.model_dump(mode="json"),
                                "split": "train", "schema": c.schema.name}, ensure_ascii=False) + "\n")
    print(f"{len(corrected)}/{len(drafts)} chunks answered, {sum(len(a) for a in answers.values())} change lines, "
          f"{skipped} skipped or illegal")
    have = [c for c in corrected if chunks[c].gold is not None]
    if have:
        g = [chunks[c] for c in have]
        for name, outs in (("draft", drafts), ("corrected", corrected)):
            m = summarize(g, [ExtractionResult(chunk_id=c, raw="", output=outs[c]) for c in have])
            s = m["rel_strict"]
            print(f"{name:10} vs existing gold: typed F1 {s['f1']:.3f} (P {s['precision']:.3f} R {s['recall']:.3f}), "
                  f"relations {sum(len(outs[c].relations) for c in have)} vs gold {sum(len(chunks[c].gold.relations) for c in have)}")
