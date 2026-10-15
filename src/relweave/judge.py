"""Pooled judging: score systems without trusting one labelling.

Every relation written by the gold or by any system is a claim; language-model judges (or people) rule each claim true
or false against the text, with a verbatim quote. A system's precision is the share of its claims judged true; its
pooled recall is the share of all judged-true claims it wrote. The gold is scored like any other source, so a narrow or
partly wrong gold stops deciding what counts as right. Planted decoys (direction swapped, another entity of the same
type, a neighbouring relation type) test the judges: a sound judge rejects almost all of them.

Pooled recall counts only relations some source found; relations every source missed are invisible to it.

Files of a pool directory:
  items.jsonl             per document: text and numbered claims, sources hidden (what the judges see)
  key.jsonl               per claim: its sources and, for a decoy, the decoy kind
  verdicts_<judge>.jsonl  per document: {"id", "verdicts": [{"n", "verdict": "true"|"false", "quote"}]}
Labels files (the gold, each system): one object per document or chunk, {"id", "entities": [{"name", "type",
"mentions"}], "relations": [{"source", "type", "target"}]} with entities named; the gold's records carry "text".
"""

from __future__ import annotations

import collections
import json
import random
import re
from pathlib import Path

from relweave.schema import Schema

RULES = """## Rules

1. **The text, not the world.** A claim is true only if the document states it, directly or through a clear reference
   ("the inquiry", "the company", "she" when the referent is unambiguous). A fact you know but the text does not state
   is false.
2. **As fact.** Proposals, plans, allegations not yet proven, claims someone makes, negations and hypotheticals are
   false. A decision taken, an appointment made, a court's finding and a plain statement by the author are fact.
3. **Direction matters.** `A SUBSIDIARY_OF B` is not `B SUBSIDIARY_OF A`; check source and target against the
   definition's direction.
4. **The type must fit its definition.** Each definition says where its neighbours begin; a claim whose type is the
   neighbour's is false. A different type that also fits is a separate claim, judged on its own.
5. **Names.** Entity names may be written differently from the text (a short or long form, a translation); judge the
   thing meant.

## Output

For every claim: `{"n": <n>, "verdict": "true" | "false", "quote": "<the shortest verbatim span of the text that
states it, required when true, copied exactly>"}`. A "true" without a verbatim quote counts as false.
"""


def brief(schema: Schema) -> str:
    """The judging brief: the schema's own definitions and the domain-free rules, so brief and schema cannot drift."""
    return ("# Judging brief: is this relation stated in the text?\n\nYou check claimed relations against the document "
            "they were extracted from. Each claim is\n`source TYPE target`. Decide only one thing: **does this text "
            "state this relation as fact, and does it fit the\ndefinition of TYPE?**\n\n## Definitions\n\nThe claims "
            "use these types (and only these definitions count):\n\n" + schema.render() + "\n\n" + RULES)


def judge_prompt(brief_path: str | Path, items_path: str | Path, verdicts_path: str | Path) -> str:
    """The task for one judge (a language-model agent with file access, or a person)."""
    return (f"You are a judge of extracted relations. Read the brief {brief_path} first and follow it exactly.\n\n"
            f"Input: {items_path}. One JSON object per line: {{\"id\", \"text\", \"claims\": [{{\"n\", \"source\", "
            f"\"type\", \"target\"}}]}}. Some claims are deliberately wrong; judge each claim on its own against its "
            f"document's text only.\n\nOutput: {verdicts_path}, one line per document in input order: {{\"id\": <id>, "
            f"\"verdicts\": [{{\"n\": <n>, \"verdict\": \"true\"|\"false\", \"quote\": \"<shortest verbatim span of the "
            f"text that states it; required when true, copied exactly>\"}}]}}. Every claim gets exactly one verdict.\n\n"
            f"Read each text in full and decide every verdict yourself; do not use keyword rules or scripts to decide. A "
            f"script may only read the input in batches, write the output, and at the end check that every claim has "
            f"one verdict and every \"true\" quote occurs verbatim in its text (whitespace collapsed).")


def key(s: str) -> str:
    return re.sub(r"[^\w ]", "", s.lower().replace("’", "'")).strip()


def canon(sources):
    """name -> canonical name, merging entities of all sources by shared names and capitalised mentions. A lowercase
    mention ("the inquiry") joins an entity another source named with exactly that text, when only one entity of that
    type lists it: a chunk's "the inquiry" and the inquiry's labelled name are one claim's end, not two."""
    can = {}
    for d in sources:
        for e in d.get("entities", []):
            names = [e["name"], *e.get("mentions", [])]
            c = next((can[key(m)] for m in names if key(m) in can), e["name"])
            for m in names:
                if m[:1].isupper() or m == e["name"]:
                    can.setdefault(key(m), c)
    lower = collections.defaultdict(set)
    for d in sources:
        for e in d.get("entities", []):
            for m in e.get("mentions", []):
                if not m[:1].isupper() and m != e["name"]:
                    lower[(key(m), e.get("type"))].add(can.get(key(e["name"]), e["name"]))
    for d in sources:
        for e in d.get("entities", []):
            owners = lower.get((key(e["name"]), e.get("type")), set())
            if len(owners) == 1 and not e["name"][:1].isupper():
                can[key(e["name"])] = next(iter(owners))
    return lambda n: can.get(key(n), n)


def _read(path):
    return [json.loads(x) for x in open(path) if x.strip()]


def neighbour_types(schema: Schema) -> dict[str, list[str]]:
    """Each relation type -> the other types legal for some of its endpoint types: the near misses a judge must tell
    apart (the "type" decoys)."""
    out = {}
    for t in schema.edge_names():
        pairs = [(s, o) for s in schema.sources(t) for o in schema.targets(t)]
        out[t] = [u for u in schema.edge_names() if u != t and any(schema.legal(u, s, o) for s, o in pairs)]
    return out


def items(out: str | Path, labels: dict[str, str | Path], schema: Schema | None = None, gold: str | None = None,
          decoys: int = 2, seed: int = 1012, swap: dict[str, str] | None = None, directed: set[str] | None = None,
          world: list[tuple[str, str, str]] = ()) -> tuple[int, int]:
    """Build a pool from labels files ({source name: path}); GOLD names the source whose records carry the text and
    whose ids define the documents (default: "gold", else the first). Per document DECOYS decoys of different kinds:
    direction (types not symmetric), entity (another entity of the target's type), type (SWAP, or a neighbouring type of
    SCHEMA). WORLD: (source, type, target) facts true in the world, planted where the source entity appears but the text
    does not state them. Returns (documents, claims)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    src = {name: {d["id"]: d for d in _read(path)} for name, path in labels.items()}
    gold = gold or ("gold" if "gold" in src else next(iter(src)))
    near = neighbour_types(schema) if schema is not None else {}
    if directed is None:
        directed = {t for t in schema.edge_names() if not schema.symmetric(t)} if schema is not None else set()
    rng = random.Random(seed)
    n_claims = 0
    with open(out / "items.jsonl", "w") as wi, open(out / "key.jsonl", "w") as wk:
        for i, g in src[gold].items():
            docs = {n: s.get(i, {"entities": [], "relations": []}) for n, s in src.items()}
            c = canon(list(docs.values()))
            pool = collections.defaultdict(set)
            for n, d in docs.items():
                for r in d["relations"]:
                    pool[(c(r["source"]), r["type"], c(r["target"]))].add(n)
            types = {}
            for d in docs.values():
                for e in d["entities"]:
                    types.setdefault(c(e["name"]), e["type"])
            claims = [(t, sorted(s), "") for t, s in pool.items()]
            real = list(pool)
            for t in rng.sample(real, min(decoys, len(real))):
                kind = rng.choice(["direction", "entity", "type"])
                s, ty, o = t
                d = None
                if kind == "direction" and ty in directed:
                    d = (o, ty, s)
                elif kind == "entity":
                    alt = [e for e, et in types.items() if et == types.get(o) and e not in (s, o)]
                    d = (s, ty, rng.choice(alt)) if alt else None
                elif kind == "type":
                    alt = [swap[ty]] if swap and ty in swap else near.get(ty, [])
                    d = (s, rng.choice(alt), o) if alt else None
                if d and d not in pool:
                    claims.append((d, [], kind))
            for w in world:
                if any(key(w[0]) == key(e) for e in types) and w not in pool and rng.random() < 0.5:
                    claims.append((w, [], "world"))
            rng.shuffle(claims)
            n_claims += len(claims)
            wi.write(json.dumps({"id": i, "text": g.get("text", ""), "claims": [
                {"n": k, "source": t[0], "type": t[1], "target": t[2]} for k, (t, _, _) in enumerate(claims)]},
                ensure_ascii=False) + "\n")
            wk.write(json.dumps({"id": i, "claims": [{"n": k, "sources": s, "decoy": dk}
                                                       for k, (_, s, dk) in enumerate(claims)]}) + "\n")
    return len(src[gold]), n_claims


def load(out: str | Path, judges):
    """(claims by (doc, n), verdicts by (judge, doc, n), count of "true" verdicts dropped for a missing quote)."""
    out = Path(out)
    texts = {d["id"]: re.sub(r"\s+", " ", d["text"]) for d in _read(out / "items.jsonl")}
    keyed = {(d["id"], c["n"]): c for d in _read(out / "key.jsonl") for c in d["claims"]}
    verdicts, dropped = {}, collections.Counter()
    for j in judges:
        for d in _read(out / f"verdicts_{j}.jsonl"):
            for v in d["verdicts"]:
                ok = v["verdict"] == "true"
                if ok and re.sub(r"\s+", " ", v.get("quote", "")) not in texts[d["id"]]:
                    ok, dropped[j] = False, dropped[j] + 1
                verdicts[(j, d["id"], v["n"])] = ok
    return keyed, verdicts, dropped


def judged_true(keyed, verdicts, judges) -> set:
    """Claims every judge calls true (decoys excluded)."""
    if not judges:
        raise ValueError("name the judges; with none every claim would count as true")
    return {k for k, c in keyed.items() if not c["decoy"] and all(verdicts.get((j, *k), False) for j in judges)}


def check(out, judges) -> dict:
    """Per judge: decoys accepted (stop above about 10%) and "true" verdicts without a verbatim quote; for two judges,
    their disagreement (stop above about 15%)."""
    keyed, verdicts, dropped = load(out, judges)
    dec = [k for k, c in keyed.items() if c["decoy"]]
    res = {"judges": {}}
    for j in judges:
        acc = [k for k in dec if verdicts.get((j, *k), False)]
        res["judges"][j] = {"decoys": len(dec), "decoys_accepted": len(acc), "no_quote": dropped[j],
                            "accepted_by_kind": dict(collections.Counter(keyed[k]["decoy"] for k in acc))}
    if len(judges) == 2:
        both = [k for k in keyed if (judges[0], *k) in verdicts and (judges[1], *k) in verdicts]
        res["disagreement"] = (sum(verdicts[(judges[0], *k)] != verdicts[(judges[1], *k)] for k in both), len(both))
    return res


def _f1(tp, wrote, true):
    p, r = tp / max(wrote, 1), tp / max(true, 1)
    return 2 * p * r / max(p + r, 1e-9)


def _per_doc(keyed, true, names, select=lambda k: True):
    per = collections.defaultdict(lambda: {"t": 0, **{n: [0, 0] for n in names}})
    for k, c in keyed.items():
        if c["decoy"] or not select(k):
            continue
        per[k[0]]["t"] += k in true
        for n in names:
            per[k[0]][n][0] += n in c["sources"]
            per[k[0]][n][1] += n in c["sources"] and k in true
    return per


def _sample_f1(per, sample, n):
    w = sum(per[i][n][0] for i in sample)
    return _f1(sum(per[i][n][1] for i in sample), w, sum(per[i]["t"] for i in sample))


def score(out, judges, boot: int = 2000) -> dict:
    """Per source: claims written, judged true, precision, pooled recall, F1 and its 95% document-bootstrap interval."""
    keyed, verdicts, _ = load(out, judges)
    true = judged_true(keyed, verdicts, judges)
    names = sorted({s for c in keyed.values() for s in c["sources"]})
    per = _per_doc(keyed, true, names)
    docs = sorted({i for i, _ in keyed})
    rng = random.Random(0)
    samples = [rng.choices(docs, k=len(docs)) for _ in range(boot)]
    res = {"claims": sum(1 for c in keyed.values() if not c["decoy"]), "true": len(true), "sources": {}}
    for n in names:
        wrote = sum(per[i][n][0] for i in docs)
        tp = sum(per[i][n][1] for i in docs)
        f1s = sorted(_sample_f1(per, s, n) for s in samples)
        res["sources"][n] = {"wrote": wrote, "true": tp, "precision": tp / max(wrote, 1), "recall": tp / max(len(true), 1),
                             "f1": _f1(tp, wrote, len(true)), "ci": (f1s[int(0.025 * boot)], f1s[int(0.975 * boot) - 1])}
    return res


def by_type(out, judges, without: str | None = None) -> dict:
    """Per relation type and source: (F1, precision, pooled recall, judged-true count), plus "without <TYPE>" when one
    type dominates the pool. A headline F1 carried by one frequent, easy type hides the rest."""
    keyed, verdicts, _ = load(out, judges)
    typ = {(d["id"], c["n"]): c["type"] for d in _read(Path(out) / "items.jsonl") for c in d["claims"]}
    true = judged_true(keyed, verdicts, judges)
    names = sorted({s for c in keyed.values() for s in c["sources"]})
    groups = [(t, lambda k, t=t: typ[k] == t) for t in sorted({typ[k] for k in true})]
    if without:
        groups.append((f"without {without}", lambda k: typ[k] != without))
    res = {}
    for label, sel in groups:
        tr = {k for k in true if sel(k)}
        row = {}
        for n in names:
            wrote = {k for k, c in keyed.items() if n in c["sources"] and not c["decoy"] and sel(k)}
            tp = len(wrote & tr)
            row[n] = (_f1(tp, len(wrote), len(tr)), tp / max(len(wrote), 1), tp / max(len(tr), 1))
        res[label] = {"true": len(tr), "sources": row}
    return res


def pair(out, a: str, b: str, judges, boot: int = 4000) -> tuple[float, float, float, int]:
    """Paired document-bootstrap difference of judged F1, B minus A, both in one pool: (difference, low, high, docs)."""
    keyed, verdicts, _ = load(out, judges)
    true = judged_true(keyed, verdicts, judges)
    per = _per_doc(keyed, true, [a, b])
    docs = sorted({i for i, _ in keyed})
    rng = random.Random(1)
    ds = []
    for _ in range(boot):
        s = rng.choices(docs, k=len(docs))
        ds.append(_sample_f1(per, s, b) - _sample_f1(per, s, a))
    ds.sort()
    return _sample_f1(per, docs, b) - _sample_f1(per, docs, a), ds[int(0.025 * boot)], ds[int(0.975 * boot) - 1], len(docs)


def add(out, name: str, path: str | Path) -> int:
    """Add one more system to an existing pool: its relations already in the pool get its name as a source; the rest
    become new claims in OUT/items_add.jsonl for the judges (verdicts_add_<judge>.jsonl, then `merge`). Earlier
    verdicts are kept, so systems stay comparable in one pool. Returns the number of new claims."""
    out = Path(out)
    items_, keys = _read(out / "items.jsonl"), _read(out / "key.jsonl")
    new = {d["id"]: d for d in _read(path)}
    added = 0
    with open(out / "items_add.jsonl", "w") as wa:
        for it, kd in zip(items_, keys):
            d = new.get(it["id"], {"entities": [], "relations": []})
            ents = [{"name": c[x], "mentions": [c[x]]} for c in it["claims"] for x in ("source", "target")]
            can = canon([{"entities": ents}, d])
            have = {(can(c["source"]), c["type"], can(c["target"])): c["n"] for c in it["claims"]}
            fresh = []
            for r in d["relations"]:
                k = (can(r["source"]), r["type"], can(r["target"]))
                if k in have:
                    kc = next(x for x in kd["claims"] if x["n"] == have[k])
                    if name not in kc["sources"] and not kc["decoy"]:
                        kc["sources"].append(name)
                    continue
                n = max([c["n"] for c in it["claims"]], default=-1) + 1
                c = {"n": n, "source": r["source"], "type": r["type"], "target": r["target"]}
                it["claims"].append(c)
                kd["claims"].append({"n": n, "sources": [name], "decoy": ""})
                have[k] = n
                fresh.append(c)
                added += 1
            wa.write(json.dumps({"id": it["id"], "text": it["text"], "claims": fresh}, ensure_ascii=False) + "\n")
    for f, rows in (("items", items_), ("key", keys)):
        with open(out / f"{f}.jsonl", "w") as w:
            for r in rows:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
    return added


def merge(out, judges) -> None:
    """Fold verdicts_add_<judge>.jsonl into verdicts_<judge>.jsonl."""
    out = Path(out)
    for j in judges:
        base = {d["id"]: d for d in _read(out / f"verdicts_{j}.jsonl")}
        for d in _read(out / f"verdicts_add_{j}.jsonl"):
            base.setdefault(d["id"], {"id": d["id"], "verdicts": []})["verdicts"] += d["verdicts"]
        with open(out / f"verdicts_{j}.jsonl", "w") as w:
            for d in base.values():
                w.write(json.dumps(d, ensure_ascii=False) + "\n")


def export(results: str | Path, out: str | Path, chunks: str | Path | None = None, scores: str | Path | None = None,
           cut: float | None = None, margin: float | None = None) -> int:
    """`relweave generate` output as a labels file (per chunk, entities named, asserted relations only); with CHUNKS
    and SCORES the union of generator and head at CUT and MARGIN, as `relweave eval --scores` computes it. With CHUNKS
    each record also carries the chunk's text, so the file can serve as a pool's gold. Returns the relation count."""
    texts = {}
    if chunks is not None:
        from relweave.training.data import load_chunks
        cs = load_chunks(Path(chunks))
        texts = {c.chunk_id: c.text for c in cs}
    if scores is not None:
        from relweave.formats import load_results
        from relweave.union import CUT, MARGIN, union
        schemas = {c.chunk_id: c.schema for c in cs}
        res = load_results(Path(results), texts, schema=schemas)
        sc = json.loads(Path(scores).read_text())
        graphs = {}
        for r in res:
            if r.output is not None:
                graphs.update(union({r.chunk_id: r}, sc, CUT if cut is None else cut, MARGIN if margin is None else margin,
                                    schema=schemas[r.chunk_id]))
        graphs = {cid: r.output.model_dump() for cid, r in graphs.items()}
    else:
        graphs = {d["chunk_id"]: d.get("output") for d in _read(results)}
    n = 0
    with open(out, "w") as w:
        for cid, g in graphs.items():
            g = g or {"entities": [], "relations": []}
            name = {e["id"]: e["name"] for e in g["entities"]}
            rels = [{"source": name[r["source"]], "type": r["type"], "target": name[r["target"]]}
                    for r in g["relations"] if (r.get("modality") or "asserted") == "asserted"
                    and r["source"] in name and r["target"] in name]
            n += len(rels)
            rec = {"id": cid, "entities": [{"name": e["name"], "type": e["type"], "mentions": e.get("mentions") or [e["name"]]}
                                           for e in g["entities"]], "relations": rels}
            if cid in texts:
                rec["text"] = texts[cid]
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return n


def labels_from_chunks(chunks: str | Path, out: str | Path) -> int:
    """A labelled chunk file (gold in relweave's chunk format) as a labels file with text, to serve as a pool's gold.
    Returns the relation count."""
    from relweave.training.data import load_chunks
    n = 0
    with open(out, "w") as w:
        for c in load_chunks(Path(chunks)):
            if c.gold is None:
                continue
            g = c.gold.model_dump() if hasattr(c.gold, "model_dump") else c.gold
            name = {e["id"]: e["name"] for e in g["entities"]}
            rels = [{"source": name[r["source"]], "type": r["type"], "target": name[r["target"]]}
                    for r in g["relations"] if r["source"] in name and r["target"] in name]
            n += len(rels)
            w.write(json.dumps({"id": c.chunk_id, "text": c.text, "entities": [
                {"name": e["name"], "type": e["type"], "mentions": e.get("mentions") or [e["name"]]} for e in g["entities"]],
                "relations": rels}, ensure_ascii=False) + "\n")
    return n
