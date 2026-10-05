"""Chunk-level scores of extraction results against gold: entities (exact spans, names), coreference, relations
(endpoints aligned to gold entities by span overlap; also by name, by pair, and per relation type).

A chunk is anything with .text, .gold (ExtractionOutput) and .schema (LabelledChunk from relweave.training.data):
symmetric relation types are compared without direction under the chunk's own schema.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

from relweave.formats import ExtractionResult, display_name, resolve_spans, sentence_bounds
from relweave.schema import ExtractionOutput, canonical_triple


def _f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def spans_of(text: str, out: ExtractionOutput) -> dict[str, set[tuple[int, int, str]]]:
    """entity local id -> set of (start, end, type)."""
    spans, _ = resolve_spans(text, out)
    return {e.id: {(a, b, str(e.type)) for a, b in spans[e.id]} for e in out.entities}


def sentence_distance(text: str, out: ExtractionOutput, source: str, target: str) -> int | None:
    """Fewest sentences between any mention of entity `source` and any mention of `target` (0 = same sentence)."""
    bounds = sentence_bounds(text)
    idx = lambda pos: next((i for i, (a, b) in enumerate(bounds) if a <= pos < b), len(bounds) - 1)  # noqa: E731
    spans = spans_of(text, out)
    a, b = spans.get(source, []), spans.get(target, [])
    if not a or not b:
        return None
    return min(abs(idx(x[0]) - idx(y[0])) for x in a for y in b)


def _norm(name: str) -> str:
    return " ".join(name.lower().replace(".", "").split())


def _label(e) -> str:
    """Name used by the relaxed metrics: the entity's longest surface in the chunk.

    Gold canonical names come from the world and may never appear in the text, and
    formats differ in whether they emit a name at all, so both sides use surfaces.
    """
    return _norm(display_name(e.mentions))


@dataclass
class Counts:
    ent_strict: list[int]   # exact mention spans with type
    ent_relaxed: list[int]  # entity-level, normalised canonical name with type
    coref: list[int]        # mention pairs grouped under one entity
    rel_strict: list[int]   # type + endpoints aligned to gold entities by span overlap
    rel_relaxed: list[int]  # type + endpoints matched by normalised canonical name
    rel_pair: list[int]     # are these two entities related at all: aligned endpoint pairs, type and direction ignored
    rel_type: list[int]     # [right type, wrong type] over the pairs found in both (type given the pair is right)
    rel_by_type: dict[str, list[int]] = field(default_factory=dict)  # strict relation [tp, fp, fn] per relation type


def align(text: str, gold: ExtractionOutput, pred: ExtractionOutput) -> dict[str, str | None]:
    """Map predicted entity id -> gold entity id by best span overlap with matching type."""
    g_spans = spans_of(text, gold)
    p_spans = spans_of(text, pred)
    g_type = {e.id: e.type for e in gold.entities}
    p_type = {e.id: e.type for e in pred.entities}
    mapping: dict[str, str | None] = {}
    for pid, ps in p_spans.items():
        best, best_n = None, 0
        for gid, gs in g_spans.items():
            if g_type[gid] != p_type[pid]:
                continue
            # characters shared by overlapping mentions, so "Wallenberg family" still matches
            # "the Wallenberg family" (identical spans only, as before, missed such near-misses)
            n = sum(max((max(0, min(pe, ge) - max(ps_, gs_)) for gs_, ge, _ in gs), default=0) for ps_, pe, _ in ps)
            if n > best_n:
                best, best_n = gid, n
        mapping[pid] = best
    return mapping


def score_chunk(chunk, result: ExtractionResult) -> Counts:
    gold = chunk.gold
    assert gold is not None
    text = chunk.text
    pred = result.output or ExtractionOutput()

    # entities, strict: exact (start, end, type) mention sets
    g_m = set().union(*spans_of(text, gold).values()) if gold.entities else set()
    p_m = set().union(*spans_of(text, pred).values()) if pred.entities else set()
    ent_strict = [len(g_m & p_m), len(p_m - g_m), len(g_m - p_m)]

    # entities, relaxed: normalised (name, type) at entity level
    g_e = {(_label(e), str(e.type)) for e in gold.entities}
    p_e = {(_label(e), str(e.type)) for e in pred.entities}
    ent_relaxed = [len(g_e & p_e), len(p_e - g_e), len(g_e - p_e)]

    # coreference: pairwise over mention spans (which pairs share an entity)
    g_pairs = set()
    for spans in spans_of(text, gold).values():
        for a, b in combinations(sorted((s, e) for s, e, _ in spans), 2):
            g_pairs.add((a, b))
    p_pairs = set()
    for spans in spans_of(text, pred).values():
        for a, b in combinations(sorted((s, e) for s, e, _ in spans), 2):
            p_pairs.add((a, b))
    coref = [len(g_pairs & p_pairs), len(p_pairs - g_pairs), len(g_pairs - p_pairs)]

    # relations
    mapping = align(text, gold, pred)
    schema = chunk.schema
    g_r = {canonical_triple(r.type, r.source, r.target, schema) for r in gold.relations}
    p_r_mapped = set()
    for r in pred.relations:
        p_r_mapped.add(canonical_triple(r.type, mapping.get(r.source), mapping.get(r.target), schema))
    matched = {t for t in p_r_mapped if t in g_r}
    rel_strict = [len(matched), len(p_r_mapped) - len(matched), len(g_r - matched)]
    rel_pair, rel_type = _pair_counts(g_r, p_r_mapped)

    g_name = {e.id: (_label(e), str(e.type)) for e in gold.entities}
    p_name = {e.id: (_label(e), str(e.type)) for e in pred.entities}
    g_rn = {canonical_triple(r.type, g_name[r.source], g_name[r.target], schema) for r in gold.relations}
    p_rn = {canonical_triple(r.type, p_name[r.source], p_name[r.target], schema) for r in pred.relations}
    rel_relaxed = [len(g_rn & p_rn), len(p_rn - g_rn), len(g_rn - p_rn)]

    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for t in matched:
        by_type[t[0]][0] += 1
    for t in p_r_mapped - matched:
        by_type[t[0]][1] += 1
    for t in g_r - matched:
        by_type[t[0]][2] += 1
    return Counts(ent_strict, ent_relaxed, coref, rel_strict, rel_relaxed, rel_pair, rel_type, dict(by_type))


def _pair_counts(gold: set[tuple], pred: set[tuple]) -> tuple[list[int], list[int]]:
    """Relation scoring split in two: detection (is the pair related at all, as tp/fp/fn over
    unordered endpoint pairs) and typing (of the pairs both sides relate, how many carry a
    predicted triple that gold has: right type and direction)."""
    def pairs(triples):
        out: dict[frozenset, set] = {}
        for t in triples:
            if t[1] is not None and t[2] is not None:
                out.setdefault(frozenset(t[1:]), set()).add(t)
        return out
    g, p = pairs(gold), pairs(pred)
    n_unaligned = sum(1 for t in pred if t[1] is None or t[2] is None)  # endpoint not a gold entity: a wrong pair
    both = g.keys() & p.keys()
    right = sum(1 for k in both if g[k] & p[k])
    return [len(both), len(p.keys() - g.keys()) + n_unaligned, len(g.keys() - p.keys())], [right, len(both) - right]


def summarize(chunks: list, results: list[ExtractionResult], by_type: bool = False) -> dict:
    """Pooled scores. by_type=True adds "rel_by_type": {relation type: {tp, fp, fn, f1}} (strict relations)."""
    by_id = {r.chunk_id: r for r in results}
    totals = {k: [0, 0, 0] for k in ("ent_strict", "ent_relaxed", "coref", "rel_strict", "rel_relaxed", "rel_pair")}
    typed = [0, 0]
    per_type: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    valid = 0
    unresolved_m = unresolved_e = total_m = total_e = illegal = 0
    seconds = 0.0
    for c in chunks:
        r = by_id.get(c.chunk_id)
        if r is None:
            r = ExtractionResult(chunk_id=c.chunk_id, raw="", output=None, error="missing")
        if r.output is not None:
            valid += 1
            total_m += sum(len(set(e.mentions)) for e in r.output.entities) + r.unresolved_mentions
            total_e += len(r.output.relations) + r.unresolved_evidence + r.illegal_relations
            illegal += r.illegal_relations
        unresolved_m += r.unresolved_mentions
        unresolved_e += r.unresolved_evidence
        seconds += r.seconds
        counts = score_chunk(c, r)
        for k in totals:
            for i in range(3):
                totals[k][i] += getattr(counts, k)[i]
        for t, v in counts.rel_by_type.items():
            for i in range(3):
                per_type[t][i] += v[i]
        typed = [typed[0] + counts.rel_type[0], typed[1] + counts.rel_type[1]]
    out = {k: _f1(*v) for k, v in totals.items()}
    n_both = sum(typed)
    out["rel_type"] = {"accuracy": round(typed[0] / n_both, 4) if n_both else 0.0, "right": typed[0], "pairs": n_both}
    out["schema_validity"] = round(valid / len(chunks), 4) if chunks else 0.0
    out["hallucinated_mentions"] = round(unresolved_m / total_m, 4) if total_m else 0.0
    out["hallucinated_evidence"] = round(unresolved_e / total_e, 4) if total_e else 0.0
    out["illegal_relations"] = round(illegal / total_e, 4) if total_e else 0.0
    out["chunks"] = len(chunks)
    out["chunks_per_minute"] = round(60 * len(chunks) / seconds, 2) if seconds else None
    if by_type:
        out["rel_by_type"] = {t: {"tp": v[0], "fp": v[1], "fn": v[2], "f1": _f1(*v)["f1"]}
                              for t, v in sorted(per_type.items())}
    return out


def evaluate(chunks_path, results_path, scores_path=None, cut: float | None = None, margin: float | None = None,
             by_type: bool = False) -> dict:
    """Scores of saved generator results on a labelled chunk file, each chunk under its own schema: {"generator":
    summary} and, with head scores, {"union": summary} as well (relweave.union with cut and margin, default
    relweave.union.CUT and MARGIN)."""
    import json
    from pathlib import Path

    from relweave.formats import load_results
    from relweave.training.data import load_chunks
    from relweave.union import CUT, MARGIN, union
    chunks = [c for c in load_chunks(Path(chunks_path)) if c.gold is not None]
    schemas = {c.chunk_id: c.schema for c in chunks}
    res = load_results(Path(results_path), {c.chunk_id: c.text for c in chunks}, schema=schemas)
    out = {"generator": summarize(chunks, res, by_type)}
    if scores_path is not None:
        scores = json.loads(Path(scores_path).read_text())
        cut, margin = CUT if cut is None else cut, MARGIN if margin is None else margin
        u = {}
        for r in res:
            if r.output is not None and r.chunk_id in schemas:
                u.update(union({r.chunk_id: r}, scores, cut, margin, schema=schemas[r.chunk_id]))
        out["union"] = summarize(chunks, list(u.values()), by_type)
        out["union"]["cut"], out["union"]["margin"] = cut, margin
    return out
