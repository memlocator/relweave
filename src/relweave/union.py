"""Union of the generator and the pair head: keep a relation the generator wrote unless the head scores it at or
below CUT (lenient: drops only what the head firmly rejects), and add every relation the head scores above MARGIN on
its own (legal for the endpoint types, not already present, up to the per-chunk cap). The defaults were chosen on
validation for the 4B generator and head."""

from __future__ import annotations

from relweave.schema import BUSINESS, MAX_RELATIONS_PER_CHUNK, ExtractionOutput, Schema
from relweave.schema.output import Relation

CUT = -2.0
MARGIN = 0.5


def union(res: dict, scores: dict, cut: float = CUT, margin: float = MARGIN, schema: Schema | None = None) -> dict:
    """res: chunk id -> ExtractionResult of the generator; scores: chunk id -> {"TYPE i j": head margin} (entity
    numbers). Returns chunk id -> a new result of the same class with the union's relations. schema: legality of
    added relations and the output's validation context (business by default)."""
    schema = schema or BUSINESS
    out = {}
    for cid, r in res.items():
        s = scores.get(cid, {})
        et = {e.id: e.type for e in r.output.entities}
        keep = [x for x in r.output.relations if s.get(f"{x.type} {x.source[1:]} {x.target[1:]}", -1e9) > cut]
        have = {(x.type, frozenset((x.source, x.target))) for x in keep}
        for key, v in sorted(s.items(), key=lambda kv: -kv[1]):
            if len(keep) >= MAX_RELATIONS_PER_CHUNK or v <= margin:
                break
            typ, a, b = key.split()
            a, b = f"e{a}", f"e{b}"
            if a != b and a in et and b in et and schema.legal(typ, et[a], et[b]) and (typ, frozenset((a, b))) not in have:
                keep.append(Relation(type=typ, source=a, target=b, evidence=""))
                have.add((typ, frozenset((a, b))))
        o = ExtractionOutput.model_validate({"entities": r.output.entities, "relations": keep}, context={"schema": schema})
        out[cid] = type(r)(chunk_id=cid, raw=r.raw, output=o, fmt="sentences")
    return out
