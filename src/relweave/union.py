"""Union of the generator and the pair head: keep a relation the generator wrote unless the head scores it at or
below CUT (lenient: drops only what the head firmly rejects), and add every relation the head scores above MARGIN on
its own (legal for the endpoint types, not already present, up to the per-chunk cap). The defaults were chosen on
validation for the 4B generator and head (MARGIN); CUT was raised to 0 for real documents (see CUT).

ALT_TYPES: an entity left without any relation is tried as each other of Person/Org/Place; if the head scores a
relation above MARGIN that is legal only under that type ("Kappahl's sustainability director, Sandra Roos" with
Kappahl written as a Person), the entity takes that type and the relations are added."""

from __future__ import annotations

from relweave.schema import BUSINESS, MAX_RELATIONS_PER_CHUNK, ExtractionOutput, Schema
from relweave.schema.output import Relation

CUT = 0.0  # the head vetoes a generated relation it does not lean towards: on real documents -2 -> 0 cut a third of the
# false relations (0.624 -> 0.648 F1), Wikipedia unchanged (0.755 -> 0.756); the published head's config still says -2
MARGIN = 0.5
ALT_TYPES = True  # switchable for evaluation
RETEST = True  # an unlinked entity is re-scored by the head as each other type (ChunkExtractor._try_other_types)
RETEST_MAX = 4  # unlinked entities re-scored per chunk (each costs one head pass per other type)
SWAP = ("Person", "Org", "Place")
MAX_UNION_RELATIONS = 400  # relations per chunk after the head's additions. The generator's own cap (40, in its
# decoding grammar) stops a looping model; the head scores pairs and cannot loop, and on a dense list the generator's
# lines alone reach 40, so the shared cap blocked every relation of the names it had skipped (2026-10-16).


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
            if len(keep) >= MAX_UNION_RELATIONS or v <= margin:
                break
            typ, a, b = key.split()
            a, b = f"e{a}", f"e{b}"
            if a != b and a in et and b in et and schema.legal(typ, et[a], et[b]) and (typ, frozenset((a, b))) not in have:
                keep.append(Relation(type=typ, source=a, target=b, evidence=""))
                have.add((typ, frozenset((a, b))))
        ents = list(r.output.entities)
        if ALT_TYPES:
            linked = {x.source for x in keep} | {x.target for x in keep}
            for k, e in enumerate(ents):
                if e.id in linked or e.type not in SWAP:
                    continue
                best = None  # (score, type, relations) of the best other type
                for alt in SWAP:
                    if alt == e.type:
                        continue
                    t2 = {**et, e.id: alt}
                    found = []
                    for key, v in s.items():
                        typ, a, b = key.split()
                        a, b = f"e{a}", f"e{b}"
                        if v > margin and e.id in (a, b) and a != b and a in t2 and b in t2 \
                                and schema.legal(typ, t2[a], t2[b]) and not schema.legal(typ, et[a], et[b]):
                            found.append((v, typ, a, b))
                    if found and (best is None or max(found)[0] > best[0]):
                        best = (max(found)[0], alt, found)
                if best and len(keep) < MAX_UNION_RELATIONS:
                    ents[k] = e.model_copy(update={"type": best[1]})
                    et[e.id] = best[1]
                    for v, typ, a, b in sorted(best[2], reverse=True):
                        if len(keep) >= MAX_UNION_RELATIONS or (typ, frozenset((a, b))) in have:
                            continue
                        keep.append(Relation(type=typ, source=a, target=b, evidence=""))
                        have.add((typ, frozenset((a, b))))
        # every relation was checked for legality above; the output caps belong to the generator, not the union
        o = ExtractionOutput.model_construct(entities=ents, relations=keep)
        out[cid] = type(r)(chunk_id=cid, raw=r.raw, output=o, fmt="sentences")
    return out
