"""A simple view of the business schema: people, organisations and places, and how they relate, in six relation
types that group the schema's existing Person/Org/Place relation types (no new relation is introduced; the official
types stay the fine ones). coarsen() maps a graph to this view, keeping the fine type as the attribute `fine_type`;
the evaluation reports it as a second score with coarse_type().

  AFFILIATED_WITH  Person -> Org     works for, leads, sits on the board of, is a member of, or founded the Org
  OWNS             Person|Org -> Org owns a stake in, controls, or bought the Org (or, for an Org, founded it)
  PART_OF          Org -> Org        is a subsidiary, division, unit or brand of the Org
  LOCATED_IN       Person|Org|Place -> Place   lives, is based, operates or lies in the Place, or was born there
  FAMILY_OF        Person - Person   a family tie (symmetric)
  ASSOCIATED_WITH  Person - Person   associates, met, communicated (ASSOCIATE_OF, MET_WITH, COMMUNICATED_WITH; symmetric)
"""
from __future__ import annotations

from dataclasses import replace

IN_SCOPE = ("Person", "Org", "Place")

TYPES = {  # name: (sources, targets, symmetric, definition)
    "AFFILIATED_WITH": (("Person",), ("Org",), False,
                        "Person works for, leads, sits on the board of, is a member of, or founded the Org; "
                        "attributes: role"),
    "OWNS": (("Person", "Org"), ("Org",), False,
             "Person or Org owns a stake in, controls or bought the Org, or (an Org) founded it; attributes: share"),
    "PART_OF": (("Org",), ("Org",), False, "Org is a subsidiary, division, unit or brand of the target Org"),
    "LOCATED_IN": (("Person", "Org", "Place"), ("Place",), False,
                   "lives, is based, operates or lies in the Place, or was born there"),
    "FAMILY_OF": (("Person",), ("Person",), True, "a family tie; attributes: kind"),
    "ASSOCIATED_WITH": (("Person",), ("Person",), True, "persons described as associates or partners, or who met or "
                        "communicated"),
}

_FINE = {"EMPLOYED_BY": "AFFILIATED_WITH", "EXECUTIVE_OF": "AFFILIATED_WITH", "BOARD_MEMBER_OF": "AFFILIATED_WITH",
         "MEMBER_OF": "AFFILIATED_WITH", "OWNS_STAKE_IN": "OWNS", "ACQUIRED": "OWNS", "SUBSIDIARY_OF": "PART_OF",
         "LOCATED_IN": "LOCATED_IN", "HEADQUARTERED_IN": "LOCATED_IN", "OPERATES_IN": "LOCATED_IN",
         "BORN_IN": "LOCATED_IN", "FAMILY_OF": "FAMILY_OF", "ASSOCIATE_OF": "ASSOCIATED_WITH",
         "MET_WITH": "ASSOCIATED_WITH", "COMMUNICATED_WITH": "ASSOCIATED_WITH"}


def coarse_type(fine: str, source_type: str | None = None) -> str | None:
    """The simple-ontology type of a business-schema relation type, or None when it is out of scope (events,
    objects, coordinates). FOUNDED depends on the founder: a Person is AFFILIATED_WITH, an Org OWNS."""
    if fine == "FOUNDED":
        return "OWNS" if source_type == "Org" else "AFFILIATED_WITH"
    return _FINE.get(fine)


def symmetric(coarse: str) -> bool:
    return TYPES[coarse][2]


def coarsen(graph):
    """The graph with in-scope relations retyped to the simple ontology (fine type kept as attribute `fine_type`),
    out-of-scope relations and entities (Event, Object, Coordinate) dropped, duplicates merged."""
    types = {e.id: e.type for e in graph.entities}
    keep = {e.id for e in graph.entities if e.type in IN_SCOPE}
    rels, seen = [], set()
    for r in graph.relations:
        c = coarse_type(r.type, types.get(r.source))
        if c is None or r.source not in keep or r.target not in keep:
            continue
        s, t = (sorted((r.source, r.target)) if symmetric(c) else (r.source, r.target))
        if (s, c, t) in seen:
            continue
        seen.add((s, c, t))
        rels.append(replace(r, type=c, source=s, target=t, attributes={**r.attributes, "fine_type": r.type}))
    return replace(graph, entities=[e for e in graph.entities if e.id in keep], relations=rels)


def render() -> str:
    """The ontology as prompt or brief text."""
    lines = ["Entity types: Person (a human being); Org (a company, institution, party, team, court, agency or other "
             "organised body); Place (a country, region, city or other location).", "Relation types (source -> target):"]
    for name, (src, tgt, sym, d) in TYPES.items():
        lines.append(f"- {name}: {'|'.join(src)} {'-' if sym else '->'} {'|'.join(tgt)}. {d}")
    return "\n".join(lines)
