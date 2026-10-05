"""Merge per-chunk graphs into one document graph: entity resolution, then relation rewriting.

Entity resolution, in order (union-find; two entities of one chunk are not joined, since the model wrote them
apart on purpose, except for a surname-only entity beside its full name):
  (a) overlap anchor: entities of different chunks, same type, sharing a NAME mention at the same document offset
      (pronouns and descriptions never anchor);
  (b) same type and the same normalised name or name-like mention (last word capitalised, not a role such as CEO);
      a one-word key (a surname) joins the unique full name containing it, and is skipped when several do;
  (c) an entity whose mentions are all generic descriptions joins the one entity of its type, in a chunk next to
      its own, that uses the same description; computed from a snapshot after (b); pure pronouns never join.
"""

from __future__ import annotations

import re

from relweave.schema import BUSINESS, Schema
from relweave.schema.output import canonical_triple
from relweave.chunk import sentence_at, sentence_spans
from relweave.graph import Entity, Evidence, Graph, Occurrence, Relation
from relweave.records import ChunkEntity, ChunkGraph

LEGAL_SUFFIXES = {"as", "asa", "ab", "oy", "aps", "ou", "gmbh", "ltd", "inc", "plc", "sa",
                  "ag", "nv", "bv", "se", "llc", "corp", "co", "limited"}
PRONOUNS = {"it", "he", "she", "they", "him", "her", "them", "his", "hers", "its", "their", "theirs", "we", "us",
            "our", "i", "you", "who", "which", "that", "this", "these", "those", "itself", "himself", "herself",
            "themselves"}
ROLE_WORDS = {"ceo", "cfo", "coo", "cto", "chairman", "chairwoman", "chair", "president", "vice", "minister", "director",
              "head", "manager", "founder", "owner", "executive", "secretary", "governor", "mayor", "leader", "officer",
              "spokesman", "spokesperson", "spokeswoman", "chief", "partner", "member", "employee", "employer",
              "representative", "adviser", "advisor", "board"}
FUNCTION_WORDS = {"and", "or", "of", "for", "in", "on", "at", "to", "by", "with", "from", "but", "not", "if", "as", "is",
                  "was", "are", "be", "the"}
SURNAME_TYPES = {"Person"}  # types whose one-word names are surnames
DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "its", "his", "her", "their", "our", "my", "your"}
GENERIC_NOUNS = {"company", "group", "unit", "area", "firm", "organisation", "organization", "corporation", "business",
                 "brand", "division", "subsidiary", "parent", "enterprise", "agency", "institution", "venture",
                 "city", "town", "country", "region", "state", "district", "site", "facility", "location", "place",
                 "man", "woman", "person", "executive", "founder", "owner", "event", "meeting", "deal", "project",
                 "team", "club", "party", "bank", "operator", "manufacturer", "supplier", "customer", "partner"}


def _tokens(s: str) -> list[str]:
    return re.findall(r"[\w'&-]+", s)


def normalise(name: str) -> str:
    """Casefolded name without punctuation, a leading "the" and trailing legal suffixes."""
    toks = [t.casefold().strip("'-&") for t in _tokens(name.replace(".", ""))]
    toks = [t for t in toks if t]
    if toks and toks[0] == "the":
        toks = toks[1:]
    while len(toks) > 1 and toks[-1] in LEGAL_SUFFIXES:
        toks.pop()
    return " ".join(toks)


def is_pronoun(mention: str) -> bool:
    """A pronoun, in any case but not as an all-caps abbreviation (US, WHO, IT)."""
    toks = _tokens(mention)
    return len(toks) == 1 and toks[0].casefold() in PRONOUNS and not (len(toks[0]) > 1 and toks[0].isupper())


def is_generic(mention: str) -> bool:
    """A pronoun or a generic description ("the company", "its parent company"); a name is never generic."""
    toks = _tokens(mention)
    if is_pronoun(mention):
        return True
    if toks and toks[0].casefold() in DETERMINERS:
        toks = toks[1:]
    elif len(toks) != 1:
        return False
    # a description ends in a generic noun and has no capitalised word (which would make it a name)
    return bool(toks) and toks[-1].casefold() in GENERIC_NOUNS and all(t[0].islower() for t in toks)


def is_key(mention: str) -> bool:
    """A mention that can identify an entity by itself: a name whose last word is capitalised and is not a role or
    title ("Amundsen" yes; "the CEO", "the chairman", "the plant", "the Norwegian company" no)."""
    toks = _tokens(mention)
    if not toks or is_generic(mention) or not toks[-1][0].isupper() or toks[-1].casefold() in ROLE_WORDS:
        return False
    # a determiner-led phrase ("The Company") and a bare function word ("AND", "THE") are not names
    return toks[0].casefold() not in DETERMINERS and not (len(toks) == 1 and toks[0].casefold() in FUNCTION_WORDS)


def description(mention: str) -> str:
    """The comparison key of a generic mention: casefolded, determiner dropped ("The company" -> "company")."""
    toks = [t.casefold() for t in _tokens(mention)]
    return " ".join(toks[1:] if len(toks) > 1 and toks[0] in DETERMINERS else toks)


class _Forest:
    """Union-find over (chunk, local id) nodes that refuses a union joining two entities of one chunk."""

    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}
        self.chunks = {n: {n[0]} for n in nodes}

    def find(self, n):
        while self.parent[n] != n:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def union(self, a, b, force: bool = False) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self.chunks[ra] & self.chunks[rb] and not force:
            return False
        self.parent[rb] = ra
        self.chunks[ra] |= self.chunks.pop(rb)
        return True


def _resolve(graphs: list[ChunkGraph], log: list | None = None) -> _Forest:
    ents = {(g.chunk, e.id): e for g in graphs for e in g.entities}
    forest = _Forest(ents)
    rule = "a"

    def union(a, b, force=False):
        """forest.union that records which rule joined which entities (log: (rule, node, node))."""
        if log is not None and forest.find(a) != forest.find(b) and (
                force or not forest.chunks[forest.find(a)] & forest.chunks[forest.find(b)]):
            log.append((rule, a, b))
        return forest.union(a, b, force)

    # (a) overlap anchor: the same name at the same document offset, same type; pronouns and descriptions are
    # never anchors (the same "It" position can belong to different entities of different chunks)
    anchor: dict[tuple, tuple] = {}
    for node, e in ents.items():
        for m in e.mentions:
            if m.start is not None and not is_generic(m.text):
                key = (e.type, m.start, m.text.casefold())
                if key in anchor:
                    union(anchor[key], node)
                else:
                    anchor[key] = node
    rule = "b"
    # (b) same type and normalised name; a single-word key (a surname) needs a unique full name
    keys: dict[tuple, set] = {}  # node -> normalised name keys
    for node, e in ents.items():
        k = {normalise(e.name)} if not is_generic(e.name) else set()
        k |= {normalise(m.text) for m in e.mentions if is_key(m.text)}
        keys[node] = k - {""}
    full: dict[tuple, dict[str, list]] = {}  # (type, key) multi-word -> {key: [nodes]}
    single: dict[tuple, list] = {}
    for node, ks in keys.items():
        for k in ks:
            if " " in k:
                full.setdefault((ents[node].type, k), []).append(node)
            else:
                single.setdefault((ents[node].type, k), []).append(node)
    for nodes in full.values():
        for n in nodes[1:]:
            union(nodes[0], n)
    for (typ, k), nodes in single.items():
        names = {f for (t, f) in full if t == typ and k in f.split()}
        if len(names) > 1:
            continue  # ambiguous: several people (companies, ...) carry this word
        if names:
            owners = full[(typ, names.pop())]
            for n in nodes:
                # only people have surnames: a stray surname-only person beside the full name joins it even within
                # one chunk; other types (Telenor / Telenor Norge) keep the within-chunk guard
                union(owners[0], n, force=typ in SURNAME_TYPES)
        else:
            for n in nodes[1:]:
                union(nodes[0], n)
    rule = "c"
    # (c) all-generic entities join the single nearby entity that uses the same description. Pure pronouns never
    # join; candidates are read from a snapshot, so the outcome does not depend on the order
    snap = {n: forest.find(n) for n in ents}
    comp_chunks = {r: set(forest.chunks[r]) for r in set(snap.values())}
    uses: dict[tuple, set] = {}  # (type, description) -> snapshot components that use it
    for node, e in ents.items():
        for m in e.mentions:
            if is_generic(m.text) and not is_pronoun(m.text):
                uses.setdefault((e.type, description(m.text)), set()).add(snap[node])
    joins = []
    for node, e in ents.items():
        if not e.mentions or not all(is_generic(m.text) for m in e.mentions):
            continue
        wanted = {description(m.text) for m in e.mentions if not is_pronoun(m.text)}
        mine = snap[node]
        cands = {c for d in wanted for c in uses.get((e.type, d), ())
                 if c != mine and not comp_chunks[c] & comp_chunks[mine]
                 and any(abs(node[0] - k) <= 1 for k in comp_chunks[c])}
        if len(cands) == 1:
            joins.append((mine, cands.pop()))
    for a, b in joins:
        union(a, b)
    return forest


def _best_name(members: list[ChunkEntity]) -> str:
    names = [e.name for e in members if not is_generic(e.name)] or [e.name for e in members]
    return max(names, key=len)  # the fullest name; the first of equals


def _groups(graphs: list[ChunkGraph]):
    """(forest, root -> [(chunk, entity)] in document order, root -> merged id "e1", "e2", ...)."""
    forest = _resolve(graphs)
    members: dict[tuple, list[tuple[int, ChunkEntity]]] = {}
    for g in graphs:
        for e in g.entities:
            members.setdefault(forest.find((g.chunk, e.id)), []).append((g.chunk, e))
    return forest, members, {root: f"e{k}" for k, root in enumerate(members, 1)}


def union_log(graphs: list[ChunkGraph]) -> list[tuple[str, tuple, tuple]]:
    """Every join entity resolution makes, as (rule "a"/"b"/"c", (chunk, id), (chunk, id)); for evaluation."""
    log: list = []
    _resolve(sorted(graphs, key=lambda g: g.chunk), log)
    return log


def entity_groups(graphs: list[ChunkGraph]) -> dict[tuple[int, str], str]:
    """(chunk index, local entity id) -> the merged entity id merge() gives it."""
    graphs = sorted(graphs, key=lambda g: g.chunk)
    forest, members, new_id = _groups(graphs)
    return {(c, e.id): new_id[root] for root, ms in members.items() for c, e in ms}


def _doc_sentence_for_relation(chunk_entities: dict[str, ChunkEntity], r, sents, lo: int, hi: int):
    """The first sentence of the chunk that holds a located mention of both endpoints."""
    def inside(e, a, b):
        return any(m.start is not None and a <= m.start < b for m in e.mentions)
    s, t = chunk_entities.get(r.source), chunk_entities.get(r.target)
    if s is None or t is None:
        return None
    return next(((a, b) for a, b in sents if lo <= a < hi and inside(s, a, b) and inside(t, a, b)), None)


def merge(graphs: list[ChunkGraph], text: str, schema: Schema | None = None, doc_id: str = "document",
          model: dict | None = None, version: str = "") -> Graph:
    """One document graph from the chunk graphs of `text` (see the module docstring)."""
    schema = schema or BUSINESS
    graphs = sorted(graphs, key=lambda g: g.chunk)
    forest, members, new_id = _groups(graphs)
    sents = sentence_spans(text)
    entities = []
    for root, ms in members.items():
        eid = new_id[root]
        attrs: dict[str, str] = {}
        for _, e in ms:
            for k, v in e.attributes.items():
                attrs.setdefault(k, v)
        occ, seen = [], set()
        for c, e in ms:
            for m in e.mentions:
                if m.start is not None and text[m.start:m.start + len(m.text)] == m.text and (m.text, m.start) not in seen:
                    seen.add((m.text, m.start))
                    occ.append(Occurrence(m.text, m.start, m.start + len(m.text), c, sentence_at(sents, m.start)))
        occ.sort(key=lambda o: (o.start, o.text))
        found = {o.text for o in occ}
        aliases = list(dict.fromkeys(m.text for _, e in ms for m in e.mentions if m.text not in found))
        entities.append(Entity(eid, ms[0][1].type, _best_name([e for _, e in ms]), occ, attrs,
                               sorted({c for c, _ in ms}), aliases))
    relations: dict[tuple, Relation] = {}
    for g in graphs:
        local = {e.id: e for e in g.entities}
        for r in g.relations:
            ends = [(g.chunk, r.source), (g.chunk, r.target)]
            if not all(n in forest.parent for n in ends):
                continue  # a relation naming an entity the chunk never listed
            s, t = (new_id[forest.find(n)] for n in ends)
            if s == t:
                continue
            sent = r.sentence or _doc_sentence_for_relation(local, r, sents, g.start, g.end or len(text))
            key = canonical_triple(r.type, s, t, schema)
            cur = relations.get(key)
            if cur is None:
                relations[key] = Relation(r.type, s, t, r.modality, dict(r.attributes), r.score, r.origin,
                                          [Evidence(g.chunk, sent)])
                continue
            if all(v.chunk != g.chunk for v in cur.evidence):
                cur.evidence.append(Evidence(g.chunk, sent))
            if r.score is not None and (cur.score is None or r.score > cur.score):
                cur.score = r.score
            if r.origin == "generator":
                cur.origin = "generator"
            for k, v in r.attributes.items():
                cur.attributes.setdefault(k, v)
    spans = [{"index": g.chunk, "start": g.start, "end": g.end} for g in graphs]
    meta = {"name": schema.name, "entity_types": list(schema.node_names()), "relation_types": list(schema.edge_names()),
            "symmetric_types": [n for n in schema.edge_names() if schema.symmetric(n)]}
    return Graph(entities=entities, relations=list(relations.values()), text=text,
                 document={"id": doc_id, "chunks": spans}, schema=meta, model=model or {}, version=version,
                 warnings=[w for g in graphs for w in g.warnings])
