"""The document graph and its JSON Graph Format (v2) serialisation (https://jsongraphformat.info/).

Entities are nodes (with every located mention: exact document offsets and the containing sentence), relations are
edges (with the evidence sentences), plus one document node that every entity is MENTIONED_IN. JGF allows only a few
keys per object, so everything else lives in `metadata`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_FILE = Path(__file__).with_name("jgf-v2.schema.json")  # vendored: validation needs no network
DOC = "doc"  # id of the document node


@dataclass
class Occurrence:
    """One mention at an exact place: text == document[start:end]."""
    text: str
    start: int
    end: int
    chunk: int  # the chunk it was read from
    sentence: tuple[int, int] | None = None  # document offsets of the containing sentence


@dataclass
class Evidence:
    chunk: int
    sentence: tuple[int, int] | None = None  # document offsets


@dataclass
class Entity:
    id: str
    type: str
    name: str
    mentions: list[Occurrence] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)
    chunks: list[int] = field(default_factory=list)  # chunks that mention it
    aliases: list[str] = field(default_factory=list)  # written mention strings with no located occurrence


@dataclass
class Relation:
    type: str
    source: str
    target: str
    modality: str = "asserted"
    attributes: dict[str, str] = field(default_factory=dict)
    score: float | None = None  # best pair-head margin over the chunks that state it
    origin: str = "generator"  # "generator", or "head" when only the head proposed it
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class Graph:
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    text: str = ""  # the document
    document: dict = field(default_factory=dict)  # {"id": ..., "chunks": [{"index", "start", "end"}]}
    schema: dict = field(default_factory=dict)  # {"name", "entity_types", "relation_types", "symmetric_types"}
    model: dict = field(default_factory=dict)  # model ids, free-form
    version: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def mentioned_in(self) -> list[dict]:
        """One record per (entity, mention, start, end): derived from the located mentions, never extracted."""
        return [{"entity": e.id, "mention": m.text, "start": m.start, "end": m.end}
                for e in self.entities for m in sorted(e.mentions, key=lambda m: (m.start, m.text))]

    # -- JSON Graph Format -------------------------------------------------------------------------------------

    def _sentence(self, span):
        return None if span is None else {"start": span[0], "end": span[1], "text": self.text[span[0]:span[1]]}

    def to_jgf(self, include_text: bool = True) -> dict:
        doc_id = self.document.get("id", "document")
        meta = {"schema": self.schema, "model": self.model, "relweave_version": self.version,
                "chars": len(self.text), "chunks": self.document.get("chunks", []), "warnings": self.warnings}
        if include_text:
            meta["text"] = self.text
        nodes = {e.id: {"label": e.name, "metadata": {
            "kind": "entity", "type": e.type, "attributes": e.attributes, "chunks": e.chunks, "aliases": e.aliases,
            "mentions": [{"text": m.text, "start": m.start, "end": m.end, "chunk": m.chunk, "located_by": "match",
                          "sentence": self._sentence(m.sentence)} for m in e.mentions]}} for e in self.entities}
        nodes[DOC] = {"label": doc_id, "metadata": {"kind": "document"}}
        sym = set(self.schema.get("symmetric_types", []))
        edges = [{"id": f"r{k}", "source": r.source, "target": r.target, "relation": r.type,
                  "directed": r.type not in sym, "label": r.type,
                  "metadata": {"kind": "relation", "modality": r.modality, "attributes": r.attributes,
                               "score": r.score, "origin": r.origin,
                               "evidence": [{"chunk": v.chunk, "sentence": self._sentence(v.sentence)}
                                            for v in r.evidence]}}
                 for k, r in enumerate(self.relations, 1)]
        edges += [{"id": f"m{k}", "source": e.id, "target": DOC, "relation": "MENTIONED_IN", "directed": True,
                   "label": "MENTIONED_IN", "metadata": {"kind": "mentioned_in", "count": len(e.mentions)}}
                  for k, e in enumerate(self.entities, 1)]
        return {"graph": {"id": doc_id, "label": doc_id, "directed": True, "type": "relweave", "metadata": meta,
                          "nodes": nodes, "edges": edges}}

    def to_json(self, indent: int | None = 1, include_text: bool = True) -> str:
        return json.dumps(self.to_jgf(include_text), indent=indent, ensure_ascii=False)

    @classmethod
    def from_jgf(cls, d: dict) -> Graph:
        g = d["graph"]
        meta = g.get("metadata", {})
        span = lambda s: None if s is None else (s["start"], s["end"])  # noqa: E731
        entities = []
        for nid, n in g.get("nodes", {}).items():
            m = n.get("metadata", {})
            if m.get("kind") != "entity":
                continue
            entities.append(Entity(nid, m["type"], n["label"],
                                   [Occurrence(o["text"], o["start"], o["end"], o["chunk"], span(o.get("sentence")))
                                    for o in m.get("mentions", [])],
                                   dict(m.get("attributes", {})), list(m.get("chunks", [])), list(m.get("aliases", []))))
        relations = []
        for e in g.get("edges", []):
            m = e.get("metadata", {})
            if m.get("kind") == "relation":
                relations.append(Relation(e["relation"], e["source"], e["target"], m.get("modality", "asserted"),
                                          dict(m.get("attributes", {})), m.get("score"), m.get("origin", "generator"),
                                          [Evidence(v["chunk"], span(v.get("sentence"))) for v in m.get("evidence", [])]))
        return cls(entities, relations, meta.get("text", ""), {"id": g.get("id", "document"), "chunks": meta.get("chunks", [])},
                   meta.get("schema", {}), meta.get("model", {}), meta.get("relweave_version", ""), list(meta.get("warnings", [])))

    @classmethod
    def from_json(cls, s: str) -> Graph:
        return cls.from_jgf(json.loads(s))

    # -- networkx, GraphML -------------------------------------------------------------------------------------

    def to_networkx(self):
        """MultiDiGraph derived from the JGF data: node attributes are label + metadata, edge attributes the edge's
        keys + metadata (a symmetric relation has directed=False; the graph itself is a MultiDiGraph)."""
        import networkx as nx
        g = self.to_jgf()["graph"]
        out = nx.MultiDiGraph(id=g["id"], label=g["label"], type=g["type"], **g["metadata"])
        for nid, n in g["nodes"].items():
            out.add_node(nid, label=n["label"], **n["metadata"])
        for e in g["edges"]:
            out.add_edge(e["source"], e["target"], key=e["id"], relation=e["relation"], directed=e["directed"],
                         label=e["label"], **e["metadata"])
        return out

    def to_graphml(self, path: str | Path) -> None:
        """GraphML (nested values stored as JSON strings, which GraphML cannot hold natively)."""
        import networkx as nx
        flat = lambda d: {k: (v if isinstance(v, (str, int, float, bool)) else json.dumps(v)) for k, v in d.items() if v is not None}  # noqa: E731
        g = self.to_networkx()
        out = nx.MultiDiGraph(**flat(g.graph))
        for n, a in g.nodes(data=True):
            out.add_node(n, **flat(a))
        for u, v, k, a in g.edges(keys=True, data=True):
            out.add_edge(u, v, key=k, **flat(a))
        nx.write_graphml(out, str(path))
