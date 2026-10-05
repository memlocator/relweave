"""Schema authoring: entity and relation types as Pydantic classes, one definition per type.

The docstring of a class is the definition the model reads; the relation name is derived from the
class name (OperatesIn -> OPERATES_IN); endpoint types come from Relation[Source, Target] (a union
allows several); optional fields are the relation's attributes, required fields of an Entity class its
required attributes. Schema derives everything else (prompt text, legality, grammar inputs, JSON).
"""

from __future__ import annotations

import inspect
import re
import types
import typing
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

S = TypeVar("S")
T = TypeVar("T")


class Entity(BaseModel):
    """Base for entity types; subclass docstring = definition."""


class Relation(BaseModel, Generic[S, T]):
    """Base for relation types; subclass as Relation[Source, Target]."""
    aliases: ClassVar[tuple[str, ...]] = ()
    paraphrases: ClassVar[tuple[str, ...]] = ()
    symmetric: ClassVar[bool] = False


def edge_name(cls: type[Relation]) -> str:
    split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", cls.__name__)
    return re.sub("_+", "_", split).upper()


def _types(arg) -> frozenset[str]:
    if isinstance(arg, types.UnionType) or typing.get_origin(arg) is typing.Union:
        return frozenset(a.__name__ for a in typing.get_args(arg))
    return frozenset({arg.__name__})


def _endpoints(cls: type[Relation]) -> tuple[frozenset[str], frozenset[str]]:
    for base in cls.__mro__[1:]:
        args = getattr(base, "__pydantic_generic_metadata__", {}).get("args", ())
        if len(args) == 2:
            return _types(args[0]), _types(args[1])
    raise ValueError(f"{cls.__name__} must subclass Relation[Source, Target]")


def definition_of(cls) -> str:
    return " ".join(inspect.cleandoc(cls.__doc__ or "").split())


class Schema:
    """Entity and relation classes plus derived lookups. A variant (see variant()) shares the original
    classes and only overrides the shown relation name and definition (`names`, `defs`, keyed by class) and
    narrows endpoint types to the kept entity types (`restrict`)."""

    def __init__(self, name: str, entities: list[type[Entity]], relations: list[type[Relation]],
                 names: dict[type[Relation], str] | None = None, defs: dict[type[Relation], str] | None = None,
                 restrict: frozenset[str] | None = None, root: Schema | None = None):
        self.name = name
        self._root = root  # the schema a variant was derived from; names in definitions refer to it
        self.entity_classes = list(entities)
        self.relation_classes = list(relations)
        self._names = dict(names or {})
        self._defs = dict(defs or {})
        self._restrict = restrict
        self._nodes: dict[str, type[Entity]] = {}
        for c in entities:
            node_name = c.__name__
            if node_name in self._nodes:
                raise ValueError(f"duplicate entity name {node_name} ({c.__name__}, {self._nodes[node_name].__name__})")
            self._nodes[node_name] = c
        self._edges: dict[str, type[Relation]] = {}
        for c in relations:
            n = self._shown(c)
            if n in self._edges:
                raise ValueError(f"duplicate relation name {n} ({c.__name__}, {self._edges[n].__name__})")
            if n in self._nodes:
                raise ValueError(f"entity name {n} conflicts with relation name {n}")
            src, tgt = _endpoints(c)
            if restrict is not None:
                src, tgt = src & restrict, tgt & restrict
                if not (src and tgt):
                    raise ValueError(f"{n}: no endpoint types left in schema {name}")
            elif missing := (src | tgt) - set(self._nodes):
                raise ValueError(f"{n}: endpoint types {sorted(missing)} not in schema {name}")
            self._edges[n] = c
        self._check_aliases()

    def _shown(self, cls: type[Relation]) -> str:
        return self._names.get(cls, edge_name(cls))

    def _check_aliases(self) -> None:
        """Aliases are names a type may be shown under: unique, and disjoint from every edge and entity name."""
        taken = set(self._nodes) | {edge_name(c) for c in self.relation_classes}
        seen: dict[str, str] = {}
        for c in self.relation_classes:
            for a in c.aliases:
                if a in taken:
                    raise ValueError(f"alias {a} of {c.__name__} equals an existing relation or entity name")
                if a in seen:
                    raise ValueError(f"alias {a} is used by both {seen[a]} and {c.__name__}")
                seen[a] = c.__name__

    def edge_names(self) -> tuple[str, ...]:
        return tuple(self._edges)

    def node_names(self) -> tuple[str, ...]:
        return tuple(self._nodes)

    def relation_class(self, edge: str) -> type[Relation]:
        return self._edges[edge]

    def _endpoint_types(self, edge: str) -> tuple[frozenset[str], frozenset[str]]:
        src, tgt = _endpoints(self._edges[edge])
        if self._restrict is None:
            return src, tgt
        return src & self._restrict, tgt & self._restrict

    def sources(self, edge: str) -> frozenset[str]:
        return self._endpoint_types(edge)[0]

    def targets(self, edge: str) -> frozenset[str]:
        return self._endpoint_types(edge)[1]

    def legal(self, edge: str, source_type: str, target_type: str) -> bool:
        return edge in self._edges and source_type in self.sources(edge) and target_type in self.targets(edge)

    def unique_edge_for_pair(self, source_type: str, target_type: str) -> str | None:
        """The only edge type that admits this endpoint type pair, or None (used to repair a wrong label)."""
        names = [e for e in self._edges if self.legal(e, source_type, target_type)]
        return names[0] if len(names) == 1 else None

    def variant(self, rng, drop_relation: float = 0.3, drop_entity: float = 0.15, rename: float = 0.3,
                keep: frozenset[str] = frozenset()) -> tuple[Schema, dict[str, str]]:
        """A random sub-schema: types dropped, relations renamed to aliases, order shuffled. Returns it and
        the mapping from this schema's relation names to the variant's (dropped relations are absent).
        `keep` lists entity or relation type names (as shown by this schema) that are never dropped; a kept
        relation whose endpoint types are all dropped is still skipped. A relation type left without
        sources or targets is dropped too. The variant is named "<name>/variant", so business-only
        behaviour keyed on the schema name does not apply to it."""
        nodes = [c for c in self.entity_classes if c.__name__ in keep or rng.random() >= drop_entity] \
            or self.entity_classes[:1]
        kept_nodes = frozenset(c.__name__ for c in nodes)
        taken = set(self.edge_names()) | set(self.node_names())
        rels, names, defs, mapping = [], {}, {}, {}
        for c in self.relation_classes:
            orig = self._shown(c)
            src, tgt = self.sources(orig) & kept_nodes, self.targets(orig) & kept_nodes
            if not (src and tgt):
                continue
            if orig not in keep and rng.random() < drop_relation:
                continue
            shown = orig
            if c.aliases and rng.random() < rename:
                options = [a for a in c.aliases if a not in taken]
                if options:
                    shown = rng.choice(options)
                    taken.add(shown)
                    if c.paraphrases:
                        defs[c] = rng.choice(c.paraphrases)
            if shown == orig and c in self._defs:
                defs[c] = self._defs[c]
            names[c], mapping[orig] = shown, shown
            rels.append(c)
        rng.shuffle(rels)
        rng.shuffle(nodes)
        root = self._root or self
        return Schema(f"{root.name}/variant", nodes, rels, names=names, defs=defs, restrict=kept_nodes, root=root), mapping

    def attributes(self, edge: str) -> tuple[str, ...]:
        return tuple(self._edges[edge].model_fields)

    def symmetric(self, edge: str) -> bool:
        return self._edges[edge].symmetric

    def definition(self, name: str) -> str:
        if name in self._edges:
            c = self._edges[name]
            return self._follow_variant(self._defs.get(c) or definition_of(c))
        return definition_of(self._nodes[name])

    def _follow_variant(self, text: str) -> str:
        """Definition text written against the root schema, rewritten for this variant: relation names
        become their shown names; a clause (split on ';', never the first) naming a dropped relation or
        entity type is removed."""
        if self._root is None:
            return text
        shown = {edge_name(c): self._shown(c) for c in self.relation_classes}
        dropped = ({edge_name(c) for c in self._root.relation_classes} - set(shown)) \
            | (set(self._root.node_names()) - set(self._nodes))
        pattern = re.compile(r"\b(" + "|".join(sorted(set(shown) | dropped | set(self._nodes), key=len, reverse=True))
                             + r")(?:s\b|\b)")
        out = []
        for i, clause in enumerate(c.strip() for c in text.split(";")):
            names = {m.group(1) for m in pattern.finditer(clause)}
            if i and names & dropped:
                continue
            out.append(pattern.sub(lambda m: m.group(0).replace(m.group(1), shown.get(m.group(1), m.group(1)), 1), clause))
        return "; ".join(out)

    def required(self, node: str) -> tuple[str, ...]:
        return tuple(k for k, f in self._nodes[node].model_fields.items() if f.is_required())

    def render(self) -> str:
        """Compact prompt text: entity types, then relation types with endpoints and definitions."""
        lines = ["Entity types:"]
        for n in self._nodes:
            req = self.required(n)
            lines.append(f"- {n}: {self.definition(n)}" + (f" (required attributes: {', '.join(req)})" if req else ""))
        lines.append("Relation types (source -> target):")
        for e in self._edges:
            attrs = f"; attributes: {', '.join(self.attributes(e))}" if self.attributes(e) else ""
            lines.append(f"- {e}: {'|'.join(sorted(self.sources(e)))} -> {'|'.join(sorted(self.targets(e)))}. "
                         f"{self.definition(e)}{attrs}")
        return "\n".join(lines)

    def json_schema(self) -> dict:
        return {"name": self.name,
                "entities": {n: c.model_json_schema() | {"description": self.definition(n)} for n, c in self._nodes.items()},
                "relations": {e: c.model_json_schema() | {"description": self.definition(e),
                              "sources": sorted(self.sources(e)), "targets": sorted(self.targets(e)),
                              "symmetric": c.symmetric} for e, c in self._edges.items()}}
