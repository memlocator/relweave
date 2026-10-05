"""The extraction output of one chunk (what the generator emits) and the text derived from a schema.

Entity, Relation and ExtractionOutput are validated against a schema given as context={"schema": ...} (default
BUSINESS). The JSON Schema for constrained decoding and the schema text in prompts are derived from the Schema;
nothing here is typed by hand.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, ValidationInfo, model_validator

from relweave.schema.base import Schema
from relweave.schema.business import BUSINESS


class Modality(StrEnum):
    ASSERTED = "asserted"
    NEGATED = "negated"
    HEDGED = "hedged"
    REPORTED = "reported"


def canonical_triple(edge_type: str, source, target, schema: Schema | None = None) -> tuple:
    """Order-insensitive key for symmetric edges, so scoring ignores their direction. `schema` (default
    business) says which edge names are symmetric."""
    schema = schema or BUSINESS
    symmetric = edge_type in schema.edge_names() and schema.symmetric(edge_type)
    if symmetric and source is not None and target is not None:
        a, b = sorted((source, target), key=str)
        return (edge_type, a, b)
    return (edge_type, source, target)


# List caps. They flow into the JSON Schema as maxItems, which the decoding grammar
# enforces, so a looping model is forced to close the list instead of running to
# the token limit. Sized well above anything in the gold data (dense Wikipedia chunks
# of about 1,000 characters reach 31 entities and 32 relations).
MAX_MENTIONS_PER_ENTITY = 12
MAX_ENTITIES_PER_CHUNK = 40
MAX_RELATIONS_PER_CHUNK = 40


def _schema(context) -> Schema:
    """The schema a validation runs against: context={"schema": ...}, default BUSINESS."""
    return (context or {}).get("schema") or BUSINESS


class Entity(BaseModel):
    id: str = Field(description="Chunk-local id such as e1")
    type: str
    name: str = Field(description="Canonical name, usually the fullest mention")
    attributes: dict[str, str] = Field(default_factory=dict)
    mentions: list[str] = Field(
        min_length=1, max_length=MAX_MENTIONS_PER_ENTITY,
        description="Distinct surface strings that refer to this entity, copied verbatim from the chunk; "
                    "every occurrence of each string in the chunk is taken as a mention",
    )


class Relation(BaseModel):
    type: str
    source: str
    target: str
    attributes: dict[str, str] = Field(default_factory=dict)
    modality: Modality = Modality.ASSERTED
    evidence: str = Field(description="Verbatim sentence or clause that states the relation")


class ExtractionOutput(BaseModel):
    entities: list[Entity] = Field(default_factory=list, max_length=MAX_ENTITIES_PER_CHUNK)
    relations: list[Relation] = Field(default_factory=list, max_length=MAX_RELATIONS_PER_CHUNK)

    @model_validator(mode="after")
    def _check_references_and_types(self, info: ValidationInfo) -> ExtractionOutput:
        schema = _schema(info.context)
        by_id = {e.id: e for e in self.entities}
        if len(by_id) != len(self.entities):
            raise ValueError("duplicate entity ids")
        for e in self.entities:
            if e.type not in schema.node_names():
                raise ValueError(f"entity type {e.type} not in schema {schema.name}")
        for r in self.relations:
            if r.source not in by_id or r.target not in by_id:
                raise ValueError(f"relation {r.type} references unknown entity id")
            if not schema.legal(r.type, by_id[r.source].type, by_id[r.target].type):
                raise ValueError(
                    f"{r.type} not allowed from {by_id[r.source].type} to {by_id[r.target].type}"
                )
        return self


def parse_lenient(data: dict, repair: bool = False, schema: Schema | None = None,
                  stats: dict | None = None) -> tuple[ExtractionOutput, int, int]:
    """Validate entities strictly; drop relations that dangle or violate endpoint types.

    With repair=True, an illegal relation whose endpoint type pair admits exactly
    one edge type is relabelled to that type instead of dropped (for example
    LOCATED_IN from an Event to a Place becomes HELD_AT).

    Legality and the repair table (Schema.unique_edge_for_pair) come from the schema (default BUSINESS).

    Entities whose type is not in the schema are dropped (their relations then dangle and are dropped);
    their count goes to stats["dropped_entities"] when a stats dict is given.

    Returns the output, the number of relations dropped, and the number repaired.
    """
    schema = schema or BUSINESS
    parsed = [Entity.model_validate(e) for e in data.get("entities", [])]
    entities = [e for e in parsed if e.type in schema.node_names()]
    if stats is not None:
        stats["dropped_entities"] = len(parsed) - len(entities)
    by_id = {e.id: e for e in entities}
    if len(by_id) != len(entities):
        raise ValueError("duplicate entity ids")
    kept, dropped, repaired = [], 0, 0
    for raw in data.get("relations", []):
        try:
            r = Relation.model_validate(raw)
        except Exception:
            dropped += 1
            continue
        if r.type not in schema.edge_names() or r.source not in by_id or r.target not in by_id:
            dropped += 1
            continue
        pair = (by_id[r.source].type, by_id[r.target].type)
        if not schema.legal(r.type, *pair):
            unique = schema.unique_edge_for_pair(*pair) if repair else None
            if unique:
                attrs = {k: v for k, v in r.attributes.items() if k in schema.attributes(unique)}
                r = r.model_copy(update={"type": unique, "attributes": attrs})
                repaired += 1
            else:
                dropped += 1
                continue
        kept.append(r)
    dropped += max(0, len(kept) - MAX_RELATIONS_PER_CHUNK)
    out = ExtractionOutput.model_validate(
        {"entities": entities, "relations": kept[:MAX_RELATIONS_PER_CHUNK]}, context={"schema": schema})
    return out, dropped, repaired


# --------------------------------------------------------------------------
# Derived artefacts
# --------------------------------------------------------------------------


def json_schema(schema: Schema | None = None) -> dict:
    """JSON Schema of ExtractionOutput, for prompts and constrained decoding (default BUSINESS)."""
    schema = schema or BUSINESS
    out = ExtractionOutput.model_json_schema()
    defs = out["$defs"]
    defs["Entity"]["properties"]["type"] = {"$ref": "#/$defs/NodeType"}
    defs["NodeType"] = {"enum": list(schema.node_names()), "title": "NodeType", "type": "string"}
    defs["Relation"]["properties"]["type"] = {"enum": list(schema.edge_names()), "title": "Type", "type": "string"}
    return out


def schema_description(schema: Schema | None = None) -> str:
    """Human-readable schema text inserted into prompts. Derived, not typed by hand."""
    schema = schema or BUSINESS
    lines = ["Node types:"]
    for name in schema.node_names():
        req = schema.required(name)
        extra = f" (required attributes: {', '.join(req)})" if req else ""
        lines.append(f"- {name}{extra}")
    lines.append("")
    lines.append("Edge types (source -> target):")
    for name in schema.edge_names():
        src = "|".join(sorted(schema.sources(name)))
        tgt = "|".join(sorted(schema.targets(name)))
        attrs = schema.attributes(name)
        lines.append(f"- {name}: {src} -> {tgt}. {schema.definition(name)}"
                     + (f"; attributes: {', '.join(attrs)}" if attrs else ""))
    lines.append("")
    lines.append("Modality of every relation: " + ", ".join(m.value for m in Modality) + ".")
    return "\n".join(lines)
