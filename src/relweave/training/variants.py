"""Gold labels under a schema variant (Schema.variant): the training-time counterpart of random schemas."""
from __future__ import annotations

import random

from relweave.schema import ExtractionOutput, Schema
from relweave.training.data import schema_of


def map_gold(gold: ExtractionOutput, schema: Schema, variant: Schema, mapping: dict[str, str]) -> ExtractionOutput:
    """Gold under `schema` -> gold under `variant`: entities of dropped types and relations of dropped types
    or touching dropped entities removed; kept relations renamed."""
    keep_types = set(variant.node_names())
    ents = [e for e in gold.entities if e.type in keep_types]
    type_of = {e.id: e.type for e in ents}
    rels = [r.model_copy(update={"type": mapping[r.type]}) for r in gold.relations
            if r.type in mapping and r.source in type_of and r.target in type_of
            and variant.legal(mapping[r.type], type_of[r.source], type_of[r.target])]
    return gold.model_copy(update={"entities": ents, "relations": rels})


def reduced(schema: Schema, drop_relations=(), drop_entities=()) -> tuple[Schema, dict[str, str]]:
    """A variant with exactly these relation and entity types removed (names as `schema` shows them) and
    nothing renamed; types left without endpoints go too (Schema.variant). Order is shuffled with a fixed
    seed, so the result is the same every call. Returns it and the mapping, as Schema.variant does."""
    keep = (set(schema.node_names()) | set(schema.edge_names())) - set(drop_relations) - set(drop_entities)
    return schema.variant(random.Random(0), drop_relation=1.0, drop_entity=1.0, rename=0.0, keep=frozenset(keep))


def training_view(chunk: dict, rng: random.Random, variants: bool = False,
                  held_out: dict | None = None) -> tuple[ExtractionOutput, Schema, str] | None:
    """(gold, schema, root schema name) as a training example of this chunk record sees them: held-out
    types removed from the chunk's schema and gold, then (variants) a random variant of that. None when
    the chunk's schema is held out."""
    held = held_out or {}
    name = chunk.get("schema", "business")
    if name in held.get("schemas", []):
        return None
    schema = schema_of(chunk)
    gold = ExtractionOutput.model_validate(chunk["gold"], context={"schema": schema})
    if held.get("relations") or held.get("entities"):
        base, base_map = reduced(schema, held.get("relations", []), held.get("entities", []))
        gold, schema = map_gold(gold, schema, base, base_map), base
    if variants:
        v, m = schema.variant(rng)
        gold, schema = map_gold(gold, schema, v, m), v
    return gold, schema, name
