import json
import random
from pathlib import Path

import pytest

from relweave.schema.base import Entity, Relation, Schema
from relweave.schema import BUSINESS
from relweave.schema.output import ExtractionOutput, parse_lenient
from relweave.training.variants import map_gold

GOLD = ExtractionOutput.model_validate({
    "entities": [{"id": "e1", "type": "Org", "name": "Acme", "mentions": ["Acme"]},
                 {"id": "e2", "type": "Place", "name": "Oslo", "mentions": ["Oslo"]},
                 {"id": "e3", "type": "Event", "name": "the sale", "attributes": {"kind": "sale"}, "mentions": ["the sale"]}],
    "relations": [{"type": "OPERATES_IN", "source": "e1", "target": "e2", "evidence": "x"},
                  {"type": "HELD_AT", "source": "e3", "target": "e2", "evidence": "y"}]})


def test_variant_is_deterministic_and_has_unique_names():
    for seed in range(50):
        v1, m1 = BUSINESS.variant(random.Random(seed))
        v2, m2 = BUSINESS.variant(random.Random(seed))
        assert v1.edge_names() == v2.edge_names() and m1 == m2
        assert len(set(v1.edge_names())) == len(v1.edge_names())
        assert not set(v1.edge_names()) & set(v1.node_names())


def test_dropped_entity_type_removes_its_entities_and_relations():
    for seed in range(200):
        v, m = BUSINESS.variant(random.Random(seed), drop_entity=0.5)
        if "Event" not in v.node_names():
            out = map_gold(GOLD, BUSINESS, v, m)
            assert all(e.type != "Event" for e in out.entities)
            assert all(r.source != "e3" and r.target != "e3" for r in out.relations)
            return
    raise AssertionError("no variant dropped Event")


def test_renamed_type_is_renamed_in_gold_and_prompt():
    for seed in range(200):
        v, m = BUSINESS.variant(random.Random(seed), rename=1.0, drop_relation=0.0, drop_entity=0.0)
        new = m["OPERATES_IN"]
        if new != "OPERATES_IN":
            out = map_gold(GOLD, BUSINESS, v, m)
            assert {r.type for r in out.relations} == {new, m["HELD_AT"]}
            assert f"- {new}: Org -> Place." in v.render()
            ExtractionOutput.model_validate(out.model_dump(), context={"schema": v})
            return
    raise AssertionError("no rename happened")


def test_no_change_variant_is_the_original_up_to_order():
    v, m = BUSINESS.variant(random.Random(0), drop_relation=0.0, drop_entity=0.0, rename=0.0)
    assert set(v.edge_names()) == set(BUSINESS.edge_names()) and set(v.node_names()) == set(BUSINESS.node_names())
    assert all(k == x for k, x in m.items())
    out = map_gold(GOLD, BUSINESS, v, m)
    assert out.model_dump() == GOLD.model_dump()


def test_keep_is_never_dropped():
    for seed in range(100):
        v, m = BUSINESS.variant(random.Random(seed), drop_relation=0.9, drop_entity=0.9,
                                keep=frozenset({"Event", "HELD_AT", "Place"}))
        assert {"Event", "Place"} <= set(v.node_names()) and "HELD_AT" in m


def test_variant_of_variant_keeps_working():
    v1, m1 = BUSINESS.variant(random.Random(1), rename=1.0, drop_relation=0.0, drop_entity=0.0)
    v2, m2 = v1.variant(random.Random(2), rename=1.0, drop_relation=0.0, drop_entity=0.0)
    assert set(m2) == set(v1.edge_names()) and set(m2.values()) == set(v2.edge_names())
    assert len(set(v2.edge_names())) == len(v2.edge_names())


def test_renamed_definition_uses_a_paraphrase():
    for seed in range(200):
        v, m = BUSINESS.variant(random.Random(seed), rename=1.0, drop_relation=0.0, drop_entity=0.0)
        if m["HELD_AT"] != "HELD_AT":
            assert v.definition(m["HELD_AT"]) in BUSINESS.relation_class("HELD_AT").paraphrases
            return
    raise AssertionError("no rename happened")


def test_variant_endpoints_are_restricted_to_kept_types():
    for seed in range(100):
        v, _ = BUSINESS.variant(random.Random(seed), drop_entity=0.5, drop_relation=0.0)
        for e in v.edge_names():
            assert v.sources(e) <= set(v.node_names()) and v.targets(e) <= set(v.node_names())
            assert v.sources(e) and v.targets(e)


def test_variant_parse_lenient_rejects_original_names_of_renamed_types():
    for seed in range(200):
        v, m = BUSINESS.variant(random.Random(seed), rename=1.0, drop_relation=0.0, drop_entity=0.0)
        if m["OPERATES_IN"] != "OPERATES_IN":
            data = GOLD.model_dump()
            out, dropped, _ = parse_lenient(data, schema=v)
            assert dropped >= 1 and all(r.type != "OPERATES_IN" for r in out.relations)
            return
    raise AssertionError("no rename happened")


def test_repair_table_follows_the_schema_not_its_name():
    # Event -> Place admits only HELD_AT in business; in a variant that renames it, repair uses the shown name
    for seed in range(200):
        v, m = BUSINESS.variant(random.Random(seed), rename=1.0, drop_relation=0.0, drop_entity=0.0)
        if m["HELD_AT"] != "HELD_AT":
            data = {"entities": [e.model_dump() for e in GOLD.entities],
                    "relations": [{"type": m["OPERATES_IN"], "source": "e3", "target": "e2", "evidence": "y"}]}
            out, dropped, repaired = parse_lenient(data, repair=True, schema=v)
            assert repaired == 1 and out.relations[0].type == m["HELD_AT"]
            return
    raise AssertionError("no rename happened")


class A(Entity):
    """a"""


class B(Entity):
    """b"""


def _rel(name, aliases):
    return type(name, (Relation[A, B],), {"__doc__": "d", "aliases": aliases, "__annotations__": {}})


def test_alias_clashes_are_rejected_at_construction():
    with pytest.raises(ValueError, match="alias"):
        Schema("x", [A, B], [_rel("Foo", ("BAR",)), _rel("Bar", ())])        # alias equals another edge name
    with pytest.raises(ValueError, match="alias"):
        Schema("x", [A, B], [_rel("Foo", ("SAME",)), _rel("Baz", ("SAME",))])  # alias shared
    with pytest.raises(ValueError, match="alias"):
        Schema("x", [A, B], [_rel("Foo", ("A",))])                             # alias equals an entity name


def _variant_where(pred, **kw):
    for seed in range(500):
        v, m = BUSINESS.variant(random.Random(seed), **kw)
        if pred(v, m):
            return v, m
    raise AssertionError("no matching variant")


def test_definitions_follow_renamed_relation_names():
    v, m = _variant_where(lambda v, m: m.get("HEADQUARTERED_IN", "HEADQUARTERED_IN") != "HEADQUARTERED_IN"
                          and "LOCATED_IN" in m, rename=0.6, drop_relation=0.0, drop_entity=0.0)
    d = v.definition(m["LOCATED_IN"])
    assert m["HEADQUARTERED_IN"] in d and "HEADQUARTERED_IN" not in d


def test_definition_clause_naming_a_dropped_relation_is_removed():
    v, m = _variant_where(lambda v, m: "BOARD_MEMBER_OF" in m and "EXECUTIVE_OF" not in m, drop_entity=0.0)
    d = v.definition(m["BOARD_MEMBER_OF"])
    assert "EXECUTIVE_OF" not in d and "chair" not in d and "board" in d


def test_definition_clause_naming_a_dropped_entity_is_removed():
    v, m = _variant_where(lambda v, m: "SUBSIDIARY_OF" in m and "Object" not in v.node_names(),
                          drop_relation=0.0, drop_entity=0.3, rename=0.0)
    assert "Object" not in v.definition(m["SUBSIDIARY_OF"])
    assert v.definition(m["SUBSIDIARY_OF"]).startswith("Source Org is a subsidiary")


def test_business_definitions_are_unchanged():
    from relweave.schema.base import definition_of
    for e in BUSINESS.edge_names():
        assert BUSINESS.definition(e) == definition_of(BUSINESS.relation_class(e))


def test_variant_has_its_own_name_and_no_worked_example():
    from relweave.formats import FORMATS, messages
    v, _ = BUSINESS.variant(random.Random(3))
    assert v.name != BUSINESS.name
    assert v.variant(random.Random(4))[0].name == v.name
    fmt = FORMATS["lines"].with_schema(v)
    assert len(messages(fmt, "Acme is in Oslo.")) == 2          # system + chunk, no example turns
    assert len(messages(FORMATS["lines"], "Acme is in Oslo.")) == 4


def test_first_paraphrase_is_the_docstring_where_it_repeats_it():
    from relweave.schema.base import definition_of
    for e in BUSINESS.edge_names():
        c = BUSINESS.relation_class(e)
        if c.paraphrases and definition_of(c) in c.paraphrases:
            assert c.paraphrases[0] == definition_of(c)


def _chunks(tmp_path: Path, **extra) -> Path:
    rec = {"chunk_id": "d#1", "doc_id": "d", "text": "Acme operates in Oslo. The sale took place in Oslo.",
           "gold": GOLD.model_dump(mode="json"), **extra}
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps(rec) + "\n")
    return p


def test_training_examples_carry_their_variant_and_hold_out_types(tmp_path: Path):
    from relweave.training.generator import build_examples
    p = _chunks(tmp_path)
    held = {"relations": ["OPERATES_IN"], "entities": [], "schemas": []}
    ex = build_examples(p, "sentences", False, conditioned=True, variants=True, held_out=held, seed=1)
    system, answer = ex[0]["messages"][0]["content"], ex[0]["messages"][-1]["content"]
    assert "OPERATES_IN" not in system and "OPERATES_IN" not in answer


def test_reduced_drops_exactly_the_given_types_and_renames_nothing():
    from relweave.training.variants import reduced
    v, m = reduced(BUSINESS, drop_relations=["OPERATES_IN"], drop_entities=["Event"])
    assert "OPERATES_IN" not in v.edge_names() and "Event" not in v.node_names()
    assert set(v.node_names()) == set(BUSINESS.node_names()) - {"Event"}
    assert all(k == w for k, w in m.items())
    assert set(m) == set(v.edge_names())


def test_held_out_schema_chunks_are_skipped_and_summary_names_root(tmp_path: Path):
    from relweave.training.generator import build_examples
    p = _chunks(tmp_path)
    assert build_examples(p, "sentences", held_out={"schemas": ["business"]}) == []
    ex = build_examples(p, "sentences", variants=True, seed=3)
    assert ex[0]["schema"] == "business"


def test_train_summary_keeps_counts_and_names_root_schemas():
    from relweave.training.generator import train_summary
    exs = [{"messages": [], "chunk_id": "a", "schema": "business"}, {"messages": [], "chunk_id": "b"}]
    s = train_summary(exs, [10, 30], task="extract", pair_chunks=None, pair_distance=False, init_adapter=None,
                      conditioned=True, schema_variants=True, held_out={"schemas": ["sports"]}, fmt="sentences",
                      compact=False, model_id="m", last_response_only=True, fast=False, epochs=1, lr=1e-4, r=8,
                      lora_dropout=0.0, relation_weight=None, train_loss=0.5, runtime_s=1.0, peak_vram_gb=1.0)
    assert s["examples"] == 2 and s["max_tokens"] == 30 and s["mean_tokens"] == 20
    assert s["conditioned"] and s["schema_variants"] and s["held_out"] == {"schemas": ["sports"]}
    assert s["schemas"] == ["business"]
