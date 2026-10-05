import pytest
from relweave.schema import Entity, Relation, Schema
from relweave.schema.base import edge_name
from toy import S, FamilyOf, LocatedIn, OperatesIn, Org, Person, Place  # noqa: F401


def test_edge_name_is_derived_from_the_class_name():
    assert edge_name(OperatesIn) == "OPERATES_IN"
    assert S.edge_names() == ("OPERATES_IN", "FAMILY_OF", "LOCATED_IN")
    assert S.node_names() == ("Person", "Org", "Place")


def test_endpoints_attributes_and_symmetry_come_from_the_class():
    assert S.sources("LOCATED_IN") == frozenset({"Person", "Org", "Place"})
    assert S.targets("OPERATES_IN") == frozenset({"Place"})
    assert S.attributes("OPERATES_IN") == ("start",)
    assert S.symmetric("FAMILY_OF") and not S.symmetric("OPERATES_IN")
    assert S.legal("OPERATES_IN", "Org", "Place") and not S.legal("OPERATES_IN", "Person", "Place")


def test_render_lists_every_type_with_its_definition():
    text = S.render()
    assert "- Org: A company or other organised body." in text
    assert "- OPERATES_IN: Org -> Place. The Org has operations in the Place.; attributes: start" in text
    assert "- LOCATED_IN: Org|Person|Place -> Place." in text


def test_duplicate_names_are_rejected():
    class Operates_In(Relation[Org, Place]):
        """Duplicate after derivation."""
    with pytest.raises(ValueError, match="duplicate"):
        Schema(name="bad", entities=[Org, Place], relations=[OperatesIn, Operates_In])


def test_endpoint_types_must_be_in_the_schema():
    with pytest.raises(ValueError, match="not in schema"):
        Schema(name="bad", entities=[Org], relations=[OperatesIn])


def test_json_schema_has_every_type():
    js = S.json_schema()
    assert set(js["relations"]) == set(S.edge_names()) and set(js["entities"]) == set(S.node_names())


def test_duplicate_entity_names_are_rejected():
    # Create two classes with the same __name__
    AnotherPerson = type('Person', (Entity,), {'__doc__': 'Another person'})
    with pytest.raises(ValueError, match="duplicate entity name"):
        Schema(name="bad", entities=[Person, AnotherPerson], relations=[])


def test_entity_name_conflicting_with_relation_name_is_rejected():
    class OPERATES_IN(Entity):
        """Conflicts with OperatesIn relation."""
    with pytest.raises(ValueError, match="entity name.*relation name|relation name.*entity name"):
        Schema(name="bad", entities=[OPERATES_IN, Place], relations=[OperatesIn])


def test_extraction_output_validates_against_the_given_schema():
    from relweave.schema.output import ExtractionOutput
    data = {"entities": [{"id": "e1", "type": "Org", "name": "Acme", "mentions": ["Acme"]},
                         {"id": "e2", "type": "Place", "name": "Oslo", "mentions": ["Oslo"]}],
            "relations": [{"type": "OPERATES_IN", "source": "e1", "target": "e2", "evidence": "x"}]}
    assert ExtractionOutput.model_validate(data, context={"schema": S}).relations[0].type == "OPERATES_IN"
    bad = dict(data, relations=[{"type": "HEADQUARTERED_IN", "source": "e1", "target": "e2", "evidence": "x"}])
    with pytest.raises(ValueError):
        ExtractionOutput.model_validate(bad, context={"schema": S})  # business type, not in toy schema
    ExtractionOutput.model_validate(bad)  # no context: business, where it is legal


def test_parse_lenient_drops_unknown_relation_type_even_with_repair():
    from relweave.schema.output import parse_lenient
    data = {"entities": [{"id": "e1", "type": "Event", "name": "Fair", "mentions": ["Fair"], "attributes": {"kind": "fair"}},
                         {"id": "e2", "type": "Place", "name": "Oslo", "mentions": ["Oslo"]}],
            "relations": [{"type": "HOSTED_AT", "source": "e1", "target": "e2", "evidence": "x"}]}
    out, dropped, repaired = parse_lenient(data, repair=True)
    assert (len(out.relations), dropped, repaired) == (0, 1, 0)


def test_edge_name_handles_acronyms_and_repeated_underscores():
    class OperatesInUSA(Relation[Org, Place]):
        """x"""

    class HQIn(Relation[Org, Place]):
        """x"""

    class Has_Part(Relation[Org, Place]):
        """x"""

    assert edge_name(OperatesInUSA) == "OPERATES_IN_USA"
    assert edge_name(HQIn) == "HQ_IN"
    assert edge_name(Has_Part) == "HAS_PART"


def test_register_rejects_a_different_schema_under_an_existing_name():
    from relweave.schema import SCHEMAS, register
    first = Schema(name="reg_test", entities=[Org, Place], relations=[OperatesIn])
    try:
        assert register(first) is first
        register(first)
        with pytest.raises(ValueError):
            register(Schema(name="reg_test", entities=[Org, Place], relations=[OperatesIn]))
    finally:
        SCHEMAS.pop("reg_test", None)


def test_relation_without_endpoints_raises_clear_error():
    class Bare(Relation):
        """x"""

    with pytest.raises(ValueError, match="Relation\\[Source, Target\\]"):
        Schema(name="bad", entities=[Org, Place], relations=[Bare])


def test_union_target_gives_both_types():
    class Reaches(Relation[Org, Place | Org]):
        """x"""

    sch = Schema(name="u", entities=[Org, Place], relations=[Reaches])
    assert sch.targets("REACHES") == {"Place", "Org"}


def test_domain_schemas_register_and_render():
    from relweave.schema import SCHEMAS
    for name in ("business", "politics", "legal", "sports"):
        s = SCHEMAS[name]
        assert 8 <= len(s.edge_names()) <= 21 and s.render()
        assert all(s.definition(e) for e in s.edge_names())


def test_relation_names_in_definitions_are_edges_of_the_schema():
    import re
    from relweave.schema import SCHEMAS
    from relweave.schema.base import edge_name
    for s in SCHEMAS.values():
        edges = set(s.edge_names())
        for c in s.relation_classes:
            for text in (s.definition(edge_name(c)), *c.paraphrases):
                stray = set(re.findall(r"\b[A-Z][A-Z_]{3,}\b", text)) - edges
                assert not stray, f"{s.name}.{c.__name__} names {sorted(stray)}"


def test_aliases_unique_per_schema():
    from relweave.schema import SCHEMAS
    for s in SCHEMAS.values():
        aliases = [a for c in s.relation_classes for a in c.aliases]
        assert len(aliases) == len(set(aliases)), s.name
        assert not set(aliases) & (set(s.edge_names()) | set(s.node_names())), s.name


def test_every_relation_class_has_three_aliases_and_two_paraphrases():
    import re
    from relweave.schema import SCHEMAS
    from relweave.schema.base import definition_of
    for s in SCHEMAS.values():
        for c in s.relation_classes:
            where = f"{s.name}.{c.__name__}"
            assert len(c.aliases) == 3, where
            assert all(re.fullmatch(r"[A-Z]+(_[A-Z]+)*", a) for a in c.aliases), where
            assert len(c.paraphrases) == 2 and len(set(c.paraphrases)) == 2, where
            assert c.paraphrases[0] != definition_of(c), f"{where}: first paraphrase repeats the definition"
            assert all(p.strip() and "\n" not in p for p in c.paraphrases), where
