from relweave.graph import Entity, Graph, Relation
from relweave.ontology import coarse_type, coarsen, render
from relweave.schema import BUSINESS


def test_every_in_scope_business_relation_has_a_coarse_type():
    out = {"PARTICIPATED_IN", "HELD_AT", "TRANSFERRED_OBJECT", "OWNS_OBJECT", "HAS_COORDINATE"}
    for name in BUSINESS.edge_names():
        assert (coarse_type(name, "Person") is None) == (name in out), name


def test_founded_depends_on_the_founder():
    assert coarse_type("FOUNDED", "Person") == "AFFILIATED_WITH" and coarse_type("FOUNDED", "Org") == "OWNS"


def test_coarsen_retypes_merges_and_drops_out_of_scope():
    ents = [Entity("e1", "Person", "Ana Varga"), Entity("e2", "Org", "Kessler Logistics"), Entity("e3", "Event", "Fair")]
    rels = [Relation("EXECUTIVE_OF", "e1", "e2"), Relation("BOARD_MEMBER_OF", "e1", "e2"),
            Relation("PARTICIPATED_IN", "e1", "e3")]
    g = coarsen(Graph(entities=ents, relations=rels, text=""))
    assert [(r.type, r.source, r.target, r.attributes["fine_type"]) for r in g.relations] == \
        [("AFFILIATED_WITH", "e1", "e2", "EXECUTIVE_OF")]
    assert [e.id for e in g.entities] == ["e1", "e2"]


def test_render_names_every_type():
    t = render()
    assert all(n in t for n in ("AFFILIATED_WITH", "OWNS", "PART_OF", "LOCATED_IN", "FAMILY_OF", "ASSOCIATED_WITH"))
