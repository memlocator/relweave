import json

import pytest
from pydantic import ValidationError

from relweave.schema import BUSINESS, ExtractionOutput, canonical_triple, json_schema, schema_description


def test_every_edge_has_at_least_one_legal_triple():
    for e in BUSINESS.edge_names():
        assert any(BUSINESS.legal(e, s, t) for s in BUSINESS.node_names() for t in BUSINESS.node_names()), e


def test_legal_matches_definitions():
    assert BUSINESS.legal("EXECUTIVE_OF", "Person", "Org")
    assert not BUSINESS.legal("EXECUTIVE_OF", "Org", "Person")
    assert BUSINESS.legal("LOCATED_IN", "Place", "Place")
    assert not BUSINESS.legal("HAS_COORDINATE", "Person", "Coordinate")


def _output(edge: str, src_type: str, tgt_type: str) -> dict:
    return {
        "entities": [
            {"id": "e1", "type": src_type, "name": "A", "mentions": ["A"]},
            {"id": "e2", "type": tgt_type, "name": "B", "mentions": ["B"]},
        ],
        "relations": [{"type": edge, "source": "e1", "target": "e2", "evidence": "A x B"}],
    }


def test_output_rejects_illegal_endpoint_types():
    with pytest.raises(ValidationError):
        ExtractionOutput.model_validate(_output("EXECUTIVE_OF", "Org", "Person"))


def test_output_rejects_unknown_edge_type():
    with pytest.raises(ValidationError):
        ExtractionOutput.model_validate(_output("OWNS", "Person", "Object"))


def test_output_rejects_dangling_reference():
    data = _output("EXECUTIVE_OF", "Person", "Org")
    data["relations"][0]["target"] = "e9"
    with pytest.raises(ValidationError):
        ExtractionOutput.model_validate(data)


def test_output_accepts_legal_relation():
    out = ExtractionOutput.model_validate(_output("EXECUTIVE_OF", "Person", "Org"))
    assert out.relations[0].modality == "asserted"


def test_schema_description_mentions_every_edge_and_node():
    text = schema_description()
    for e in BUSINESS.edge_names():
        assert e in text
    for n in BUSINESS.node_names():
        assert n in text


def test_json_schema_enumerates_edge_types():
    js = json.dumps(json_schema())
    for e in BUSINESS.edge_names():
        assert e in js


def test_symmetric_edges_have_order_insensitive_keys():
    assert canonical_triple("ASSOCIATE_OF", "e2", "e1") == canonical_triple("ASSOCIATE_OF", "e1", "e2")
    assert canonical_triple("EXECUTIVE_OF", "e2", "e1") != canonical_triple("EXECUTIVE_OF", "e1", "e2")
    assert canonical_triple("ASSOCIATE_OF", None, "e1") == ("ASSOCIATE_OF", None, "e1")
