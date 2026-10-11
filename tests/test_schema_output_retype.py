from relweave.schema.output import parse_lenient


def test_relation_retypes_a_mistyped_org():
    data = {"entities": [{"id": "e1", "type": "Person", "name": "Kappahl", "mentions": ["Kappahl"]},
                         {"id": "e2", "type": "Person", "name": "Sandra Roos", "mentions": ["Sandra Roos"]}],
            "relations": [{"type": "EMPLOYED_BY", "source": "e2", "target": "e1", "evidence": "x"}]}
    out, dropped, _ = parse_lenient(data)
    assert dropped == 0 and {e.name: e.type for e in out.entities}["Kappahl"] == "Org"


def test_no_retype_when_it_would_break_a_legal_relation():
    data = {"entities": [{"id": "e1", "type": "Person", "name": "Ana", "mentions": ["Ana"]},
                         {"id": "e2", "type": "Person", "name": "Ben", "mentions": ["Ben"]},
                         {"id": "e3", "type": "Person", "name": "Cy", "mentions": ["Cy"]}],
            "relations": [{"type": "EMPLOYED_BY", "source": "e2", "target": "e1", "evidence": "x"},
                          {"type": "FAMILY_OF", "source": "e1", "target": "e3", "attributes": {"kind": "spouse"}, "evidence": "x"}]}
    out, dropped, _ = parse_lenient(data)
    assert {e.name: e.type for e in out.entities}["Ana"] == "Person" and dropped == 1
