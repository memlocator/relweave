from relweave.schema import BUSINESS


def test_markdown_lists_every_type_with_its_definition():
    md = BUSINESS.markdown()
    for name in BUSINESS.node_names() + BUSINESS.edge_names():
        assert f"| {name} |" in md and BUSINESS.definition(name) in md
    assert "| FAMILY_OF | Person -> Person (symmetric) |" in md
