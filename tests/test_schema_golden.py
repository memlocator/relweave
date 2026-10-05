"""Pin the schema-derived outputs (business schema) to files captured before the schema became Pydantic classes."""
import json

from golden_snapshot import GOLDEN, snapshot
from relweave.schema import schema_description

SCHEMA_KEYS = ("edge_names", "symmetric_edges", "node_required_attributes", "allowed_triples", "json_schema")


def test_schema_outputs_match_golden():
    golden = json.loads((GOLDEN / "schema_golden.json").read_text())
    assert snapshot(with_formats=False) == {k: golden[k] for k in SCHEMA_KEYS}


def test_schema_description_matches_golden():
    assert schema_description() + "\n" == (GOLDEN / "schema_description.txt").read_text()


def test_format_prompts_match_golden():
    golden = json.loads((GOLDEN / "schema_golden.json").read_text())
    snap = snapshot()
    assert snap["system_prompts"] == golden["system_prompts"]
    assert snap["compact_messages"] == golden["compact_messages"]
