"""The schema-derived outputs pinned by tests/golden/ (business schema). Run this file to regenerate the golden
files, only when a schema or prompt change is intended: uv run python tests/golden_snapshot.py"""
import json
from itertools import product
from pathlib import Path

from relweave.schema import BUSINESS, json_schema, schema_description

GOLDEN = Path(__file__).resolve().parent / "golden"
FORMAT_NAMES = ("json", "lines", "sentences")


def snapshot(with_formats: bool = True) -> dict:
    nodes = BUSINESS.node_names()
    out = {
        "edge_names": list(BUSINESS.edge_names()),
        "symmetric_edges": sorted(e for e in BUSINESS.edge_names() if BUSINESS.symmetric(e)),
        "node_required_attributes": {n: list(BUSINESS.required(n)) for n in nodes},
        "allowed_triples": sorted([e, s, t] for e, s, t in product(BUSINESS.edge_names(), nodes, nodes)
                                  if BUSINESS.legal(e, s, t)),
        "json_schema": json_schema(),
    }
    if with_formats:
        from relweave.formats import FORMATS, messages
        out["system_prompts"] = {k: FORMATS[k].system_prompt() for k in FORMAT_NAMES}
        out["compact_messages"] = {k: messages(FORMATS[k], "CHUNK", compact=True) for k in FORMAT_NAMES}
    return out


if __name__ == "__main__":
    GOLDEN.mkdir(exist_ok=True)
    (GOLDEN / "schema_golden.json").write_text(json.dumps(snapshot(), indent=1, sort_keys=True) + "\n")
    (GOLDEN / "schema_description.txt").write_text(schema_description() + "\n")
