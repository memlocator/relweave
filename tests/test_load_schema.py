"""A user's own schema is given as "path/to/module.py:NAME" or "package.module:NAME", without editing relweave."""
import pytest

from relweave.schema import BUSINESS, SCHEMAS, Schema, load_schema

MODULE = '''
from relweave.schema import Entity, Relation, Schema


class Ship(Entity):
    """A vessel."""


class Port(Entity):
    """A harbour."""


class DockedAt(Relation[Ship, Port]):
    """The Ship lies in the Port."""


SHIPPING = Schema(name="shipping_test", entities=[Ship, Port], relations=[DockedAt])
NOT_A_SCHEMA = 3
'''


@pytest.fixture
def schema_file(tmp_path):
    p = tmp_path / "my_schema.py"
    p.write_text(MODULE)
    yield p
    SCHEMAS.pop("shipping_test", None)


def test_a_registered_name_or_a_schema_object_is_returned_as_is():
    assert load_schema("business") is BUSINESS
    assert load_schema(BUSINESS) is BUSINESS


def test_a_file_path_with_an_attribute_loads_the_schema_and_registers_it(schema_file):
    s = load_schema(f"{schema_file}:SHIPPING")
    assert isinstance(s, Schema) and s.edge_names() == ("DOCKED_AT",)
    assert SCHEMAS["shipping_test"] is s
    assert load_schema(f"{schema_file}:SHIPPING") is s  # loading the file again gives the same object
    assert load_schema("shipping_test") is s  # and its name now resolves


def test_a_module_path_with_an_attribute_loads_the_schema(schema_file, monkeypatch):
    monkeypatch.syspath_prepend(str(schema_file.parent))
    assert load_schema("my_schema:SHIPPING").name == "shipping_test"


def test_errors_name_the_problem(schema_file):
    with pytest.raises(ValueError, match="unknown schema 'nope'.*business"):
        load_schema("nope")
    with pytest.raises(TypeError, match="NOT_A_SCHEMA is not a Schema"):
        load_schema(f"{schema_file}:NOT_A_SCHEMA")
    with pytest.raises(AttributeError, match="MISSING"):
        load_schema(f"{schema_file}:MISSING")


def test_a_schema_named_in_data_never_loads_code(tmp_path):
    import json
    import sys
    from relweave.training.data import load_chunks
    evil = tmp_path / "evil.py"
    evil.write_text("raise SystemExit('executed')\n")
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"chunk_id": "c", "doc_id": "d", "text": "x", "gold": None, "schema": f"{evil}:S"}) + "\n")
    with pytest.raises(ValueError, match="load_schema"):
        load_chunks(p)
    assert not any("evil" in m for m in sys.modules)
