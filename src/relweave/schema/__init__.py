"""Extraction schemas and the per-chunk extraction output.

A schema is written as Pydantic classes (see base.py: one class per entity and relation type, the docstring is
the definition the model reads). SCHEMAS maps a schema name to its Schema; importing this package registers the
bundled schemas (business, politics, legal, sports). The output models the generator emits per chunk
(ExtractionOutput, with its entities and relations) live in relweave.schema.output and validate against a schema
given as context={"schema": ...}.
"""
from relweave.schema.base import Entity, Relation, Schema

SCHEMAS: dict[str, Schema] = {}


def register(schema: Schema) -> Schema:
    if SCHEMAS.get(schema.name, schema) is not schema:
        raise ValueError(f"a different schema is already registered as {schema.name}")
    SCHEMAS[schema.name] = schema
    return schema


def registered_schema(name: str) -> Schema:
    """A schema by a name read from data or a config file (chunk records, train_summary.json, relweave_config.json,
    head_config.json): only already registered schemas, never code. A user schema must be loaded first by the caller
    with load_schema("path/to/module.py:NAME")."""
    if name not in SCHEMAS:
        raise ValueError(f"unknown schema {name!r}; known: {', '.join(SCHEMAS)}. Load your own schema first with "
                         "load_schema('path/to/module.py:NAME') (CLI: --schema path/to/module.py:NAME)")
    return SCHEMAS[name]


def load_schema(spec: str | Schema) -> Schema:
    """A Schema from a registered name ("business"), "path/to/module.py:NAME" or "package.module:NAME" (the
    module's attribute NAME must be a Schema). A loaded schema is registered, so chunk records naming it resolve;
    a file is executed once per process (loading it again returns the same object). Only for specs the caller
    gives (API keyword, CLI option): names read from data or config files go through registered_schema."""
    import importlib
    import importlib.util
    import sys
    from pathlib import Path

    if isinstance(spec, Schema):
        return spec
    if ":" not in spec:
        return registered_schema(spec)
    where, attr = spec.rsplit(":", 1)
    if where.endswith(".py") or Path(where).is_file():
        path = Path(where).resolve()
        key = f"relweave_user_schema_{abs(hash(str(path)))}"
        module = sys.modules.get(key)
        if module is None:
            loader = importlib.util.spec_from_file_location(key, path)
            if loader is None:
                raise ValueError(f"cannot load {path} as a Python module")
            module = importlib.util.module_from_spec(loader)
            sys.modules[key] = module
            try:
                loader.loader.exec_module(module)
            except BaseException:
                del sys.modules[key]
                raise
    else:
        module = importlib.import_module(where)
    if not hasattr(module, attr):
        raise AttributeError(f"{where} has no attribute {attr}")
    schema = getattr(module, attr)
    if not isinstance(schema, Schema):
        raise TypeError(f"{where}:{attr} is not a Schema (got {type(schema).__name__})")
    return register(schema)


# Importing the domain modules registers their schemas; at the end because they import register from here.
from relweave.schema import business, legal, politics, sports  # noqa: E402,F401
from relweave.schema.business import BUSINESS  # noqa: E402
from relweave.schema.output import (  # noqa: E402
    MAX_ENTITIES_PER_CHUNK,
    MAX_MENTIONS_PER_ENTITY,
    MAX_RELATIONS_PER_CHUNK,
    ExtractionOutput,
    Modality,
    canonical_triple,
    json_schema,
    parse_lenient,
    schema_description,
)

__all__ = ["Entity", "Relation", "Schema", "SCHEMAS", "register", "load_schema", "registered_schema", "BUSINESS",
           "ExtractionOutput", "Modality", "parse_lenient", "canonical_triple", "json_schema", "schema_description", "MAX_ENTITIES_PER_CHUNK",
           "MAX_MENTIONS_PER_ENTITY", "MAX_RELATIONS_PER_CHUNK"]
