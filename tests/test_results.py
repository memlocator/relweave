"""Stored generator answers are re-read under the schema they were written for, not under business."""
import pytest

from relweave.formats import finish, load_results, save_results
from toy import S as TOY

TEXT = "Acme operates in Oslo. Ann and Bo are siblings."
RAW = "E1 Org: Acme\nE2 Place: Oslo\nE3 Person: Ann\nE4 Person: Bo\nS1\nR OPERATES_IN E1 E2 asserted\nS2\nR FAMILY_OF E3 E4 asserted"


def _saved(tmp_path):
    r = finish("c1", TEXT, RAW, 0.0, fmt="sentences", schema=TOY)
    assert len(r.output.relations) == 2 and r.schema == "toy"
    path = tmp_path / "results.jsonl"
    save_results([r], path)
    return path


def test_reparse_uses_the_schema_argument(tmp_path):
    path = _saved(tmp_path)
    (r,) = load_results(path, {"c1": TEXT}, schema=TOY)
    assert [x.type for x in r.output.relations] == ["OPERATES_IN", "FAMILY_OF"]


def test_a_per_chunk_schema_mapping_is_accepted(tmp_path):
    path = _saved(tmp_path)
    (r,) = load_results(path, {"c1": TEXT}, schema={"c1": TOY})
    assert len(r.output.relations) == 2


def test_the_stored_schema_name_is_used_when_it_is_registered(tmp_path):
    from relweave.schema import SCHEMAS
    path = _saved(tmp_path)
    SCHEMAS["toy"] = TOY
    try:
        (r,) = load_results(path, {"c1": TEXT})
        assert len(r.output.relations) == 2
        (r,) = load_results(path)  # the stored output (no re-parse) validates under it too
        assert len(r.output.relations) == 2
    finally:
        SCHEMAS.pop("toy")


def test_an_unknown_stored_schema_is_an_error_not_a_silent_business_parse(tmp_path):
    path = _saved(tmp_path)
    with pytest.raises(ValueError, match="schema 'toy'"):
        load_results(path, {"c1": TEXT})
