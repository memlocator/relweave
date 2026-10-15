import json

import pytest

from relweave import judge
from relweave.schema import BUSINESS


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


TEXT = "Anna Berg is chief executive of Acme. Acme is based in Oslo. The firm hired Per Lund."


def ent(name, typ, *mentions):
    return {"name": name, "type": typ, "mentions": [name, *mentions]}


def pool(tmp_path):
    gold = write(tmp_path / "gold.jsonl", [{"id": "d1", "text": TEXT,
        "entities": [ent("Anna Berg", "Person"), ent("Acme", "Org", "the firm"), ent("Oslo", "Place"), ent("Per Lund", "Person")],
        "relations": [{"source": "Anna Berg", "type": "EXECUTIVE_OF", "target": "Acme"},
                      {"source": "Acme", "type": "HEADQUARTERED_IN", "target": "Oslo"}]}])
    sys_ = write(tmp_path / "sys.jsonl", [{"id": "d1",
        "entities": [ent("Anna Berg", "Person"), ent("the firm", "Org"), ent("Per Lund", "Person"), ent("Oslo", "Place")],
        "relations": [{"source": "Per Lund", "type": "EMPLOYED_BY", "target": "the firm"},
                      {"source": "Anna Berg", "type": "EMPLOYED_BY", "target": "Oslo"}]}])
    out = tmp_path / "pool"
    judge.items(out, {"gold": gold, "sys": sys_}, BUSINESS, decoys=2, seed=3)
    return out


def verdicts(out, name, true_of):
    items = [json.loads(x) for x in open(out / "items.jsonl")]
    with open(out / f"verdicts_{name}.jsonl", "w") as w:
        for d in items:
            vs = []
            for c in d["claims"]:
                q = true_of.get((c["source"], c["type"], c["target"]))
                vs.append({"n": c["n"], "verdict": "true" if q else "false", "quote": q or ""})
            w.write(json.dumps({"id": d["id"], "verdicts": vs}) + "\n")


TRUE = {("Anna Berg", "EXECUTIVE_OF", "Acme"): "Anna Berg is chief executive of Acme",
        ("Acme", "HEADQUARTERED_IN", "Oslo"): "Acme is based in Oslo",
        ("Per Lund", "EMPLOYED_BY", "Acme"): "The firm hired Per Lund"}


def test_lowercase_mention_merges_with_named_entity(tmp_path):
    out = pool(tmp_path)
    claims = [c for d in map(json.loads, open(out / "items.jsonl")) for c in d["claims"]]
    # the system's "the firm" is the gold's Acme (listed as its mention): one entity, named Acme
    assert ("Per Lund", "EMPLOYED_BY", "Acme") in {(c["source"], c["type"], c["target"]) for c in claims}
    assert not any("the firm" in (c["source"], c["target"]) for c in claims)


def test_decoys_are_hidden_and_keyed(tmp_path):
    out = pool(tmp_path)
    key = [c for d in map(json.loads, open(out / "key.jsonl")) for c in d["claims"]]
    assert sum(1 for c in key if c["decoy"]) >= 1
    assert all(c["sources"] == [] for c in key if c["decoy"])
    assert "sources" not in open(out / "items.jsonl").read()


def test_score_check_and_types(tmp_path):
    out = pool(tmp_path)
    for j in ("a", "b"):
        verdicts(out, j, TRUE)
    r = judge.check(out, ["a", "b"])
    assert r["judges"]["a"]["decoys_accepted"] == 0 and r["disagreement"][0] == 0
    s = judge.score(out, ["a", "b"], boot=50)
    assert s["true"] == 3
    assert s["sources"]["gold"]["precision"] == 1.0 and s["sources"]["gold"]["recall"] == pytest.approx(2 / 3)
    assert s["sources"]["sys"]["precision"] == 0.5 and s["sources"]["sys"]["recall"] == pytest.approx(1 / 3)
    t = judge.by_type(out, ["a", "b"], without="EMPLOYED_BY")
    assert t["without EMPLOYED_BY"]["true"] == 2 and t["EMPLOYED_BY"]["sources"]["sys"][0] == pytest.approx(2 / 3)


def test_true_without_verbatim_quote_counts_false(tmp_path):
    out = pool(tmp_path)
    verdicts(out, "a", {k: "not in the text" for k in TRUE})
    keyed, ver, dropped = judge.load(out, ["a"])
    assert dropped["a"] == 3 and not any(ver.values())


def test_score_needs_judges(tmp_path):
    out = pool(tmp_path)
    with pytest.raises(ValueError):
        judge.score(out, [])


def test_add_and_merge_keep_earlier_verdicts(tmp_path):
    out = pool(tmp_path)
    for j in ("a", "b"):
        verdicts(out, j, TRUE)
    new = write(tmp_path / "new.jsonl", [{"id": "d1", "entities": [ent("Acme", "Org"), ent("Oslo", "Place"),
                                                                    ent("Anna Berg", "Person")],
        "relations": [{"source": "Acme", "type": "HEADQUARTERED_IN", "target": "Oslo"},
                      {"source": "Anna Berg", "type": "BOARD_MEMBER_OF", "target": "Acme"}]}])
    assert judge.add(out, "new", new) == 1
    for j in ("a", "b"):
        write(out / f"verdicts_add_{j}.jsonl", [{"id": "d1", "verdicts": [
            {"n": c["n"], "verdict": "false", "quote": ""} for c in json.loads(open(out / "items_add.jsonl").readline())["claims"]]}])
    judge.merge(out, ["a", "b"])
    s = judge.score(out, ["a", "b"], boot=20)
    assert s["true"] == 3 and s["sources"]["new"]["true"] == 1 and s["sources"]["new"]["wrote"] == 2


def test_brief_renders_schema_definitions():
    b = judge.brief(BUSINESS)
    assert "EXECUTIVE_OF" in b and "## Rules" in b and "verbatim" in b


def test_neighbour_types_share_endpoints():
    near = judge.neighbour_types(BUSINESS)
    assert "EMPLOYED_BY" in near["EXECUTIVE_OF"] and "EXECUTIVE_OF" not in near["EXECUTIVE_OF"]
