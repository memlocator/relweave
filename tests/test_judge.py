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


CASE_TEXT = "04-25-00453-CV\nMP II, Inc., Appellant\nThe appeal is from the County Court at Law No. 3."


def test_a_fact_named_two_ways_is_scored_once(tmp_path):
    # the labeller names the case by a caption and lists its docket number; a system added later names it by the docket
    # number, and `add` sees only the pool's claim names, not the labeller's mentions
    gold = write(tmp_path / "gold.jsonl", [{"id": "d1", "text": CASE_TEXT,
        "entities": [ent("MP II, Inc. v. Kollars", "Case", "04-25-00453-CV"), ent("MP II, Inc.", "Org"),
                     ent("County Court at\nLaw No. 3", "Court")],
        "relations": [{"source": "MP II, Inc. v. Kollars", "type": "PARTY_IN", "target": "MP II, Inc."}]}])
    sys_ = write(tmp_path / "sys.jsonl", [{"id": "d1",
        "entities": [ent("04-25-00453-CV", "Case"), ent("MP II, Inc.", "Org")],
        "relations": [{"source": "04-25-00453-CV", "type": "PARTY_IN", "target": "MP II, Inc."}]}])
    out = tmp_path / "pool"
    judge.items(out, {"gold": gold}, decoys=0)
    assert judge.add(out, "sys", sys_) == 1
    true = {("MP II, Inc. v. Kollars", "PARTY_IN", "MP II, Inc."): "MP II, Inc., Appellant",
            ("04-25-00453-CV", "PARTY_IN", "MP II, Inc."): "MP II, Inc., Appellant"}
    for j in ("a", "b"):
        verdicts(out, j, true)
    s = judge.score(out, ["a", "b"], boot=20)
    assert s["facts"] and s["true"] == 1
    assert s["sources"]["sys"]["recall"] == 1.0 and s["sources"]["gold"]["recall"] == 1.0
    (out / "aliases.jsonl").unlink()  # without aliases the one fact is two claims, each source missing one
    s = judge.score(out, ["a", "b"], boot=20)
    assert not s["facts"] and s["true"] == 2 and s["sources"]["sys"]["recall"] == 0.5


def test_aliases_keep_entities_of_different_types_apart(tmp_path):
    gold = write(tmp_path / "gold.jsonl", [{"id": "d1", "text": "State v. Smith. Smith appeals.",
        "entities": [ent("State v. Smith", "Case", "Smith"), ent("Smith", "Person")], "relations": []}])
    out = tmp_path / "pool"
    judge.items(out, {"gold": gold}, decoys=0)
    assert json.loads(open(out / "aliases.jsonl").read())["groups"] == []
    assert judge.key("County Court at\n   Law No. 3") == judge.key("County Court at Law No. 3")


def test_brief_renders_schema_definitions():
    b = judge.brief(BUSINESS)
    assert "EXECUTIVE_OF" in b and "## Rules" in b and "verbatim" in b


def test_neighbour_types_share_endpoints():
    near = judge.neighbour_types(BUSINESS)
    assert "EMPLOYED_BY" in near["EXECUTIVE_OF"] and "EXECUTIVE_OF" not in near["EXECUTIVE_OF"]


def test_labels_from_graphs(tmp_path):
    jgf = {"graph": {"id": "doc7", "label": "doc7", "directed": True, "type": "relweave",
                     "metadata": {"text": TEXT, "chunks": []},
                     "nodes": {"e1": {"label": "Anna Berg", "metadata": {"kind": "entity", "type": "Person", "mentions": [
                         {"text": "Anna Berg", "start": 0, "end": 9, "chunk": 0}]}},
                               "e2": {"label": "Acme", "metadata": {"kind": "entity", "type": "Org", "mentions": [
                                   {"text": "Acme", "start": 32, "end": 36, "chunk": 0}, {"text": "The firm", "start": 59, "end": 67, "chunk": 1}]}},
                               "doc": {"label": "doc7", "metadata": {"kind": "document"}}},
                     "edges": [{"id": "r1", "source": "e1", "target": "e2", "relation": "EXECUTIVE_OF",
                                "metadata": {"kind": "relation", "modality": "asserted"}},
                               {"id": "r2", "source": "e1", "target": "e2", "relation": "FOUNDED",
                                "metadata": {"kind": "relation", "modality": "negated"}}]}}
    (tmp_path / "doc7.json").write_text(json.dumps(jgf))
    assert judge.labels_from_graphs([tmp_path / "doc7.json"], tmp_path / "out.jsonl") == 1
    rec = json.loads((tmp_path / "out.jsonl").read_text())
    assert rec["id"] == "doc7" and rec["text"] == TEXT
    assert rec["relations"] == [{"source": "Anna Berg", "type": "EXECUTIVE_OF", "target": "Acme"}]
    assert {"Acme", "The firm"} == set(next(e for e in rec["entities"] if e["name"] == "Acme")["mentions"])
