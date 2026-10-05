import json
import re

import jsonschema
import pytest

from relweave.chunk import chunk
from relweave.graph import SCHEMA_FILE, Evidence, Graph
from relweave.merge import merge
from relweave.records import ChunkEntity, ChunkGraph, ChunkRelation, Mention

TEXT = ("Acme Corp makes tools. It sells them in Bergen. Bolt Ltd is a partner of Acme Corp.\n\n"
        "Acme Corp owns a stake in Bolt Ltd. The company is based in Bergen.")


def occ(text, name, nth=0):
    return [Mention(name, m.start()) for m in re.finditer(re.escape(name), text)][nth:nth + 1]


def build():
    cs = chunk(TEXT, max_words=200)
    assert len(cs) == 1
    ents = [ChunkEntity("e1", "Org", "Acme Corp", [Mention("Acme Corp", m.start()) for m in re.finditer("Acme Corp", TEXT)] + occ(TEXT, "It")),
            ChunkEntity("e2", "Org", "Bolt Ltd", [Mention("Bolt Ltd", m.start()) for m in re.finditer("Bolt Ltd", TEXT)]),
            ChunkEntity("e3", "Place", "Bergen", [Mention("Bergen", m.start()) for m in re.finditer("Bergen", TEXT)])]
    owns = TEXT.index("Acme Corp owns a stake in Bolt Ltd.")
    rels = [ChunkRelation("OWNS_STAKE_IN", "e1", "e2", score=1.5, origin="generator", sentence=(owns, owns + 35)),
            ChunkRelation("ASSOCIATE_OF", "e1", "e2", score=0.7, origin="head")]  # no evidence: sentence found from mentions
    g = ChunkGraph(0, ents, rels, start=0, end=len(TEXT))
    return merge([g], TEXT, doc_id="doc1", model={"base": "m"}, version="9.9")


def schema():
    return json.loads(SCHEMA_FILE.read_text())


def test_output_validates_against_the_vendored_jgf_v2_schema():
    jsonschema.validate(build().to_jgf(), schema())
    jsonschema.validate(json.loads(build().to_json(include_text=False)), schema())


def test_graph_level_content():
    d = build().to_jgf()["graph"]
    assert (d["id"], d["directed"], d["type"]) == ("doc1", True, "relweave")
    m = d["metadata"]
    assert m["text"] == TEXT and m["schema"]["name"] == "business" and m["relweave_version"] == "9.9"
    assert m["chunks"] == [{"index": 0, "start": 0, "end": len(TEXT)}] and m["model"] == {"base": "m"}
    assert "text" not in build().to_jgf(include_text=False)["graph"]["metadata"]


def test_every_name_mention_has_exact_offsets_and_its_sentence():
    nodes = build().to_jgf()["graph"]["nodes"]
    n = 0
    for nid, node in nodes.items():
        for m in node["metadata"].get("mentions", []):
            n += 1
            assert TEXT[m["start"]:m["end"]] == m["text"] and m["located_by"] == "match"
            s = m["sentence"]
            assert TEXT[s["start"]:s["end"]] == s["text"] and s["start"] <= m["start"] and m["end"] <= s["end"]
    assert n >= 7
    assert nodes["doc"]["metadata"] == {"kind": "document"} and nodes["doc"]["label"] == "doc1"


def test_relations_evidence_sentences_and_mentioned_in_edges():
    edges = build().to_jgf()["graph"]["edges"]
    rel = {e["relation"]: e for e in edges if e["metadata"]["kind"] == "relation"}
    owns = rel["OWNS_STAKE_IN"]["metadata"]["evidence"][0]
    assert owns["chunk"] == 0 and owns["sentence"]["text"] == "Acme Corp owns a stake in Bolt Ltd."
    found = rel["ASSOCIATE_OF"]["metadata"]["evidence"][0]["sentence"]["text"]  # first sentence with both endpoints
    assert found == "Bolt Ltd is a partner of Acme Corp."
    mi = [e for e in edges if e["relation"] == "MENTIONED_IN"]
    assert len(mi) == 3 and all(e["target"] == "doc" and e["directed"] for e in mi)
    assert {e["source"]: e["metadata"]["count"] for e in mi}["e3"] == 2


def test_a_symmetric_relation_is_undirected_and_others_directed():
    edges = {e["relation"]: e for e in build().to_jgf()["graph"]["edges"]}
    assert edges["ASSOCIATE_OF"]["directed"] is False
    assert edges["OWNS_STAKE_IN"]["directed"] is True


def test_round_trip():
    g = build()
    assert Graph.from_json(g.to_json()) == g
    g.relations[0].score = None
    assert Graph.from_jgf(g.to_jgf()) == g


def test_to_networkx_agrees_with_the_jgf():
    g = build()
    n = g.to_networkx()
    jgf = g.to_jgf()["graph"]
    assert set(n.nodes) == set(jgf["nodes"]) and n.number_of_edges() == len(jgf["edges"])
    assert n.nodes["e1"]["label"] == "Acme Corp" and n.nodes["doc"]["kind"] == "document"
    assert n.edges["e1", "e2", "r2"]["directed"] is False


def test_graphml(tmp_path):
    import networkx as nx
    g = build()
    g.to_graphml(tmp_path / "g.graphml")
    back = nx.read_graphml(tmp_path / "g.graphml")
    assert back.number_of_nodes() == len(g.entities) + 1 and back.number_of_edges() == len(g.relations) + len(g.entities)


def test_mentioned_in_property_is_derived():
    recs = build().mentioned_in
    assert all(TEXT[r["start"]:r["end"]] == r["mention"] for r in recs) and len(recs) >= 7


def test_evidence_dataclass():
    assert Evidence(1).sentence is None
