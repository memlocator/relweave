import types

import pytest
import torch

from relweave.schema import SCHEMAS

from relweave.schema import MAX_ENTITIES_PER_CHUNK, ExtractionOutput
from relweave.schema.output import Entity, Relation
from relweave.chunk import Chunk
from relweave.extract import locate, to_chunk_graph, grouped


def test_locate_whole_words_with_document_offsets():
    c = Chunk(1, "Acme owns Acmeville. Acme is big.", 100, 133)
    assert [m.start for m in locate(c, "Acme")] == [100, 121]
    assert locate(c, "Nowhere") == [type(locate(c, "Acme")[0])("Nowhere", None)]


def out(n_entities=2, rel=True):
    ents = [Entity(id=f"e{i}", type="Org", name=f"Org{i}", mentions=[f"Org{i}"]) for i in range(1, n_entities + 1)]
    rels = [Relation(type="OWNS_STAKE_IN", source="e1", target="e2", evidence="x")] if rel else []
    return ExtractionOutput.model_construct(entities=ents, relations=rels)


def test_to_chunk_graph_scores_and_origin():
    c = Chunk(0, "Org1 owns Org2.", 10, 25)
    o = out()
    scores = {"OWNS_STAKE_IN 1 2": 1.25}
    g = to_chunk_graph(c, o, written={("OWNS_STAKE_IN", "e1", "e2")}, scores=scores)
    assert g.chunk == 0 and g.warnings == []
    assert [m.start for m in g.entities[0].mentions] == [10]
    r = g.relations[0]
    assert (r.score, r.origin) == (1.25, "generator")
    g = to_chunk_graph(c, o, written=set(), scores={})
    assert (g.relations[0].score, g.relations[0].origin) == (None, "head")


def test_entity_cap_warning():
    c = Chunk(3, "x", 0, 1)
    g = to_chunk_graph(c, out(MAX_ENTITIES_PER_CHUNK, rel=False), written=set(), scores={})
    assert any("cap" in w and "chunk 3" in w for w in g.warnings)


def test_grouped_halves_the_group_on_oom_and_keeps_order():
    calls = []

    def probe(pairs):
        calls.append(len(pairs))
        if len(pairs) > 5:
            raise torch.OutOfMemoryError("oom")
        return list(pairs)

    got = grouped(list(range(20)), 20, probe, on_oom=lambda: None)
    assert [x for part in got for x in part] == list(range(20))
    assert calls[:3] == [20, 10, 5]


def test_grouped_gives_up_at_group_one():
    def probe(pairs):
        raise torch.OutOfMemoryError("oom")
    with pytest.raises(torch.OutOfMemoryError):
        grouped([1, 2, 3], 4, probe, on_oom=lambda: None)


def test_generic_mentions_are_pinned_to_one_occurrence_near_the_name():
    text = "Acme Corp makes tools. It sells them. Beta Ltd bought a plant. It paid."
    c = Chunk(0, text, 100, 100 + len(text))
    o = ExtractionOutput.model_construct(
        entities=[Entity(id="e1", type="Org", name="Acme Corp", mentions=["Acme Corp", "It"]),
                  Entity(id="e2", type="Org", name="Beta Ltd", mentions=["Beta Ltd", "It"])], relations=[])
    g = to_chunk_graph(c, o, written=set(), scores={})
    its = [[m.start for m in e.mentions if m.text == "It"] for e in g.entities]
    assert its == [[100 + text.index("It")], [100 + text.index("It paid")]]


def test_heading_occurrences_are_dropped_only_when_the_mention_occurs_elsewhere():
    text = "Group functions and governance\n\nThe group has a board. It meets monthly.\n\nHalvmane Group"
    c = Chunk(0, text, 0, len(text))
    o = ExtractionOutput.model_construct(
        entities=[Entity(id="e1", type="Org", name="Group", mentions=["Group", "The group"])], relations=[])
    g = to_chunk_graph(c, o, written=set(), scores={})
    assert len(g.entities) == 1  # never dropped
    starts = {m.text: m.start for m in g.entities[0].mentions}
    assert starts["The group"] == text.index("The group")
    assert starts["Group"] == text.index("Halvmane Group") + len("Halvmane ")  # the non-heading occurrence


def test_a_mention_only_on_a_heading_line_keeps_its_offset_and_entity():
    text = "Group functions and governance\n\nThe board meets monthly."
    c = Chunk(0, text, 10, 10 + len(text))
    o = ExtractionOutput.model_construct(entities=[Entity(id="e1", type="Org", name="Group", mentions=["Group"])], relations=[])
    g = to_chunk_graph(c, o, written=set(), scores={})
    assert [(e.name, [m.start for m in e.mentions]) for e in g.entities] == [("Group", [10])]


def test_list_items_are_not_headings_and_keep_their_entities_and_relations():
    text = "members:\n- Tor Amundsen\n- Hilde Lunde\n- Petter Lunde\nThey meet monthly."
    c = Chunk(0, text, 0, len(text))
    names = ["Tor Amundsen", "Hilde Lunde", "Petter Lunde"]
    o = ExtractionOutput.model_construct(
        entities=[Entity(id=f"e{i}", type="Person", name=n, mentions=[n]) for i, n in enumerate(names, 1)],
        relations=[Relation(type="ASSOCIATE_OF", source="e1", target="e2", evidence="")])
    g = to_chunk_graph(c, o, written=set(), scores={})
    assert [e.name for e in g.entities] == names and len(g.relations) == 1
    assert all(m.start == text.index(e.name) for e in g.entities for m in e.mentions)
    from relweave.extract import heading_spans
    assert heading_spans(text) == [] and heading_spans("1. First item\n2. Second item\nBody text goes on here.") == []


def test_heading_spans_single_short_unpunctuated_line_before_text():
    from relweave.extract import heading_spans
    text = "Overview\nThe firm makes tools and sells them everywhere in the whole world today and tomorrow.\n\nMore text."
    assert heading_spans(text) == [(0, 8)]
def test_a_single_unpunctuated_line_is_text_not_a_heading():
    c = Chunk(0, "Acme owns Bolt", 0, 14)
    o = ExtractionOutput.model_construct(entities=[Entity(id="e1", type="Org", name="Acme", mentions=["Acme"])], relations=[])
    assert len(to_chunk_graph(c, o, written=set(), scores={}).entities) == 1


def test_oom_cleanup_runs_after_the_except_block():
    import sys
    seen = []

    def probe(pairs):
        if len(pairs) > 2:
            raise torch.OutOfMemoryError("oom")
        return pairs

    grouped(list(range(8)), 8, probe, on_oom=lambda: seen.append(sys.exc_info()[0]))
    assert seen and all(x is None for x in seen)


def test_memory_fraction_is_computed_once_and_never_raised():
    from relweave.extract import fraction
    mb = 2**20
    f = fraction(free=6000 * mb, total=8000 * mb, reserve_mb=500, previous=None)
    assert f == pytest.approx(5500 / 8000)
    assert fraction(free=7000 * mb, total=8000 * mb, reserve_mb=500, previous=f) == f  # more free later: keep
    assert fraction(free=3000 * mb, total=8000 * mb, reserve_mb=500, previous=f) == pytest.approx(2500 / 8000)
    with pytest.raises(RuntimeError, match="reserve"):
        fraction(free=400 * mb, total=8000 * mb, reserve_mb=500, previous=None)


def _dirs(tmp_path, gen_schema="business", head_schema="business", head_union=None):
    """A generator and a head directory in the exported layout, without weights (enough for the checks)."""
    import json
    g, h = tmp_path / "generator", tmp_path / "head"
    g.mkdir()
    h.mkdir()
    (g / "relweave_config.json").write_text(json.dumps({"schema": gen_schema, "format": "sentences",
                                                        "compact_prompt": True, "conditioned": False}))
    (h / "head_config.json").write_text(json.dumps({"schema": head_schema, **({"union": head_union} if head_union else {})}))
    return g, h


@pytest.fixture
def no_cap(monkeypatch):
    import relweave.extract as ex
    monkeypatch.setattr(ex, "cap_memory", lambda reserve_mb: None)


def test_the_schema_must_match_the_trained_head_and_generator_and_cuda_is_required(tmp_path, no_cap):
    from relweave.extract import ChunkExtractor
    other = next(s for n, s in SCHEMAS.items() if n != "business")
    g, h = _dirs(tmp_path)
    with pytest.raises(ValueError, match="generator was trained on schema 'business'"):
        ChunkExtractor(g, h, schema=other)
    with pytest.raises(ValueError, match="cuda"):
        ChunkExtractor(g, h, device="cpu")
    (tmp_path / "x").mkdir()
    g2, h2 = _dirs(tmp_path / "x", head_schema=other.name)
    with pytest.raises(ValueError, match=f"pair head was trained on schema '{other.name}'"):
        ChunkExtractor(g2, h2)


def test_the_schema_defaults_to_the_generator_s_and_union_settings_come_from_the_head(tmp_path, no_cap):
    from relweave.extract import ChunkExtractor
    g, h = _dirs(tmp_path, head_union={"cut": -3.0, "margin": 1.5, "max_relations": 40})
    ce = ChunkExtractor(g, h)
    assert ce.schema.name == "business" and (ce.cut, ce.margin) == (-3.0, 1.5) and ce.compact
    assert (ChunkExtractor(g, h, cut=-1.0, margin=0.0).cut, ChunkExtractor(g, h, margin=0.0).margin) == (-1.0, 0.0)


def test_a_user_schema_file_is_accepted_when_generator_and_head_were_trained_on_it(tmp_path, no_cap):
    from relweave.extract import ChunkExtractor
    (tmp_path / "ships.py").write_text(
        "from relweave.schema import Entity, Relation, Schema\n"
        "class Ship(Entity):\n    'A vessel.'\n"
        "class Port(Entity):\n    'A harbour.'\n"
        "class DockedAt(Relation[Ship, Port]):\n    'The Ship lies in the Port.'\n"
        "SHIPS = Schema(name='ships_extract_test', entities=[Ship, Port], relations=[DockedAt])\n")
    g, h = _dirs(tmp_path, "ships_extract_test", "ships_extract_test")
    with pytest.raises(ValueError, match="load_schema"):  # a name from a config file never loads code
        ChunkExtractor(g, h)
    try:
        assert ChunkExtractor(g, h, schema=f"{tmp_path / 'ships.py'}:SHIPS").schema.edge_names() == ("DOCKED_AT",)
    finally:
        SCHEMAS.pop("ships_extract_test", None)


def test_generator_settings_fall_back_to_a_training_run_summary(tmp_path):
    import json
    from relweave.extract import generator_settings
    (tmp_path / "adapter").mkdir()
    (tmp_path / "train_summary.json").write_text(json.dumps({"compact_prompt": True, "schema": "legal"}))
    assert generator_settings(tmp_path)["schema"] == "legal"
    assert generator_settings(tmp_path / "adapter")["compact_prompt"] is True
    with pytest.raises(FileNotFoundError):
        generator_settings(tmp_path / "other" / "nowhere")


def test_pick_batch_takes_the_largest_size_that_fits():
    from relweave.extract import pick_batch
    mb = 2**20
    assert pick_batch(free_bytes=1000 * mb, row_bytes=60 * mb) == 8
    assert pick_batch(free_bytes=500 * mb, row_bytes=60 * mb) == 4
    assert pick_batch(free_bytes=100 * mb, row_bytes=60 * mb) == 1


class FakeGen:
    """extract_batch that records batch sizes and runs out of memory for batches of `oom_at` or more."""

    def __init__(self, oom_at=99, bad=()):
        self.sizes, self.oom_at, self.bad = [], oom_at, set(bad)

    def extract_batch(self, items, batch_size):
        self.sizes.append(len(items))
        if len(items) >= self.oom_at:
            raise torch.OutOfMemoryError("oom")
        if any(k in self.bad for k, _ in items):
            raise ValueError("bad chunk")
        return [types.SimpleNamespace(chunk_id=k, text=t) for k, t in items]


def _ce(gen):
    from relweave.extract import ChunkExtractor
    ce = ChunkExtractor.__new__(ChunkExtractor)
    ce._gen = gen
    return ce


def test_generate_many_batches_by_length_and_maps_results_back():
    gen = FakeGen()
    items = [(f"{d}:{i}", "x" * n) for d, i, n in [(0, 0, 50), (0, 1, 10), (1, 0, 30), (1, 1, 20), (2, 0, 40)]]
    out = _ce(gen).generate_many(items, batch_size=2)
    assert gen.sizes == [2, 2, 1]
    assert {k: r.text for k, r in out.items()} == dict(items)


def test_generate_many_halves_on_oom_without_losing_chunks(monkeypatch):
    import relweave.extract as ex
    monkeypatch.setattr(ex, "_free_gpu", lambda: None)
    gen = FakeGen(oom_at=4)
    items = [(f"0:{i}", "x" * (i + 1)) for i in range(10)]
    ce = _ce(gen)
    out = ce.generate_many(items, batch_size=8)
    assert gen.sizes[:2] == [8, 4] and set(gen.sizes[2:]) == {2}
    assert set(out) == {k for k, _ in items} and all(out[k].text == t for k, t in items)
    assert ce.last_batch_size == 2


def test_generate_many_isolates_a_failing_chunk():
    gen = FakeGen(bad={"0:2"})
    items = [(f"0:{i}", "x" * (i + 1)) for i in range(4)]
    out = _ce(gen).generate_many(items, batch_size=4)
    assert isinstance(out["0:2"], ValueError) and all(out[k].text for k in ("0:0", "0:1", "0:3"))


def test_evidence_sentence_is_located_in_the_document_and_head_relations_have_none():
    text = "Intro line here. Org1 owns Org2. Tail."
    c = Chunk(2, text, 50, 50 + len(text))
    o = out()
    o.relations[0].evidence = "Org1 owns Org2."
    o.relations.append(Relation(type="OWNS_STAKE_IN", source="e2", target="e1", evidence=""))
    g = to_chunk_graph(c, o, written={("OWNS_STAKE_IN", "e1", "e2")}, scores={})
    assert g.relations[0].sentence == (50 + text.index("Org1 owns"), 50 + text.index("Org1 owns") + 15)
    assert g.relations[1].sentence is None
    assert (g.start, g.end) == (50, 50 + len(text))


def test_memory_fraction_counts_this_process_own_reserved_memory():
    from relweave.extract import fraction
    mb = 2**20
    # 3000 MB free now, but this process already holds 2000 MB: its budget is (3000 + 2000 - 500) / 8000
    assert fraction(free=3000 * mb, total=8000 * mb, reserve_mb=500, previous=None, own=2000 * mb) == pytest.approx(4500 / 8000)


def test_cap_memory_uses_memory_reserved(monkeypatch):
    import relweave.extract as ex
    mb = 2**20
    calls = []
    monkeypatch.setattr(ex, "_CAP", None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (3000 * mb, 8000 * mb))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 2000 * mb)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: types.SimpleNamespace(total_memory=8000 * mb))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda f: calls.append(f))
    ex.cap_memory(500)
    assert calls == [pytest.approx(4500 / 8000)]
