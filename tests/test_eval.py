import json

from relweave.training import eval as ev
from relweave.training.data import LabelledChunk as Chunk
import relweave.formats as base
from relweave.schema.output import ExtractionOutput


def test_prompt_example_resolves_completely():
    out = ExtractionOutput.model_validate(base.EXAMPLE_OUTPUT)
    cleaned, stats = base.resolve(base.EXAMPLE_CHUNK, out)
    assert stats["unresolved_mentions"] == 0
    assert stats["unresolved_evidence"] == 0
    assert len(cleaned.relations) == 3


def test_parse_output_handles_fences_and_garbage():
    good = json.dumps(base.EXAMPLE_OUTPUT)
    assert base.parse_output("```json\n" + good + "\n```")[0] is not None
    assert base.parse_output("Sure! " + good + " done")[0] is not None
    out, err, _, _ = base.parse_output("no json here")
    assert out is None and err == "no JSON object"


def test_parse_output_drops_illegal_relations_but_keeps_the_rest():
    bad = json.loads(json.dumps(base.EXAMPLE_OUTPUT))
    bad["relations"].append(dict(bad["relations"][0], type="OWNS"))            # unknown type
    bad["relations"].append(dict(bad["relations"][0], type="HELD_AT"))         # Person -> Org illegal
    bad["relations"].append(dict(bad["relations"][0], target="e99"))           # dangling
    out, err, dropped, repaired = base.parse_output(json.dumps(bad))
    assert err is None and dropped == 3 and repaired == 0
    assert len(out.relations) == 3


def test_repair_relabels_only_unambiguous_endpoint_pairs():
    data = json.loads(json.dumps(base.EXAMPLE_OUTPUT))
    data["entities"].append({"id": "e5", "type": "Event", "name": "the meeting", "mentions": ["She"]})
    data["relations"] = [
        {"type": "LOCATED_IN", "source": "e5", "target": "e3", "evidence": "x"},   # Event->Place: only HELD_AT
        {"type": "HELD_AT", "source": "e2", "target": "e1", "evidence": "x"},      # Org->Person: nothing legal
        {"type": "PARTICIPATED_IN", "source": "e2", "target": "e3", "evidence": "x"},  # Org->Place: two legal, ambiguous
    ]
    out, err, dropped, repaired = base.parse_output(json.dumps(data), repair=True)
    assert err is None and repaired == 1 and dropped == 2
    assert [r.type for r in out.relations] == ["HELD_AT"]
    _, _, dropped_plain, repaired_plain = base.parse_output(json.dumps(data))
    assert dropped_plain == 3 and repaired_plain == 0


def test_resolve_drops_hallucinated_mentions_and_evidence():
    out = ExtractionOutput.model_validate({
        "entities": [
            {"id": "e1", "type": "Person", "name": "Maria Lind", "mentions": ["Maria Lind", "Marie"]},
            {"id": "e2", "type": "Org", "name": "Ghost AB", "mentions": ["Ghost AB"]},
        ],
        "relations": [{"type": "EXECUTIVE_OF", "source": "e1", "target": "e2", "evidence": "nope"}],
    })
    cleaned, stats = base.resolve("Maria Lind leads Acme.", out)
    assert [e.id for e in cleaned.entities] == ["e1"]
    assert cleaned.entities[0].mentions == ["Maria Lind"]
    assert stats["unresolved_mentions"] == 2
    assert stats["dropped_relations"] == 1
    assert cleaned.relations == []


def test_resolve_spans_longest_surface_wins_and_expands_all_occurrences():
    text = "Granit Bygg Group AB grew. Granit hired. Granit Bygg Group AB paid. He left."
    out = ExtractionOutput.model_validate({
        "entities": [
            {"id": "e1", "type": "Org", "name": "Granit Bygg Group AB", "mentions": ["Granit Bygg Group AB"]},
            {"id": "e2", "type": "Org", "name": "Granit", "mentions": ["Granit"]},
            {"id": "e3", "type": "Person", "name": "He", "mentions": ["He"]},
        ],
        "relations": [],
    })
    spans, unmatched = base.resolve_spans(text, out)
    assert unmatched == 0
    assert [text[a:b] for a, b in spans["e1"]] == ["Granit Bygg Group AB", "Granit Bygg Group AB"]
    assert [text[a:b] for a, b in spans["e2"]] == ["Granit"]
    assert [text[a:b] for a, b in spans["e3"]] == ["He"]


def _chunk(text, gold, entity_map):
    return Chunk(chunk_id="d1#1", doc_id="d1", start=0, end=len(text), text=text,
                 gold=ExtractionOutput.model_validate(gold), entity_map=entity_map)


GOLD = {
    "entities": [
        {"id": "e1", "type": "Person", "name": "Maria Lind", "mentions": ["Maria Lind", "She"]},
        {"id": "e2", "type": "Org", "name": "Acme Robotics AB", "mentions": ["Acme Robotics AB", "the company"]},
    ],
    "relations": [{"type": "EXECUTIVE_OF", "source": "e1", "target": "e2",
                   "evidence": "Maria Lind runs Acme Robotics AB."}],
}
TEXT = "Maria Lind runs Acme Robotics AB. She likes the company."


def test_perfect_prediction_scores_one():
    chunk = _chunk(TEXT, GOLD, {"e1": "p1", "e2": "o1"})
    pred = json.loads(json.dumps(GOLD))
    pred["entities"][0]["id"], pred["entities"][1]["id"] = "x", "y"
    pred["relations"][0].update(source="x", target="y")
    res = base.finish("d1#1", TEXT, json.dumps(pred), 1.0)
    c = ev.score_chunk(chunk, res)
    assert c.ent_strict == [4, 0, 0]
    assert c.coref == [2, 0, 0]
    assert c.rel_strict == [1, 0, 0]
    assert c.rel_relaxed == [1, 0, 0]
    assert c.rel_pair == [1, 0, 0]
    assert c.rel_type == [1, 0]


def test_wrong_coref_and_missing_relation_are_counted():
    chunk = _chunk(TEXT, GOLD, {"e1": "p1", "e2": "o1"})
    pred = {
        "entities": [
            {"id": "a", "type": "Person", "name": "Maria Lind", "mentions": ["Maria Lind"]},
            {"id": "b", "type": "Person", "name": "She", "mentions": ["She"]},
            {"id": "c", "type": "Org", "name": "Acme Robotics AB", "mentions": ["Acme Robotics AB"]},
        ],
        "relations": [],
    }
    res = base.finish("d1#1", TEXT, json.dumps(pred), 1.0)
    c = ev.score_chunk(chunk, res)
    assert c.ent_strict == [3, 0, 1]
    assert c.coref == [0, 0, 2]
    assert c.rel_strict == [0, 0, 1]
    s = ev.summarize([chunk], [res])
    assert s["schema_validity"] == 1.0
    assert s["rel_strict"]["f1"] == 0.0


def test_invalid_output_counts_as_all_missed():
    chunk = _chunk(TEXT, GOLD, {"e1": "p1", "e2": "o1"})
    res = base.finish("d1#1", TEXT, "garbage", 1.0)
    s = ev.summarize([chunk], [res])
    assert s["schema_validity"] == 0.0
    assert s["ent_strict"]["recall"] == 0.0


def test_find_all_matches_whole_words_only():
    text = "The father said he and her brother met at H&M. Hehe."
    assert [text[a:b] for a, b in base.find_all(text, "he")] == ["he"]
    start = text.index(" her ") + 1
    assert base.find_all(text, "her") == [(start, start + 3)]
    assert len(base.find_all(text, "H&M")) == 1
    assert base.find_all(text, "") == []


def test_pair_and_type_scored_separately():
    """A right pair with a wrong type is a detection hit but a typing miss; an unaligned endpoint is a wrong pair."""
    gold = {("EXECUTIVE_OF", "e1", "e2"), ("FOUNDED", "e1", "e3")}
    pred = {("EMPLOYED_BY", "e1", "e2"), ("FOUNDED", "e1", "e3"), ("OWNS_STAKE_IN", "e1", None)}
    assert ev._pair_counts(gold, pred) == ([2, 1, 0], [1, 1])


def test_rel_by_type_splits_the_relation_counts_and_default_summary_is_unchanged():
    chunk = _chunk(TEXT, GOLD, {"e1": "p1", "e2": "o1"})
    pred = {"entities": GOLD["entities"],
            "relations": [{"type": "EMPLOYED_BY", "source": "e1", "target": "e2", "evidence": "Maria Lind runs Acme Robotics AB."}]}
    res = base.finish("d1#1", TEXT, json.dumps(pred), 1.0)
    plain = ev.summarize([chunk], [res])
    assert "rel_by_type" not in plain
    by = ev.summarize([chunk], [res], by_type=True)
    assert by["rel_by_type"]["EXECUTIVE_OF"] == {"tp": 0, "fp": 0, "fn": 1, "f1": 0.0}
    assert by["rel_by_type"]["EMPLOYED_BY"] == {"tp": 0, "fp": 1, "fn": 0, "f1": 0.0}
    assert {k: v for k, v in by.items() if k != "rel_by_type"} == plain
