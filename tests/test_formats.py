import pytest

import relweave.formats as base
from relweave.formats import EXAMPLE_CHUNK, EXAMPLE_OUTPUT, FORMATS, LineFormat, messages
from relweave.schema import ExtractionOutput, canonical_triple

LINES = LineFormat()


def _keys(out: ExtractionOutput) -> tuple[set, set]:
    """Format-independent content: entity (type, surfaces) and relation (type, endpoints, modality)."""
    by = {e.id: (str(e.type), tuple(e.mentions)) for e in out.entities}
    ents = set(by.values())
    rels = {canonical_triple(r.type, by[r.source], by[r.target]) + (r.modality.value,) for r in out.relations}
    return ents, rels


def test_example_renders_numbered_sentences_and_encodes_evidence_as_reference():
    rendered = LINES.render_input(EXAMPLE_CHUNK)
    assert rendered.splitlines()[0].startswith("S1: Maria Lind")
    assert rendered.splitlines()[1].startswith("S2: She added")
    encoded = LINES.encode(ExtractionOutput.model_validate(EXAMPLE_OUTPUT), EXAMPLE_CHUNK)
    assert "E2 Org: Acme Robotics AB | the company | Acme" in encoded
    assert "R OPERATES_IN E2 E4 hedged S2" in encoded
    assert "R EXECUTIVE_OF E1 E2 asserted S1 title=chief executive" in encoded


@pytest.mark.parametrize("fmt", ["json", "lines"])
def test_round_trip_through_parser_and_resolver(fmt):
    gold = ExtractionOutput.model_validate(EXAMPLE_OUTPUT)
    raw = FORMATS[fmt].encode(gold, EXAMPLE_CHUNK)
    res = base.finish("x", EXAMPLE_CHUNK, raw, 0.0, fmt=fmt)
    assert res.error is None
    assert res.unresolved_mentions == 0 and res.unresolved_evidence == 0 and res.illegal_relations == 0
    assert _keys(res.output) == _keys(gold)


def test_line_decoder_tolerates_noise_and_flags_bad_sentence_refs():
    raw = ("E1 Person: Maria Lind | She\nE2 Org: Acme Robotics AB\ngarbage line\n"
           "R EXECUTIVE_OF E1 E2 asserted S9\nR EXECUTIVE_OF E1 E2 asserted S1 title=CEO; bogus\n")
    res = base.finish("x", EXAMPLE_CHUNK, raw, 0.0, fmt="lines")
    assert res.error is None
    assert res.unresolved_evidence == 1           # S9 does not exist
    assert [r.attributes for r in res.output.relations] == [{"title": "CEO"}]
    assert base.finish("x", EXAMPLE_CHUNK, "nothing useful", 0.0, fmt="lines").error.startswith("lines")


def test_prompt_example_is_accepted_by_its_own_grammar():
    xgr = pytest.importorskip("xgrammar")
    grammar = xgr.Grammar.from_ebnf(LINES.ebnf())
    encoded = LINES.encode(ExtractionOutput.model_validate(EXAMPLE_OUTPUT), EXAMPLE_CHUNK) + "\n"
    assert xgr.testing._is_grammar_accept_string(grammar, encoded)
    assert not xgr.testing._is_grammar_accept_string(grammar, "E1 Company: Acme\n")  # unknown node type


def test_system_prompt_is_format_specific():
    assert "one line per entity" in messages(LINES, "x")[0]["content"]
    assert "Output only JSON" in messages(FORMATS["json"], "x")[0]["content"]


# ---- sentences format -----------------------------------------------------------

from relweave.formats import SentenceFormat  # noqa: E402

SENTS = SentenceFormat()


def test_sentences_encoding_groups_relations_under_ordered_headers():
    encoded = SENTS.encode(ExtractionOutput.model_validate(EXAMPLE_OUTPUT), EXAMPLE_CHUNK)
    lines = encoded.splitlines()
    assert lines.index("S1") < lines.index("S2")
    assert "R EXECUTIVE_OF E1 E2 asserted title=chief executive" in lines
    assert lines[lines.index("S2") + 1] == "R OPERATES_IN E2 E4 hedged"


def test_sentences_grammar_forbids_going_back_or_repeating_a_sentence():
    xgr = pytest.importorskip("xgrammar")
    g = xgr.Grammar.from_ebnf(SENTS.ebnf_for(3))
    ok = "E1 Person: A\nE2 Org: B\nS1\nR EXECUTIVE_OF E1 E2 asserted\nS3\nR MET_WITH E1 E1 hedged\n"
    assert xgr.testing._is_grammar_accept_string(g, ok)
    back = ok + "S2\nR MET_WITH E1 E1 hedged\n"
    again = "E1 Person: A\nS1\nR MET_WITH E1 E1 hedged\nS1\nR MET_WITH E1 E1 hedged\n"
    past_end = "E1 Person: A\nS4\nR MET_WITH E1 E1 hedged\n"
    for bad in (back, again, past_end):
        assert not xgr.testing._is_grammar_accept_string(g, bad)


def test_display_name_prefers_proper_names_over_descriptions():
    from relweave.formats import display_name
    assert display_name(["the company name", "Tetra Pak", "it"]) == "Tetra Pak"
    assert display_name(["The service", "Spotify"]) == "Spotify"
    assert display_name(["Acme Robotics AB", "Acme"]) == "Acme Robotics AB"
    assert display_name(["the company", "it"]) == "the company"


def test_generation_prompt_ends_where_training_answer_starts():
    """The extraction prompt must be an exact prefix of the training text, for every student base."""
    from transformers import AutoTokenizer
    from relweave.formats import chat_messages, generation_prompt, render_chat
    msgs = chat_messages("Volvo is based in Gothenburg.", "sentences", compact=True)
    answer = "E1 Org: Volvo"
    for model_id in ("Qwen/Qwen3-1.7B", "unsloth/qwen3-4b-instruct-2507-unsloth-bnb-4bit"):
        tok = AutoTokenizer.from_pretrained(model_id)
        full = render_chat(tok, [*msgs, {"role": "assistant", "content": answer}])
        prompt = generation_prompt(tok, msgs)
        assert full.startswith(prompt) and full[len(prompt):].startswith(answer), model_id


def test_sentence_split_keeps_initials_and_titles_together():
    from relweave.formats import sentence_bounds
    text = "Johan W. Arnberg met Mr. Eberth at St. Andrew's. It moved to the U.S. The company grew."
    assert [text[a:b] for a, b in sentence_bounds(text)] == [
        "Johan W. Arnberg met Mr. Eberth at St. Andrew's.", "It moved to the U.S.", "The company grew."]


def test_relations_are_written_in_canonical_order_within_a_sentence():
    """Relation lines of one sentence come out sorted by (source, target, type), whatever order
    the labels list them in, so the training target has a single correct order."""
    from relweave.formats import FORMATS
    from relweave.schema import ExtractionOutput
    text = "Anna and Bo founded Acme in Oslo."
    ev = text
    gold = ExtractionOutput.model_validate({
        "entities": [{"id": "a", "type": "Person", "name": "Anna", "attributes": {}, "mentions": ["Anna"]},
                     {"id": "b", "type": "Person", "name": "Bo", "attributes": {}, "mentions": ["Bo"]},
                     {"id": "c", "type": "Org", "name": "Acme", "attributes": {}, "mentions": ["Acme"]},
                     {"id": "d", "type": "Place", "name": "Oslo", "attributes": {}, "mentions": ["Oslo"]}],
        "relations": [{"type": "HEADQUARTERED_IN", "source": "c", "target": "d", "attributes": {}, "modality": "asserted", "evidence": ev},
                      {"type": "FOUNDED", "source": "b", "target": "c", "attributes": {}, "modality": "asserted", "evidence": ev},
                      {"type": "FOUNDED", "source": "a", "target": "c", "attributes": {}, "modality": "asserted", "evidence": ev}]})
    lines = [l for l in FORMATS["sentences"].encode(gold, text).splitlines() if l.startswith("R ")]
    assert lines == ["R FOUNDED E1 E3 asserted", "R FOUNDED E2 E3 asserted", "R HEADQUARTERED_IN E3 E4 asserted"]


def test_grounded_grammar_allows_relations_only_between_entities_of_that_sentence():
    import xgrammar as xgr
    from transformers import AutoTokenizer
    from relweave.generator import mentions_by_sentence
    from relweave.formats import FORMATS
    f = FORMATS["sentences"]
    text = "Acme AB is based in Oslo. Maria Lind likes tea."
    answer = "E1 Org: Acme AB\nE2 Place: Oslo\nE3 Person: Maria Lind\n"
    grounded = mentions_by_sentence(answer, text, f)
    assert grounded == {1: {1, 2}, 2: {3}}
    g = f.ebnf_for(2, {1: "Org", 2: "Place", 3: "Person"}, grounded)
    assert "block2" not in g  # sentence 2 has one entity: no relation can be written there
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tok))
    ok = xgr.GrammarMatcher(compiler.compile_grammar(g))
    assert ok.accept_string(answer + "S1\nR HEADQUARTERED_IN E1 E2 asserted\n")
    bad = xgr.GrammarMatcher(compiler.compile_grammar(g))
    assert not bad.accept_string(answer + "S1\nR EMPLOYED_BY E3 E1 asserted\n")  # Maria Lind is not in sentence 1


from toy import S as TOY  # noqa: E402


def test_business_grammar_is_unchanged_by_the_schema_parameter():
    f = FORMATS["sentences"]
    assert f.with_schema(f.schema).ebnf_for(3) == f.ebnf_for(3)


def test_grammar_uses_the_given_schema():
    g = FORMATS["sentences"].with_schema(TOY).ebnf_for(2)
    assert '"OPERATES_IN"' in g and '"HEADQUARTERED_IN"' not in g and '"Org"' in g and '"Event"' not in g


def test_conditioned_prompt_carries_the_rendered_schema():
    m = messages(FORMATS["sentences"].with_schema(TOY), "Acme operates in Oslo.", conditioned=True)
    assert TOY.render() in m[0]["content"] and len(m) == 2


def test_with_schema_copies_and_leaves_the_registered_format_alone():
    f = FORMATS["sentences"]
    g = f.with_schema(TOY)
    assert g is not f and g.schema is TOY and f.schema is not TOY


def test_typed_grammar_and_json_grammar_follow_the_schema():
    g = FORMATS["sentences"].with_schema(TOY).ebnf_for(2, {1: "Org", 2: "Place"})
    assert '"OPERATES_IN E1 E"' in g and "HEADQUARTERED_IN" not in g
    js = FORMATS["json"].with_schema(TOY)
    assert "Event" not in js.system_prompt() and "OPERATES_IN" in js.system_prompt()


def test_unconditioned_prompt_for_other_schema_has_no_business_example():
    m = messages(FORMATS["sentences"].with_schema(TOY), "Acme operates in Oslo.")
    assert len(m) == 2 and "HEADQUARTERED_IN" not in m[0]["content"]


def test_resolve_and_finish_work_under_another_schema():
    from relweave.formats import finish
    raw = "E1 Org: Acme\nE2 Place: Oslo\nS1\nR OPERATES_IN E1 E2 asserted"
    r = finish("c", "Acme operates in Oslo.", raw, 0.0, fmt="sentences", schema=TOY)
    assert r.output is not None and len(r.output.relations) == 1


def test_out_of_schema_entity_is_dropped_and_counted_not_fatal():
    from relweave.formats import finish
    raw = "E1 Org: Acme\nE2 Place: Oslo\nE3 Event: Fair\nS1\nR OPERATES_IN E1 E2 asserted\nR OPERATES_IN E1 E3 asserted"
    r = finish("c", "Acme operates in Oslo at the Fair.", raw, 0.0, fmt="sentences", schema=TOY)
    assert r.output is not None and r.dropped_entities == 1
    assert [e.type for e in r.output.entities] == ["Org", "Place"] and len(r.output.relations) == 1


def test_equal_non_singleton_business_keeps_example():
    import copy
    f = FORMATS["sentences"]
    clone = copy.copy(f.schema)
    assert len(messages(f.with_schema(clone), "x")) == 4


def test_chat_messages_threads_schema_and_conditioned():
    from relweave.formats import chat_messages
    m = chat_messages("Acme operates in Oslo.", "sentences", schema=TOY, conditioned=True)
    assert len(m) == 2 and TOY.render() in m[0]["content"] and "HEADQUARTERED_IN" not in m[0]["content"]
