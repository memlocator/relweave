import types

from relweave.tagger import TAGS, decode, encode, spans, windows


class Tok:
    """Whitespace tokenizer with offsets, splitting long words in two (like sub-word pieces of a rare surname)."""

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        import re
        offs = []
        for m in re.finditer(r"\S+", text):
            a, b = m.span()
            if b - a > 6:
                offs += [(a, a + 4), (a + 4, b)]
            else:
                offs.append((a, b))
        return {"input_ids": list(range(len(offs))), "offset_mapping": offs}


TEXT = "The board: Amalie Gjerdrum, Birger Krogvig and Acme Holdings."


def test_echo_encoding_round_trips_gold():
    sp = spans(TEXT, [("Amalie Gjerdrum", "Person"), ("Birger Krogvig", "Person"), ("Acme Holdings", "Org")])
    ids, pos, char, tags = encode(Tok(), TEXT, sp)
    assert all(a >= 0 for a, _ in char) and len(pos) == len(tags)
    got = {(TEXT[a:b], t) for a, b, t in decode(TEXT, char, tags)}
    assert {("Amalie Gjerdrum", "Person"), ("Birger Krogvig", "Person")} <= got


def test_span_cut_inside_a_word_is_extended():
    ids, pos, char, _ = encode(Tok(), TEXT)
    pred = [0] * len(char)
    start = next(k for k, (a, b) in enumerate(char) if TEXT[a:b] == "Amalie")
    pred[start] = TAGS.index("B-Person")
    pred[start + 1] = TAGS.index("I-Person")  # "Gjer" only: the rest of the surname untagged
    assert [TEXT[a:b] for a, b, _ in decode(TEXT, char, pred)] == ["Amalie Gjerdrum"]


def test_windows_cover_the_text_at_offsets():
    text = "\n".join(f"line {i} " + "word " * 20 for i in range(40))
    ws = windows(text, 100)
    assert len(ws) > 1 and all(text[o:o + len(w)] == w for o, w in ws)


def test_augment_adds_only_missing_entities():
    from relweave.schema.output import Entity, ExtractionOutput
    from relweave.tagger import EntityTagger
    t = EntityTagger.__new__(EntityTagger)
    t.tag = lambda model, tok, text: [("Acme Holdings", "Org", 0, 0), ("Birger Krogvig", "Person", 0, 0)]
    out = ExtractionOutput.model_construct(entities=[Entity(id="e1", type="Person", name="Birger Krogvig",
                                                            mentions=["Birger Krogvig"])], relations=[])
    r = types.SimpleNamespace(output=out, raw="E1 Person: Birger Krogvig\nS1\n")
    r = t.augment(None, None, TEXT, r)
    assert [e.name for e in r.output.entities] == ["Birger Krogvig", "Acme Holdings"]
    assert "E2 Org: Acme Holdings" in r.raw


def test_span_starting_inside_a_word_starts_at_the_word():
    ids, pos, char, _ = encode(Tok(), TEXT)
    pred = [0] * len(char)
    k = next(k for k, (a, b) in enumerate(char) if TEXT[a:b] == "Krog")
    pred[k + 1] = TAGS.index("B-Person")  # only "vig" tagged
    assert [TEXT[a:b] for a, b, _ in decode(TEXT, char, pred)] == ["Krogvig"]


def test_name_mentions_keep_names_and_drop_pronouns():
    from relweave.tagger import name_mentions
    e = {"name": "Kelvorn Pumps GmbH", "mentions": ["Kelvorn Pumps GmbH", "Kelvorn", "it", "the company"]}
    assert name_mentions(e) == ["Kelvorn Pumps GmbH", "Kelvorn"]
