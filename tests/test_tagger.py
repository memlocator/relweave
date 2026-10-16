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
    # a shared lower-case word or an article does not make a description a name
    assert name_mentions({"name": "Acme Company", "mentions": ["the company", "Acme"]}) == ["Acme Company", "Acme"]
    assert name_mentions({"name": "The Lego Group", "mentions": ["the firm", "The Group", "Lego"]}) == \
        ["The Lego Group", "The Group", "Lego"]


def test_single_copy_encoding_tags_the_only_copy():
    sp = spans(TEXT, [("Amalie Gjerdrum", "Person"), ("Acme Holdings", "Org")])
    ids, pos, char, tags = encode(Tok(), TEXT, sp, echo=False)
    echo_ids, *_ = encode(Tok(), TEXT, sp)
    assert len(echo_ids) > len(ids) and pos == list(range(len(ids)))
    assert {("Amalie Gjerdrum", "Person"), ("Acme Holdings", "Org")} <= {(TEXT[a:b], t) for a, b, t in decode(TEXT, char, tags)}


class TinyLM:
    """A causal-LM shape for layer_states: .model.layers (each adds 10 to every state), .model.norm (identity), and a
    forward that fills hidden_states like transformers (embeddings, each layer's output, the norm's output)."""

    def __init__(self, n_layers=3):
        import torch

        class Add(torch.nn.Module):
            def forward(self, x):
                return x + 10

        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Add() for _ in range(n_layers)])
        self.model.norm = torch.nn.Identity()
        self.device = "cpu"

    def __call__(self, input_ids, output_hidden_states=False, logits_to_keep=None, use_cache=None, **kw):
        import torch
        n = input_ids.shape[1]
        h = torch.tensor([[[float(i), 0.0] for i in range(n)]])  # embedding of token i: [i, 0]
        states = [h]
        for layer in self.model.layers:
            h = layer(h)
            states.append(h)
        states[-1] = self.model.norm(h)
        return types.SimpleNamespace(hidden_states=states if output_hidden_states else None)


def test_layer_states_match_hidden_states():
    import torch

    from relweave.states import layer_states
    m = TinyLM()
    ids = torch.tensor([[0, 1, 2, 3]])
    full = m(ids, output_hidden_states=True).hidden_states
    for k in (-1, -2, 1, 3):
        assert torch.equal(layer_states(m, [k], input_ids=ids)[0], full[k])


def test_lookahead_features_read_the_next_tokens_states():
    from relweave.tagger import EntityTagger

    t = EntityTagger.__new__(EntityTagger)
    t.layers, t.lookahead = [-1], 2
    assert t.dim(2) == 6
    f = t._features(TinyLM(), [0, 1, 2, 3], [0, 3])  # last layer of 3: token i's state is [30 + i, 0]
    assert f[:, ::2].tolist() == [[30, 31, 32], [33, 33, 33]]  # past the end: the last token's state


def test_span_ends_at_a_line_break():
    text = "Board:\n- Amalie Bjerknes\n- Birger Krogvig\n"
    a = text.index("Amalie")
    b = text.index("- Birger") + 1
    assert decode(text, [(a, a + 6), (a + 6, b)], [TAGS.index("B-Person"), TAGS.index("I-Person")]) == \
        [(a, a + len("Amalie Bjerknes"), "Person")]
