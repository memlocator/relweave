from relweave.chunk import Chunk, chunk


def words(c):
    return len(c.text.split())


def test_empty_and_whitespace():
    assert chunk("") == []
    assert chunk("  \n\n \t ") == []


def test_offsets_are_exact():
    text = "\n  First para here.\n\nSecond para.   \n\n\nThird one.\n"
    for max_words in (2, 3, 5, 200):
        for c in chunk(text, max_words=max_words):
            assert text[c.start:c.end] == c.text


def test_single_chunk_when_it_fits():
    text = "One two three.\n\nFour five."
    cs = chunk(text, max_words=200)
    assert [(c.index, c.text) for c in cs] == [(0, text)]


def test_whole_paragraphs_with_one_paragraph_overlap():
    paras = [f"P{i} " + "w " * 3 + "end." for i in range(5)]  # 5 words each
    text = "\n\n".join(paras)
    cs = chunk(text, max_words=10)
    assert [c.index for c in cs] == list(range(len(cs)))
    # each chunk is a run of whole paragraphs
    for c in cs:
        assert c.text.split("\n\n") == [p for p in paras if p in c.text]
    # chunk k+1 starts with the last paragraph of chunk k
    for a, b in zip(cs, cs[1:]):
        assert b.text.split("\n\n")[0] == a.text.split("\n\n")[-1]
    covered = {p for c in cs for p in c.text.split("\n\n")}
    assert covered == set(paras)
    assert all(words(c) <= 10 for c in cs)


def test_no_overlap_when_it_would_not_fit():
    text = "a b c d e f\n\ng h i j k l"  # 6 + 6 words, max 8: two chunks, no room for overlap
    cs = chunk(text, max_words=8)
    assert [c.text for c in cs] == ["a b c d e f", "g h i j k l"]


def test_single_newlines_split_when_no_blank_lines():
    text = "Line one has words.\nLine two has words.\nLine three has words."
    cs = chunk(text, max_words=8)
    assert len(cs) >= 2
    for c in cs:
        assert text[c.start:c.end] == c.text
        assert c.text.split("\n")[0] in text.split("\n")


def test_single_newlines_kept_inside_paragraph_when_blank_lines_exist():
    text = "Line one.\nline two.\n\nNext paragraph."
    cs = chunk(text, max_words=200)
    assert cs[0].text == text


def test_long_paragraph_split_at_sentences_with_last_sentence_overlap():
    sents = [f"Sentence number {i} has five words." for i in range(6)]  # 6 words each
    text = " ".join(sents)
    cs = chunk(text, max_words=15)
    assert len(cs) > 1
    assert all(words(c) <= 15 for c in cs)
    for c in cs:
        assert text[c.start:c.end] == c.text
        assert c.text.endswith(".")
    for a, b in zip(cs, cs[1:]):
        last = a.text.split(". ")[-1]
        assert b.text.startswith(last.rstrip(".") )
    assert cs[0].start == 0 and cs[-1].end == len(text)


def test_long_paragraph_between_short_ones():
    long = " ".join(f"Sentence {i} is here now." for i in range(10))
    text = f"Intro paragraph.\n\n{long}\n\nOutro paragraph."
    cs = chunk(text, max_words=12)
    for c in cs:
        assert text[c.start:c.end] == c.text
    assert cs[0].text.startswith("Intro")
    assert cs[-1].text.endswith("Outro paragraph.")
    assert all(words(c) <= 12 for c in cs)


def test_sentence_split_needs_capital_quote_or_digit():
    text = "Version 2.5 was released. It is fine. e.g. this stays. " * 3
    cs = chunk(text.strip(), max_words=9)
    for c in cs:
        assert text.strip()[c.start:c.end] == c.text
        assert not c.text.startswith("5 ")


def test_oversized_sentence_falls_back_to_a_word_window_with_overlap():
    text = " ".join(f"w{i}" for i in range(50)) + "."
    cs = chunk(text, max_words=10)
    assert all(words(c) <= 10 for c in cs)
    for c in cs:
        assert text[c.start:c.end] == c.text
    assert cs[0].start == 0 and cs[-1].end == len(text)
    for a, b in zip(cs, cs[1:]):
        assert b.start < a.end  # overlap


def test_long_text_without_sentence_punctuation_uses_lines_then_words():
    lines = "\n".join(" ".join(f"l{k}w{i}" for i in range(8)) for k in range(5))  # no blank lines: lines are paragraphs
    cs = chunk(lines, max_words=10)
    assert all(words(c) <= 10 for c in cs)
    one = chunk(" ".join(["x"] * 100), max_words=10)
    assert all(words(c) <= 10 for c in one) and len(one) > 1


def test_overlong_sentence_inside_a_paragraph_with_newlines():
    long_line = " ".join(f"a{i}" for i in range(30))
    text = f"First short line. {long_line}\nsecond line of it. Last one here."
    cs = chunk(text, max_words=12)
    assert all(words(c) <= 12 for c in cs)
    for c in cs:
        assert text[c.start:c.end] == c.text
    assert cs[-1].end == len(text)


def test_chunk_is_a_dataclass_with_fields():
    c = chunk("Hello there.")[0]
    assert isinstance(c, Chunk) and (c.index, c.start, c.end) == (0, 0, 12)


def test_sentences_break_at_newlines():
    from relweave.chunk import sentence_spans
    text = "Group functions\nThe group has a board. It meets.\n- Tor Amundsen\n- Hilde Lunde\nThey meet."
    parts = [text[a:b] for a, b in sentence_spans(text)]
    assert parts == ["Group functions", "The group has a board.", "It meets.", "- Tor Amundsen", "- Hilde Lunde", "They meet."]
