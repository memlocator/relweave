"""Split a document into overlapping chunks of whole paragraphs (or whole sentences for a long paragraph).

Every chunk is a verbatim slice of the document: text[c.start:c.end] == c.text, so offsets inside a chunk map to
document offsets by adding c.start.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

BLANK_LINES = re.compile(r"\n[ \t\r]*\n\s*")
# a sentence ends at . ! ? before a capital, quote or digit, and at any line break (headings and list items are
# never glued to the next sentence)
SENTENCE_END = re.compile(r"(?<=[.!?])[ \t]+(?=[A-Z\"'“‘\d])|[ \t]*\n\s*")


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    start: int  # character offsets into the document
    end: int


Span = tuple[int, int]


def _trim(text: str, start: int, end: int) -> Span | None:
    piece = text[start:end]
    stripped = piece.strip()
    if not stripped:
        return None
    lead = len(piece) - len(piece.lstrip())
    return start + lead, start + lead + len(stripped)


def _split(text: str, separator: re.Pattern) -> list[Span]:
    spans, pos = [], 0
    for m in list(separator.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        span = _trim(text, pos, end)
        if span:
            spans.append(span)
        if m:
            pos = m.end()
    return spans


def paragraphs(text: str) -> list[Span]:
    """Paragraph spans: blank-line separated; single-newline separated when the text has no blank line."""
    if BLANK_LINES.search(text.strip()):
        return _split(text, BLANK_LINES)
    return _split(text, re.compile(r"\n"))


def sentences(text: str, span: Span) -> list[Span]:
    lo, hi = span
    out = []
    for a, b in _split(text[lo:hi], SENTENCE_END):
        out.append((lo + a, lo + b))
    return out


def _units(text: str, span: Span, max_words: int) -> list[Span]:
    """Packing units of a long paragraph: sentences; a sentence over the limit falls back to its lines, and a line
    over the limit to word windows of half the limit (so a window and its overlap fit in one chunk)."""
    out: list[Span] = []
    for sent in sentences(text, span):
        pieces = [sent] if _words(text, sent) <= max_words else [
            x for x in ((sent[0] + a, sent[0] + b) for a, b in _split(text[sent[0]:sent[1]], re.compile(r"\n")))]
        for piece in pieces:
            if _words(text, piece) <= max_words:
                out.append(piece)
                continue
            marks = [(m.start() + piece[0], m.end() + piece[0]) for m in re.finditer(r"\S+", text[piece[0]:piece[1]])]
            step = max(1, max_words // 2)
            out += [(marks[i][0], marks[min(i + step, len(marks)) - 1][1]) for i in range(0, len(marks), step)]
    return out


def _words(text: str, span: Span) -> int:
    return len(text[span[0]:span[1]].split())


def _pack(text: str, units: list[Span], max_words: int) -> list[Span]:
    """Pack consecutive units into spans of at most max_words words (a single bigger unit stays alone); the next
    span starts with the last unit of the previous one when the previous held more than one and it still fits."""
    out: list[Span] = []
    cur: list[Span] = []
    for u in units:
        if cur and _words(text, (cur[0][0], u[1])) > max_words:
            out.append((cur[0][0], cur[-1][1]))
            keep = cur[-1:] if len(cur) > 1 and _words(text, (cur[-1][0], u[1])) <= max_words else []
            cur = keep
        cur.append(u)
    if cur:  # always holds a unit that is not yet in a span
        out.append((cur[0][0], cur[-1][1]))
    return out


def chunk(text: str, max_words: int = 200) -> list[Chunk]:
    """Chunks of whole paragraphs up to max_words words, one paragraph of overlap between neighbours. A paragraph
    longer than max_words is split at sentence boundaries into pieces of at most max_words (a single longer
    sentence stays whole), overlapping by the last sentence."""
    spans: list[Span] = []
    run: list[Span] = []  # paragraphs not yet packed (short ones between long ones)
    for p in paragraphs(text):
        if _words(text, p) > max_words:
            spans += _pack(text, run, max_words)
            run = []
            spans += _pack(text, _units(text, p, max_words), max_words)
        else:
            run.append(p)
    spans += _pack(text, run, max_words)
    return [Chunk(i, text[a:b], a, b) for i, (a, b) in enumerate(spans)]


def sentence_spans(text: str) -> list[Span]:
    """Sentence spans of a whole document (the splitter chunk() uses)."""
    return sentences(text, (0, len(text)))


def sentence_at(spans: list[Span], pos: int) -> Span | None:
    """The sentence span containing character offset pos (None between sentences)."""
    import bisect
    k = bisect.bisect_right([a for a, _ in spans], pos) - 1
    return spans[k] if k >= 0 and pos < spans[k][1] else None
