"""Split a document into overlapping chunks of whole paragraphs (or whole sentences for a long paragraph).

Every chunk is a verbatim slice of the document: text[c.start:c.end] == c.text, so offsets inside a chunk map to
document offsets by adding c.start.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_WORDS = 200  # words per chunk: what the published models were trained on, at run time and for labelling
BLANK_LINES = re.compile(r"\n[ \t\r]*\n\s*")
# a sentence ends at . ! ? before a capital, quote or digit, and at any line break (headings and list items are
# never glued to the next sentence)
SENTENCE_END = re.compile(r"(?<=[.!?])[ \t]+(?=[A-Z\"'“‘\d])|[ \t]*\n\s*")


# Dense layout (tables, lists, key-value lines) is chunked smaller: on staff tables the model read 0.96 F1 at 80-word
# chunks against 0.65 at 200 (2026-10-15, data/dense), while prose keeps MAX_WORDS.
DENSE_WORDS = 80
LAYOUT_LINE = re.compile(r"^\s*(\||[-*\u2022]\s|\d+[.)]\s|(\*\*)?[A-Z][\w /&().-]{0,40}(\*\*)?:(\*\*)?\s)")
HEADING_LINE = re.compile(r"^#{1,6}\s+\S")


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    start: int  # character offsets into the document
    end: int
    context: str = ""  # lines from before the chunk the model needs to read it (a table's header row, its heading)
    dense: bool = False  # cut from a dense block (a list, table or key-value block longer than dense_words; _dense)

    @property
    def model_text(self) -> str:
        """What the model reads: the context lines, then the chunk. Mentions and evidence are still located in text."""
        return f"{self.context}\n{self.text}" if self.context else self.text


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


def _dense(text: str, span: Span) -> bool:
    """A paragraph of at least three lines, most of them table rows, list items or key-value lines."""
    lines = [x for x in text[span[0]:span[1]].split("\n") if x.strip()]
    return len(lines) >= 3 and sum(bool(LAYOUT_LINE.match(x)) for x in lines) * 2 >= len(lines)


def _lines(text: str, span: Span) -> list[Span]:
    return [(span[0] + a, span[0] + b) for a, b in _split(text[span[0]:span[1]], re.compile(r"\n"))]


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _lead(text: str, block_start: int) -> str:
    """The line introducing a dense block: the last non-empty line before it, when it ends with a colon ("The board
    consists of:") and is not itself a list item or table row."""
    prev = [x for x in text[:block_start].split("\n") if x.strip()]
    line = prev[-1].strip() if prev else ""
    return line if line.endswith(":") and not LAYOUT_LINE.match(line) else ""


def _context(text: str, start: int, block_start: int | None = None) -> str:
    """For a chunk starting at start: the nearest heading line above it and, when it starts inside a table, the
    table's header row (and separator row) and the nearest one-cell group row above the chunk (a row such as
    "| Ministry of Justice |" that names the group the following rows belong to)."""
    before = text[:start].split("\n")
    before = before[:-1]  # the last piece is empty (start at a line start) or the start of the chunk's own line
    head = next((x for x in reversed(before) if HEADING_LINE.match(x)), "")
    rows = []
    line_end = text.find("\n", start)
    own = text[start:line_end if line_end >= 0 else None]
    if own.lstrip().startswith("|") and before and before[-1].lstrip().startswith("|"):
        k = len(before) - 1
        while k > 0 and before[k - 1].lstrip().startswith("|"):
            k -= 1
        table = before[k:]
        rows.append(table[0])
        if len(table) > 1 and set(table[1].replace("|", "").strip()) <= set("-: "):
            rows.append(table[1])
        group = next((x for x in reversed(table[1:]) if len([c for c in _cells(x) if c]) == 1
                      and set(x.replace("|", "").strip()) - set("-: ")), "")
        if group and group not in rows and not own.strip() == group.strip():
            rows.append(group)
    lead = _lead(text, block_start) if block_start is not None and not rows else ""
    if lead and text[start:start + len(lead)] == lead:
        lead = ""
    return "\n".join(x for x in (head, lead, *rows) if x)


def _join_layout_lines(text: str, paras: list[Span]) -> list[Span]:
    """Join runs of one-line paragraphs that are layout lines (a table or list in text without blank lines, where
    every line is its own paragraph) into one paragraph, so the run is recognised as dense."""
    out: list[Span] = []
    run: list[Span] = []
    for p in paras + [None]:
        if p is not None and "\n" not in text[p[0]:p[1]] and LAYOUT_LINE.match(text[p[0]:p[1]]):
            run.append(p)
            continue
        if len(run) >= 3:
            out.append((run[0][0], run[-1][1]))
        else:
            out.extend(run)
        run = []
        if p is not None:
            out.append(p)
    return out


def chunk(text: str, max_words: int = MAX_WORDS, dense_words: int | None = DENSE_WORDS) -> list[Chunk]:
    """Chunks of whole paragraphs up to max_words words, one paragraph of overlap between neighbours. A paragraph
    longer than max_words is split at sentence boundaries into pieces of at most max_words (a single longer
    sentence stays whole), overlapping by the last sentence. A dense paragraph (mostly table rows, list items or
    key-value lines) is split at line ends into pieces of at most dense_words (None: as prose), and each of its
    pieces carries the heading above it, the line introducing a list ("... consists of:") and, inside a table, the
    table's header row as context."""
    spans: list[tuple[Span, int | None]] = []  # span, start of its dense block (None: prose)
    run: list[Span] = []  # paragraphs not yet packed (short ones between long ones)

    def flush():
        nonlocal run
        spans.extend((s, None) for s in _pack(text, run, max_words))
        run = []

    for p in _join_layout_lines(text, paragraphs(text)) if dense_words else paragraphs(text):
        if dense_words and _dense(text, p) and _words(text, p) > dense_words:
            flush()
            spans.extend((s, p[0]) for s in _pack(text, _lines(text, p), dense_words))
        elif _words(text, p) > max_words:
            flush()
            spans.extend((s, None) for s in _pack(text, _units(text, p, max_words), max_words))
        else:
            run.append(p)
    flush()
    return [Chunk(i, text[a:b], a, b, _context(text, a, block) if block is not None else "",
                  dense=block is not None)
            for i, ((a, b), block) in enumerate(spans)]


def sentence_spans(text: str) -> list[Span]:
    """Sentence spans of a whole document (the splitter chunk() uses)."""
    return sentences(text, (0, len(text)))


def sentence_at(spans: list[Span], pos: int) -> Span | None:
    """The sentence span containing character offset pos (None between sentences)."""
    import bisect
    k = bisect.bisect_right([a for a, _ in spans], pos) - 1
    return spans[k] if k >= 0 and pos < spans[k][1] else None
