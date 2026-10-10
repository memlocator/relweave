"""Clean up text dumped from PDFs, web pages and scans before extraction: lines hard-wrapped inside a paragraph are
joined (a word hyphenated across the break is rejoined), page numbers, form feeds and running headers or footers are
dropped, HTML entities decoded and runs of spaces collapsed. Paragraph breaks (blank lines) are kept, since the
chunker cuts at them.

tidy() is conservative: a line break is kept after sentence-ending punctuation when the next line is short or looks
like a heading or list item, so lists and headings stay separate lines.
"""
from __future__ import annotations

import collections
import html
import re

PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\s*$", re.I)
BULLET = re.compile(r"^\s*(?:[•\-*–·]|\d{1,2}[.)]|[a-z][.)])\s+")
END = re.compile(r"[.!?:;\"'”’)]\s*$")


def _running_lines(lines: list[str]) -> set[str]:
    """Running headers and footers: short lines within two lines of a page number or page break, repeated on two or
    more pages. Without page numbers nothing is dropped (a repeated sentence in the body stays)."""
    marks = [k for k, s in enumerate(lines) if PAGE_NUMBER.match(s) or "\f" in s]
    near = collections.Counter(lines[j].strip() for k in marks for j in range(max(0, k - 2), min(len(lines), k + 3))
                               if 0 < len(lines[j].strip()) <= 80 and not PAGE_NUMBER.match(lines[j]))
    return {s for s, n in near.items() if n >= 2}


def tidy(text: str) -> str:
    text = html.unescape(text).replace("\r\n", "\n").replace("\f", "\n").replace("­", "")
    lines = text.split("\n")
    running = _running_lines(lines)
    lines = [s for s in lines if not PAGE_NUMBER.match(s) and s.strip() not in running]
    width = max((len(s) for s in lines), default=0)
    out: list[str] = []
    for s in lines:
        s = re.sub(r"[ \t]+", " ", s).strip()
        if not s:
            if out and out[-1] != "":
                out.append("")
            continue
        prev = out[-1] if out else ""
        # a wrapped line continues the previous one; after a list item only a lower-case line does; table rows
        # (markdown pipes) are never joined
        wrapped = prev and not BULLET.match(s) and not s.startswith("|") and not prev.startswith("|") and (
            s[:1].islower() if BULLET.match(prev) else not END.search(prev) or (len(prev) >= 0.7 * width and s[:1].islower()))
        if wrapped and re.search(r"[A-Za-z]-$", prev) and s[:1].islower():
            out[-1] = prev[:-1] + s  # a word hyphenated across the break
        elif wrapped:
            out[-1] = prev + " " + s
        else:
            out.append(s)
    text = "\n".join(out).strip()
    return re.sub(r" ([,.;:!?])", r"\1", text)
