"""Markdown tables as sentences: each data row is rewritten with its column headers ("Name: Ana Varga; Title: Chief
Executive Officer; Company: Kessler Logistics."), so a row states its facts on its own and keeps its meaning when the
table is split across chunks, far from its header.

Header: the row above a |---| separator; without one, the first row. A two-column table without a separator is read
as key: value pairs. Header rows repeated inside the table (page breaks in a Word or PDF export) are dropped. Cells
are kept verbatim, so names in them are found again by their text.
"""
from __future__ import annotations

import re

ROW = re.compile(r"^[ \t]*\|(.*)\|[ \t]*$")
RULE = re.compile(r"^[ \t]*\|[\s:|-]*\|[ \t]*$")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in ROW.match(line).group(1).split("|")]


def _sentence(cells: list[str], header: list[str] | None) -> str:
    if header is None:  # key: value table
        pairs = [f"{cells[0]}: {cells[1]}"] if len(cells) == 2 else [c for c in cells if c]
    else:
        pairs = [f"{h}: {c}" if h else c for h, c in zip(header, cells) if c]
    text = "; ".join(pairs)
    return text if text.endswith(".") else text + "."


def verbalize(text: str) -> str:
    """text with every markdown table's rows rewritten as sentences, one per line; other text unchanged."""
    lines, out, k = text.split("\n"), [], 0
    while k < len(lines):
        if not ROW.match(lines[k]):
            out.append(lines[k])
            k += 1
            continue
        block = []
        while k < len(lines) and ROW.match(lines[k]):
            block.append(lines[k])
            k += 1
        rules = [i for i, line in enumerate(block) if RULE.match(line)]
        if rules and rules[0] > 0:
            header, body = _cells(block[rules[0] - 1]), block[rules[0] + 1:]
        elif len(_cells(block[0])) == 2:
            header, body = None, block
        else:
            header, body = _cells(block[0]), block[1:]
        for line in body:
            if RULE.match(line) or (header is not None and _cells(line) == header):
                continue  # separator, or the header repeated after a page break
            out.append(_sentence(_cells(line), header))
    return "\n".join(out)
