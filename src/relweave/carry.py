"""Carry-forward: the entities of a document's earlier chunks, given to the generator before each new chunk so it can
write a later "the company" or "she" under the entity named earlier, reusing its number (formats.with_known).

One definition for training data and inference: known_lines() selects and renders the list, so the model is trained
on exactly what relweave shows it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from relweave.formats import entity_line

MAX_KNOWN = 40  # entities in the list; the document's first entity always stays, then the most recently mentioned
MAX_SURFACES = 4  # surfaces shown per entity, names first

PRONOUNS = {"it", "its", "he", "she", "his", "her", "they", "their", "them", "him"}


@dataclass
class Known:
    number: int
    type: str
    surfaces: list[str] = field(default_factory=list)
    last_seen: int = 0  # index of the last chunk that mentioned it


def _is_name(s: str) -> bool:
    t = s.strip()
    return bool(t) and t.lower() not in PRONOUNS and not t.lower().startswith(("the ", "a ", "an ")) and \
        any(c.isupper() for c in t)


def known_lines(known: list[Known], cap: int = MAX_KNOWN) -> str:
    """'E3 Org: Acme | Acme Group | the company' lines, at most cap entities: number 1 (the document's first entity)
    always, then by most recent mention; listed in number order."""
    if not known:
        return ""
    first = min(known, key=lambda k: k.number)
    rest = sorted((k for k in known if k is not first), key=lambda k: (-k.last_seen, k.number))[:max(0, cap - 1)]
    out = []
    for k in sorted([first] + rest, key=lambda k: k.number):
        names = [s for s in k.surfaces if _is_name(s)]
        others = [s for s in k.surfaces if s not in names]
        out.append(entity_line(k.number, k.type, list(dict.fromkeys(names + others))[:MAX_SURFACES]))
    return "\n".join(out)
