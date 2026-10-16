"""Entity tagger: BERT-style entity tagging on the generator's own hidden states, so every name in a list or table cell
is found in one pass, with no list to end early and no per-chunk entity cap.

A decoder token cannot see the rest of its own name, so the text is fed twice and the second copy is tagged: every
token there has read the whole text (echo embeddings, Springer et al. 2024, arXiv 2402.15449). A small MLP over the
frozen generator's hidden states at LAYERS predicts BIO tags per entity type; a tagged span is extended to the end of
the word it stops in (a name never ends inside a word; rare surnames are split into several tokens).

In extraction (Extractor(entity_tagger=...)), the entities the tagger finds that the generator did not list are added
to the chunk's entity lines before the relation head scores every pair, so a name the generator skipped can still take
part in relations.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

TYPES = ("Person", "Org", "Place")
TAGS = ("O",) + tuple(f"{p}-{t}" for t in TYPES for p in ("B", "I"))
WINDOW_WORDS = 300


class TaggerNet(torch.nn.Module):
    def __init__(self, dim: int, n_tags: int = len(TAGS)):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.LayerNorm(dim), torch.nn.Linear(dim, 512), torch.nn.GELU(),
                                       torch.nn.Dropout(0.1), torch.nn.Linear(512, n_tags))

    def forward(self, x):
        return self.net(x)


def spans(text: str, mentions) -> list[tuple[int, int, str]]:
    """(start, end, type) for every whole-word occurrence of every (mention, type), longest first, no overlaps."""
    out, taken = [], []
    for m, t in sorted(mentions, key=lambda x: -len(x[0])):
        for x in re.finditer(r"(?<!\w)" + re.escape(m) + r"(?!\w)", text):
            a, b = x.span()
            if not any(a < e and s < b for s, e in taken):
                out.append((a, b, t))
                taken.append((a, b))
    return sorted(out)


def encode(tok, text: str, gold_spans=None):
    """Token ids of text twice, the positions of the second copy's tokens with their character spans in text, and
    (with gold_spans) their BIO tag ids."""
    full = text + "\n\n" + text
    off0 = len(text) + 2
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    pos, char, tags = [], [], []
    for i, (a, b) in enumerate(enc["offset_mapping"]):
        if a < off0 or b <= a:
            continue
        a, b = a - off0, b - off0
        pos.append(i)
        char.append((a, b))
        if gold_spans is not None:
            tag = "O"
            for s, e, t in gold_spans:
                if a < e and s < b:
                    first = a + (len(text[a:b]) - len(text[a:b].lstrip())) <= s
                    tag = f"{'B' if first else 'I'}-{t}"
                    break
            tags.append(TAGS.index(tag))
    return enc["input_ids"], pos, char, tags


def _word_end(text: str, b: int) -> int:
    while b < len(text) and (text[b].isalnum() or text[b] in "-'"):
        b += 1
    return b


def decode(text: str, char, pred) -> list[tuple[int, int, str]]:
    """Entity spans (start, end, type) from BIO tag ids over tokens with character spans; a span ending inside a word
    is extended to the word's end."""
    out, cur = [], None
    for (a, b), p in zip(char, pred):
        tag = TAGS[p]
        if tag.startswith("B-") or (tag.startswith("I-") and (cur is None or cur[2] != tag[2:])):
            if cur:
                out.append(cur)
            cur = [a, b, tag[2:]]
        elif tag.startswith("I-"):
            cur[1] = b
        else:
            if cur:
                out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    res = []
    for a, b, t in out:
        a = a + (len(text[a:b]) - len(text[a:b].lstrip()))
        b = _word_end(text, b)
        while b > a and text[b - 1] in ",.;:!?\"'":  # punctuation a token carried along is not part of the name
            b -= 1
        if text[a:b].strip():
            res.append((a, b, t))
    return res


def windows(text: str, words: int = WINDOW_WORDS) -> list[tuple[int, str]]:
    """(offset, window) line-aligned windows of about `words` words covering the text."""
    out, start, n, pos = [], 0, 0, 0
    for line in text.split("\n"):
        k = len(line.split())
        if n and n + k > words:
            out.append((start, text[start:pos - 1]))
            start, n = pos, 0
        n += k
        pos += len(line) + 1
    if start < len(text):
        out.append((start, text[start:]))
    return out


class EntityTagger:
    """A trained tagger (tagger.pt + tagger_config.json) applied to a loaded model and tokenizer."""

    def __init__(self, tagger_dir: str | Path):
        d = Path(tagger_dir)
        self.config = json.loads((d / "tagger_config.json").read_text())
        self.layers = list(self.config["layers"])
        self.tags = tuple(self.config.get("tags", TAGS))
        self.state = torch.load(d / "tagger.pt", map_location="cpu", weights_only=True)
        self.net = None

    def _features(self, model, ids, pos):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids], device=model.device), output_hidden_states=True)
        return torch.cat([out.hidden_states[k][0, pos].float() for k in self.layers], dim=-1)

    def tag(self, model, tok, text: str) -> list[tuple[str, str, int, int]]:
        """(surface, type, start, end) for every entity span in text, any length (line-aligned windows)."""
        if self.net is None:
            self.net = TaggerNet(model.config.hidden_size * len(self.layers), len(self.tags)).to(model.device)
            self.net.load_state_dict(self.state)
            self.net.eval()
        found = []
        for off, w in windows(text, self.config.get("window_words", WINDOW_WORDS)):
            ids, pos, char, _ = encode(tok, w)
            if not pos:
                continue
            with torch.no_grad():
                pred = self.net(self._features(model, ids, pos)).argmax(-1).tolist()
            found += [(w[a:b], t, off + a, off + b) for a, b, t in decode(w, char, pred)]
        return found

    def augment(self, model, tok, text: str, result):
        """Add the tagged entities the generator did not list to its result (its entity lines and its output), so the
        relation head scores their pairs too. Returns the result unchanged when nothing is missing."""
        from relweave.schema.output import Entity
        out = result.output
        known = {m.lower() for e in out.entities for m in [e.name, *e.mentions]}
        extra, seen = [], set()
        for surface, typ, _, _ in self.tag(model, tok, text):
            k = surface.lower()
            if k in known or k in seen:
                continue
            seen.add(k)
            extra.append((surface, typ))
        if not extra:
            return result
        n = max([int(e.id[1:]) for e in out.entities if e.id[1:].isdigit()], default=0)
        ents = list(out.entities)
        lines = []
        for k, (surface, typ) in enumerate(extra, n + 1):
            ents.append(Entity.model_construct(id=f"e{k}", type=typ, name=surface, attributes={}, mentions=[surface]))
            lines.append(f"E{k} {typ}: {surface}")
        result.output = out.model_copy(update={"entities": ents})
        result.raw = result.raw.rstrip("\n") + "\n" + "\n".join(lines) + "\n"
        return result
