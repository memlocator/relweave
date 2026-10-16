"""Entity tagger: BERT-style entity tagging on the generator's own hidden states, so every name in a list or table cell
is found in one pass, with no list to end early and no per-chunk entity cap.

A decoder token cannot see the rest of its own name, so the text is fed twice and the second copy is tagged: every
token there has read the whole text (echo embeddings, Springer et al. 2024, arXiv 2402.15449). A small MLP over the
frozen generator's hidden states at LAYERS predicts BIO tags per entity type; a tagged span is extended to whole words
at both ends (a name never starts or ends inside a word; rare surnames and compounds are split into several tokens).

In extraction (Extractor(entity_tagger=...)), the entities the tagger finds that the generator did not list are added
to the chunk's entity lines before the relation head scores every pair, so a name the generator skipped can still take
part in relations.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

TYPES = ("Person", "Org", "Place")  # the business tagger's types; a tagger for another schema takes that schema's


def tags_for(types) -> tuple[str, ...]:
    """BIO tags for entity types: O, then B- and I- for each type."""
    return ("O",) + tuple(f"{p}-{t}" for t in types for p in ("B", "I"))


TAGS = tags_for(TYPES)
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


def name_mentions(entity: dict) -> list[str]:
    """The mentions of a labelled entity the tagger learns to tag: its name and every mention sharing a word with it
    ("Kelvorn" for "Kelvorn Pumps GmbH", "the Riksdag" for "the Riksdag"), not pronouns or descriptions such as "it"
    or "the company". The tagger finds names the generator left out; resolving "it" stays with the generator."""
    words = _name_words(entity["name"])
    keep = [entity["name"]]
    for m in entity.get("mentions", []):
        if m != entity["name"] and words & _name_words(m):
            keep.append(m)
    return keep


def _name_words(text: str) -> set[str]:
    """Capitalised words of a name, as written, not articles: what a short form must share with the name to be one
    ("Kelvorn" for "Kelvorn Pumps GmbH"); "the company" shares nothing with "Acme Company" and "the" nothing with
    "The Lego Group"."""
    return {w for w in re.findall(r"\w+", text) if len(w) > 1 and w[0].isupper() and w not in ("The", "An")}


def encode(tok, text: str, gold_spans=None, tags=TAGS, echo: bool = True):
    """Token ids of text twice (echo) or once, the positions of the tagged copy's tokens with their character spans
    in text, and (with gold_spans) their BIO tag ids."""
    full = text + "\n\n" + text if echo else text
    off0 = len(text) + 2 if echo else 0
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    pos, char, tag_ids = [], [], []
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
            tag_ids.append(tags.index(tag))
    return enc["input_ids"], pos, char, tag_ids


def _word_end(text: str, b: int) -> int:
    while b < len(text) and (text[b].isalnum() or text[b] in "-'"):
        b += 1
    return b


def decode(text: str, char, pred, tags=TAGS) -> list[tuple[int, int, str]]:
    """Entity spans (start, end, type) from BIO tag ids over tokens with character spans; a span ending inside a word
    is extended to the word's end, and a span crossing a line break ends at it."""
    out, cur = [], None
    for (a, b), p in zip(char, pred):
        tag = tags[p]
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
        while a > 0 and text[a - 1].isalnum():  # a span starting inside a word starts at the word's start
            a -= 1
        b = _word_end(text, b)
        if "\n" in text[a:b]:  # a name does not cross a line break (a token like "\n-" carried the tag along)
            b = a + text[a:b].index("\n")
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
        self.reads = self.config.get("reads", "generator")  # whose hidden states it was trained on: generator or head
        self.echo = self.config.get("echo", True)  # text fed twice, the second copy tagged
        self.lookahead = int(self.config.get("lookahead", 0))  # or: one copy, each token read with the next k states
        self.tags = tuple(self.config.get("tags", TAGS))
        self.state = torch.load(d / "tagger.pt", map_location="cpu", weights_only=True)
        self.net = None

    def dim(self, hidden_size: int) -> int:
        return hidden_size * len(self.layers) * (1 + getattr(self, "lookahead", 0))

    def _features(self, model, ids, pos):
        """Per tagged token: its hidden states at the layers read, and with lookahead k also the states of the next k
        tokens (which have read it; a causal model's own state has not seen the rest of the name)."""
        from relweave.states import layer_states
        states = layer_states(model, self.layers, input_ids=torch.tensor([ids], device=model.device))
        k = getattr(self, "lookahead", 0)
        idx = torch.tensor(pos, device=model.device)
        parts = []
        for h in states:
            h = h[0]
            for j in range(k + 1):
                parts.append(h[torch.clamp(idx + j, max=h.shape[0] - 1)].float())
        return torch.cat(parts, dim=-1)

    def tag(self, model, tok, text: str, window_words: int | None = None) -> list[tuple[str, str, int, int]]:
        """(surface, type, start, end) for every entity span in text, any length (line-aligned windows of
        window_words, default the config's; 0: the whole text in one pass)."""
        if self.net is None:
            self.net = TaggerNet(self.dim(model.config.hidden_size), len(self.tags)).to(model.device)
            self.net.load_state_dict(self.state)
            self.net.eval()
        found = []
        size = self.config.get("window_words", WINDOW_WORDS) if window_words is None else window_words
        for off, w in (windows(text, size) if size else [(0, text)]):
            ids, pos, char, _ = encode(tok, w, echo=self.echo)
            if not pos:
                continue
            with torch.no_grad():
                pred = self.net(self._features(model, ids, pos)).argmax(-1).tolist()
            found += [(w[a:b], t, off + a, off + b) for a, b, t in decode(w, char, pred, self.tags)]
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
