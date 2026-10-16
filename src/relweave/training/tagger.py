"""Train an entity tagger (relweave.tagger) on labelled chunks: a small classifier over the hidden states of a frozen
model with an adapter (by default your trained pair head's, so the tagger runs with the head and also works when the
generator is served by vLLM). Only the classifier trains.

Labels: every whole-word occurrence of each entity's name mentions (relweave.tagger.name_mentions: the name and
mentions sharing a word with it, not pronouns or descriptions), BIO-tagged by entity type. Types: the schema's entity
types, or a subset.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from relweave.tagger import WINDOW_WORDS, EntityTagger, TaggerNet, encode, name_mentions, spans, tags_for


def _log(echo, *a):
    echo(time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a))


def train_tagger(chunks: list[str | Path], out: str | Path, adapter: str | Path, base: str, types: list[str],
                 reads: str = "head", epochs: int = 2, layers=(-1, -14), lr: float = 1e-3, seed: int = 0,
                 max_tokens: int = 4096, echo=print, double: bool = False, lookahead: int = 0) -> dict:
    """Train on labelled chunk files (records with "text" and "gold" entities) and write OUT/tagger.pt and
    OUT/tagger_config.json. Each token is tagged from its own states, the text read once (a causal state has read the
    lead sentence and the name's earlier words, which is what a tag needs). double: feed the text twice and tag the
    second copy, as taggers before relweave 0.3.1 did (no better on lists, half the speed); lookahead: also read the
    next k tokens' states (worse: a line break picked up the next name). `adapter`: the directory whose adapter's states are read (the head's, or the generator's
    with reads="generator"). Returns the config."""
    from relweave.head import adapter_dir, load_model
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tags = tags_for(types)
    tok, model = load_model(base, adapter_dir(adapter))
    for p in model.parameters():
        p.requires_grad_(False)
    rows = [json.loads(x) for f in chunks for x in open(f) if x.strip()]
    data = []
    for r in rows:
        ents = (r.get("gold") or {}).get("entities", [])
        ments = [(m, e["type"]) for e in ents if e["type"] in types for m in name_mentions(e)]
        ids, pos, _, tag_ids = encode(tok, r["text"], spans(r["text"], ments), tags, echo=double)
        if pos and len(ids) < max_tokens:
            data.append((ids, pos, tag_ids))
    layers = list(layers)
    config = {"layers": layers, "tags": list(tags), "types": list(types), "base": base, "reads": reads, "echo": double,
              "lookahead": lookahead,
              "window_words": WINDOW_WORDS, "examples": len(data)}
    _log(echo, f"{len(data)} training chunks, {len(types)} entity types, layers {layers}, reading the {reads}'s states")
    tagger = EntityTagger.__new__(EntityTagger)
    tagger.config, tagger.layers, tagger.tags, tagger.reads = config, layers, tags, reads
    tagger.echo, tagger.lookahead = double, lookahead
    net = TaggerNet(tagger.dim(model.config.hidden_size), len(tags)).to(model.device)
    tagger.net = net
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    weight = torch.ones(len(tags), device=model.device)
    weight[0] = 0.3  # most tokens are outside any entity
    rng = random.Random(seed)
    for ep in range(int(epochs)):
        net.train()
        rng.shuffle(data)
        tot = 0.0
        for k, (ids, pos, tag_ids) in enumerate(data, 1):
            loss = torch.nn.functional.cross_entropy(net(tagger._features(model, ids, pos)),
                                                     torch.tensor(tag_ids, device=model.device), weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            if k % 500 == 0:
                _log(echo, f"epoch {ep + 1} step {k}/{len(data)} loss {tot / 500:.4f}")
                tot = 0.0
        net.eval()
        torch.save(net.state_dict(), out / "tagger.pt")
        json.dump({**config, "epochs": ep + 1}, open(out / "tagger_config.json", "w"), indent=1)
        _log(echo, f"epoch {ep + 1} done, saved to {out}")
    return {**config, "epochs": int(epochs)}
