"""The pair head: relation classification on the generator's own hidden states.

Relations are decided per entity pair by classification instead of being generated line by line. After
the prompt and the entity lines, one short probe per unordered entity pair is appended:

    P E3 Org E7 Place

All probes are packed into one forward pass: a 4D attention mask lets every probe token see the prompt,
the entity lines and the earlier tokens of its own probe, but no other probe, and every probe starts at
the same position id (PL-Marker's packed levitated markers). The hidden state at a probe's last token is
the pair's representation; a small head scores it against every (relation type, direction) label plus a
learned threshold class (ATLOP's adaptive-threshold loss), so NONE is "no label above the threshold".

The head is trained jointly with a LoRA adapter on the base model (relweave.training.head); at run time it reads
the hidden states of layers LAYERS at the probes' last tokens. On disk a head directory holds the adapter (PEFT
layout, in the directory itself or in adapter/), head.safetensors and head_config.json (save_head, load_head); a
pickled head.pt with train_summary.json is read as a fallback.
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

import torch

from relweave.schema import BUSINESS, Schema

ENTITY_LINE = re.compile(r"^E(\d+) (\w+): ")
LAYERS = (-1, -9)  # hidden-state layers the head reads, concatenated


def labels(schema=None) -> list[tuple[str, str]]:
    """(type, direction) classes: '>' first entity of the probe is the source, '<' the second; symmetric
    types only '>' (direction carries no information)."""
    schema = schema or BUSINESS  # the label set follows the schema; business by default
    return [(e, d) for e in schema.edge_names() for d in ((">",) if schema.symmetric(e) else (">", "<"))]


def entity_lines(answer: str) -> tuple[str, dict[int, str]]:
    """The entity-line part of a sentences-format answer and entity number -> type."""
    lines = [line for line in answer.split("\n") if ENTITY_LINE.match(line)]
    types = {int(ENTITY_LINE.match(line).group(1)): ENTITY_LINE.match(line).group(2) for line in lines}
    return "\n".join(lines) + "\n", types


def pack(tok, prefix: str, types: dict[int, str], max_pairs: int = 400, pairs: list[tuple[int, int]] | None = None):
    """Token ids, position ids, a boolean 4D mask (True = may attend) and, per probe, its pair and the index of
    its last token. Pairs (i < j) in entity-number order, at most max_pairs, unless `pairs` gives them."""
    pre = tok(prefix, add_special_tokens=False)["input_ids"]
    pairs = list(pairs) if pairs is not None else list(combinations(sorted(types), 2))[:max_pairs]
    ids, pos, spans = list(pre), list(range(len(pre))), []
    for i, j in pairs:
        p = tok(f"P E{i} {types[i]} E{j} {types[j]}\n", add_special_tokens=False)["input_ids"]
        start = len(ids)
        ids += p
        pos += list(range(len(pre), len(pre) + len(p)))
        spans.append((start, len(ids)))
    n = len(ids)
    mask = torch.zeros(n, n, dtype=torch.bool)
    mask[:len(pre), :len(pre)] = torch.tril(torch.ones(len(pre), len(pre), dtype=torch.bool))
    for s, e in spans:
        mask[s:e, :len(pre)] = True
        mask[s:e, s:e] = torch.tril(torch.ones(e - s, e - s, dtype=torch.bool))
    last = [e - 1 for _, e in spans]
    return ids, pos, mask, pairs, last


@torch.no_grad()
def probe_states(model, tok, prefix: str, types: dict[int, str], layers=(-1,), max_pairs: int = 400,
                 pairs: list[tuple[int, int]] | None = None):
    """(pairs, tensor [n_pairs, hidden * len(layers)]) of the frozen model's probe states (`pairs`: probe exactly
    these pairs, e.g. one group of a chunk too large for one pass)."""
    if len(types) < 2:
        return [], None
    ids, pos, mask, pairs, last = pack(tok, prefix, types, max_pairs, pairs)
    dev = model.device
    # additive float mask (0 where allowed, very negative elsewhere) in the model's dtype
    big = torch.finfo(torch.bfloat16).min
    m = torch.zeros(mask.shape, dtype=torch.bfloat16)
    m.masked_fill_(~mask, big)
    out = model(input_ids=torch.tensor([ids], device=dev), position_ids=torch.tensor([pos], device=dev),
                attention_mask=m[None, None].to(dev), output_hidden_states=True,
                logits_to_keep=1)  # only hidden states are needed: no vocabulary logits over the probes
    hs = torch.cat([out.hidden_states[k][0, last] for k in layers], dim=-1).float().cpu()
    return pairs, hs


def gold_targets(gold, types_by_id: dict[str, int], pairs: list[tuple[int, int]], schema=None) -> torch.Tensor:
    """Multi-hot [n_pairs, n_labels] from gold relations; entity numbers are 1-based positions in gold.entities.
    Labels and symmetric types follow schema (business by default)."""
    schema = schema or BUSINESS
    idx = {lab: k for k, lab in enumerate(labels(schema))}
    y = torch.zeros(len(pairs), len(idx))
    where = {p: k for k, p in enumerate(pairs)}
    for r in gold.relations:
        a, b = types_by_id[r.source], types_by_id[r.target]
        if a == b:
            continue
        i, j = min(a, b), max(a, b)
        if (i, j) not in where:
            continue
        d = ">" if (r.type in schema.edge_names() and schema.symmetric(r.type)) or a == i else "<"
        if (r.type, d) in idx:
            y[where[(i, j)], idx[(r.type, d)]] = 1
    return y


class Head(torch.nn.Module):
    """MLP over a probe state -> one logit per (type, direction) label plus a threshold logit (last)."""

    def __init__(self, dim: int, n_labels: int, hidden: int = 512):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.LayerNorm(dim), torch.nn.Linear(dim, hidden), torch.nn.GELU(),
                                       torch.nn.Dropout(0.1), torch.nn.Linear(hidden, n_labels + 1))

    def forward(self, x):
        return self.net(x)


def at_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ATLOP adaptive-threshold loss (Zhou et al. 2021). logits: [n, L + 1], the last column the threshold
    class; y: [n, L] multi-hot. Part 1: each positive label above the threshold (softmax over positives and
    the threshold); part 2: the threshold above every negative label (softmax over negatives and threshold)."""
    th = torch.zeros_like(logits[:, :1])
    labels = torch.cat([y, th], 1)                               # gold labels, threshold column 0
    th_label = torch.cat([torch.zeros_like(y), torch.ones_like(th)], 1)
    p_mask = labels + th_label                                   # positives and the threshold
    n_mask = 1 - labels                                          # negatives and the threshold
    loss1 = -(torch.log_softmax(logits - (1 - p_mask) * 1e30, -1) * labels).sum(1)
    loss2 = -(torch.log_softmax(logits - (1 - n_mask) * 1e30, -1) * th_label).sum(1)
    return (loss1 + loss2).mean()


def predict(logits: torch.Tensor, margin: float = 0.0) -> list[list[int]]:
    """Per pair, the label indices whose logit exceeds the threshold logit by more than margin."""
    th = logits[:, -1:]
    keep = logits[:, :-1] > th + margin
    return [torch.nonzero(row).flatten().tolist() for row in keep]


# --------------------------------------------------------------------------
# Stage 1: joint training of the LoRA and the head
# --------------------------------------------------------------------------


def pack_ids(pre: list[int], probes: list[list[int]]):
    """Like pack(), for ready token ids: prefix ids and one id list per probe."""
    ids, pos, spans = list(pre), list(range(len(pre))), []
    for p in probes:
        start = len(ids)
        ids += p
        pos += list(range(len(pre), len(pre) + len(p)))
        spans.append((start, len(ids)))
    n = len(ids)
    mask = torch.zeros(n, n, dtype=torch.bool)
    mask[:len(pre), :len(pre)] = torch.tril(torch.ones(len(pre), len(pre), dtype=torch.bool))
    for s, e in spans:
        mask[s:e, :len(pre)] = True
        mask[s:e, s:e] = torch.tril(torch.ones(e - s, e - s, dtype=torch.bool))
    return ids, pos, mask, [e - 1 for _, e in spans]


def aligned_targets(gold, align: dict, number_of: dict[str, int], pairs: list[tuple[int, int]], schema=None) -> torch.Tensor:
    """Multi-hot targets for probes over the MODEL's entities: a probe (i, j) gets the gold relations between
    the gold entities that model entities i and j align to (relweave.training.eval.align); unaligned entities get none.
    Labels and symmetric types follow schema (business by default)."""
    schema = schema or BUSINESS
    idx = {lab: k for k, lab in enumerate(labels(schema))}
    gold_of = {number_of[m]: g for m, g in align.items() if g is not None and m in number_of}
    rels: dict[frozenset, list] = {}
    for r in gold.relations:
        rels.setdefault(frozenset((r.source, r.target)), []).append(r)
    y = torch.zeros(len(pairs), len(idx))
    for k, (i, j) in enumerate(pairs):
        gi, gj = gold_of.get(i), gold_of.get(j)
        if not gi or not gj or gi == gj:
            continue
        for r in rels.get(frozenset((gi, gj)), []):
            d = ">" if (r.type in schema.edge_names() and schema.symmetric(r.type)) or r.source == gi else "<"
            if (r.type, d) in idx:
                y[k, idx[(r.type, d)]] = 1
    return y


# --------------------------------------------------------------------------
# Loading and saving
# --------------------------------------------------------------------------


def adapter_dir(run_dir: str | Path) -> Path:
    """The PEFT adapter of a generator or head directory: the directory itself (exported layout, adapter_config.json
    inside) or its adapter/ subdirectory (a training run)."""
    d = Path(run_dir)
    return d if (d / "adapter_config.json").exists() else d / "adapter"


def load_model(base: str, adapter: str | Path):
    """(tokenizer, model): the base model in bf16 compute with the adapter, eval mode, on CUDA; the tokenizer is
    the one saved with the adapter."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(adapter))
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
    return tok, PeftModel.from_pretrained(model, str(adapter)).eval()


def head_config(head: Head, schema: Schema, hidden_size: int, layers=LAYERS, compact_prompt: bool = True,
                conditioned: bool = False, cut: float = -2.0, margin: float = 0.5) -> dict:
    """What head_config.json records: the schema and label order the head was trained on, the base model's hidden
    size and the layers read, the probe prompt settings, and the union settings to use with it."""
    from relweave.schema import MAX_RELATIONS_PER_CHUNK
    return {"schema": schema.name, "hidden_size": hidden_size, "layers": list(layers),
            "probe_hidden": head.net[1].out_features, "labels": [list(x) for x in labels(schema)],
            "threshold_logit": "last",
            "prompt": {"format": "sentences", "compact_prompt": compact_prompt, "conditioned": conditioned},
            "union": {"cut": cut, "margin": margin, "max_relations": MAX_RELATIONS_PER_CHUNK}}


def save_head(head: Head, out_dir: str | Path, config: dict) -> None:
    """head.safetensors and head_config.json in out_dir."""
    from safetensors.torch import save_file
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, str(out / "head.safetensors"))
    (out / "head_config.json").write_text(json.dumps(config, indent=1) + "\n")


def load_head(head_dir: str | Path) -> tuple[Head, dict]:
    """(Head in eval mode, config). Reads head.safetensors + head_config.json; a directory with only head.pt (an
    experiment run) gets its config from train_summary.json (schema, labels; layers LAYERS)."""
    d = Path(head_dir)
    if (d / "head.safetensors").exists():
        from safetensors.torch import load_file
        config = json.loads((d / "head_config.json").read_text())
        sd = load_file(str(d / "head.safetensors"))
    elif (d / "head.pt").exists():
        summary = json.loads((d / "train_summary.json").read_text()) if (d / "train_summary.json").exists() else {}
        sd = torch.load(d / "head.pt", weights_only=True)
        config = {"schema": summary.get("schema", BUSINESS.name), "layers": list(LAYERS),
                  "prompt": {"format": "sentences", "compact_prompt": summary.get("compact_prompt", True),
                             "conditioned": summary.get("conditioned", False)}}
        if "labels" in summary:
            config["labels"] = summary["labels"]
    else:
        raise FileNotFoundError(f"{d}: no head.safetensors (with head_config.json) or head.pt")
    hidden = sd["net.1.weight"].shape[0]
    dim = sd["net.1.weight"].shape[1]
    n_labels = sd["net.4.weight"].shape[0] - 1
    if "labels" in config and len(config["labels"]) != n_labels:
        raise ValueError(f"{d}: head has {n_labels} labels, its config lists {len(config['labels'])}")
    if "hidden_size" in config and dim != len(config.get("layers", LAYERS)) * config["hidden_size"]:
        raise ValueError(f"{d}: head input {dim} is not hidden_size x layers of its config")
    head = Head(dim, n_labels, hidden)
    head.load_state_dict(sd)
    return head.eval(), config
