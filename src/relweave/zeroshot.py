"""Zero-shot relations: relation types defined at run time (a Schema whose relation classes the trained head does not
know), scored by the yes/no adapter chrullis/relweave-4b-zeroshot.

For every pair of entities the generator found in a chunk and every schema relation type whose endpoint types fit,
the adapter reads the chunk, its entity lines and the question "Source: X | Target: Y | Relation: <definition>" and
the probability of " yes" is the relation's score; it is kept when the score is above the type's threshold. All
questions of a chunk share forward passes (packed; each question attends to the shared prefix and itself only).

Density calibration (default): the more questions a chunk asks, the more chances for a false yes, so the best raw
threshold rises with entity density (about 0.7 on short passages, about 0.9 on dense text). Each score is therefore
adjusted for the number n of entity pairs its type is asked on in the chunk, logit(p) - DENSITY_ALPHA * ln(n /
DENSITY_REF), and compared with one threshold. Counting per type keeps a type's scores independent of how many other
types the schema has. Fitted on the synthetic benchmark (short and dense) it gave 0.839 and 0.734 (best fixed threshold
per set 0.829 and 0.744) and 0.467 on held-out Re-DocRED (best fixed 0.512, raw 0.8 0.474). Thresholds:
one number for all types, or {type: threshold}, on adjusted scores; `calibrate` picks one per type from a few labelled
examples (scores taken from the graph, which holds the adjusted values).
Entities still come from the trained generator, so the schema's entity types must be among the generator's.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_WEIGHTS = "chrullis/relweave-4b-zeroshot"
DEFAULT_THRESHOLD = 0.6  # on density-adjusted scores
DENSITY_ALPHA = 0.5
DENSITY_REF = 5  # pairs a type is asked on per chunk at which the adjustment is zero
INSTRUCTION = ("For each question below, answer yes if the text states or clearly implies that the relation holds "
               "from the source to the target, otherwise no.")
GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99)


def question(source: str, target: str, definition: str) -> str:
    return f"Q: Source: {source} | Target: {target} | Relation: {definition} A:"


def entity_block(entities: list[tuple[str, list[str]]]) -> str:
    """'E1 Org: Acme | the company' lines for (type, mentions) in order."""
    return "".join(f"E{k} {t}: " + " | ".join(m) + "\n" for k, (t, m) in enumerate(entities, 1))


def pack(prefix: list[int], questions: list[list[int]]):
    """One sequence: the prefix, then every question; each question attends to the prefix and itself only, its
    positions continuing from the prefix. Returns ids, positions, a boolean mask (True = may attend) and each
    question's last index."""
    import torch
    ids, pos, spans = list(prefix), list(range(len(prefix))), []
    for q in questions:
        start = len(ids)
        ids += q
        pos += list(range(len(prefix), len(prefix) + len(q)))
        spans.append((start, len(ids)))
    n = len(ids)
    mask = torch.zeros(n, n, dtype=torch.bool)
    mask[:len(prefix), :len(prefix)] = torch.tril(torch.ones(len(prefix), len(prefix), dtype=torch.bool))
    for s, e in spans:
        mask[s:e, :len(prefix)] = True
        mask[s:e, s:e] = torch.tril(torch.ones(e - s, e - s, dtype=torch.bool))
    return ids, pos, mask, [e - 1 for _, e in spans]


def candidates(schema, entities: list[tuple[str, str]]) -> list[tuple[str, int, int]]:
    """(relation type, source index, target index) for every ordered pair of distinct entities (index into
    `entities`, a list of (name, type)) and every relation type of `schema` whose endpoint types fit. A symmetric
    type is asked once per unordered pair."""
    out = []
    for edge in schema.edge_names():
        src, tgt, sym = schema.sources(edge), schema.targets(edge), schema.symmetric(edge)
        for i, (_, ti) in enumerate(entities):
            for j, (_, tj) in enumerate(entities):
                if i == j or (sym and j < i):
                    continue
                if ti in src and tj in tgt:
                    out.append((edge, i, j))
    return out


def adjust(p: float, n_questions: int, alpha: float = DENSITY_ALPHA, ref: float = DENSITY_REF) -> float:
    """p(yes) adjusted for the number of questions its relation type was asked in the chunk (one per fitting entity
    pair): logit(p) - alpha * ln(n / ref), as a probability."""
    import math
    p = min(max(p, 1e-6), 1 - 1e-6)
    x = math.log(p / (1 - p)) - alpha * math.log(max(n_questions, 1) / ref)
    return 1 / (1 + math.exp(-x))


def threshold_for(thresholds: float | dict, edge: str) -> float:
    if isinstance(thresholds, dict):
        return thresholds.get(edge, thresholds.get("*", DEFAULT_THRESHOLD))
    return float(thresholds)


def calibrate(scored: list[tuple[str, float, bool]], grid=GRID, fallback: float = DEFAULT_THRESHOLD,
              min_positive: int = 3) -> dict[str, float]:
    """Per-type thresholds from labelled examples: `scored` is (relation type, score, is a true relation) for every
    candidate in a few labelled chunks. Each type gets the grid value with the best F1 on its examples; types with
    fewer than `min_positive` true relations get the threshold that is best over all types pooled ("*" key), or
    `fallback` when there is nothing to pool."""
    def best(rows):
        def f1(t):
            tp = sum(1 for _, p, y in rows if p > t and y)
            fp = sum(1 for _, p, y in rows if p > t and not y)
            fn = sum(1 for _, p, y in rows if p <= t and y)
            return 2 * tp / max(2 * tp + fp + fn, 1)
        return max(grid, key=f1)
    pooled = best(scored) if any(y for _, _, y in scored) else fallback
    out = {"*": pooled}
    for edge in {e for e, _, _ in scored}:
        rows = [r for r in scored if r[0] == edge]
        out[edge] = best(rows) if sum(y for _, _, y in rows) >= min_positive else pooled
    return out


class ZeroShot:
    """The yes/no adapter on its 4-bit base. score_chunk(text, entities, schema) -> [(type, i, j, p)]."""

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, base: str | None = None, group: int = 60):
        self.dir = Path(weights) if Path(weights).is_dir() else resolve_weights_any(weights)
        cfg = json.loads((self.dir / "adapter_config.json").read_text())
        self.base = base or cfg["base_model_name_or_path"]
        self.group = group
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(self.dir))
        base = AutoModelForCausalLM.from_pretrained(self.base, device_map="cuda", dtype=torch.bfloat16)
        self._model = PeftModel.from_pretrained(base, str(self.dir)).eval()
        enc = self.enc
        self.yes, self.no = enc(" yes")[-1], enc(" no")[-1]

    def unload(self) -> None:
        self._model = None
        from relweave.extract import _free_gpu
        _free_gpu()

    def enc(self, s: str) -> list[int]:
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def probabilities(self, text: str, entities: list[tuple[str, list[str]]], questions: list[str]) -> list[float]:
        """p(yes) per question about `text` with the entity lines of `entities` ((type, mentions) in order)."""
        import torch
        self.load()
        pre = self.enc(self.tok.apply_chat_template(
            [{"role": "user", "content": f"{text}\n\nEntities:\n{entity_block(entities)}\n{INSTRUCTION}"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False))
        qs = [self.enc(q) for q in questions]
        out = []
        for k in range(0, len(qs), self.group):
            ids, pos, mask, last = pack(pre, qs[k:k + self.group])
            with torch.no_grad():
                o = self._model(input_ids=torch.tensor([ids], device="cuda"),
                                position_ids=torch.tensor([pos], device="cuda"),
                                attention_mask=mask[None, None].cuda(), logits_to_keep=torch.tensor(last, device="cuda"))
            out += o.logits[0][:, [self.yes, self.no]].float().softmax(-1)[:, 0].cpu().tolist()
        return out

    def score_chunk(self, text: str, entities: list[tuple[str, str, list[str]]], schema) -> list[tuple[str, int, int, float]]:
        """entities: (name, type, mentions). Every fitting (type, i, j) with its raw p(yes); adjust() it with the
        number of questions of its type before thresholding."""
        cands = candidates(schema, [(n, t) for n, t, _ in entities])
        if not cands:
            return []
        qs = [question(entities[i][0], entities[j][0], schema.definition(edge)) for edge, i, j in cands]
        ps = self.probabilities(text, [(t, m) for _, t, m in entities], qs)
        return [(edge, i, j, p) for (edge, i, j), p in zip(cands, ps)]


def resolve_weights_any(spec: str | Path) -> Path:
    """A Hub repository with the adapter files at its root (unlike the base models' generator/ and head/ layout)."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(spec)))
