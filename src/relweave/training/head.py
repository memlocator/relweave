"""Joint training of the pair head and its LoRA adapter (relweave[train]), on a local GPU.

Starts from the generator's adapter (unfrozen) and optionally an earlier head. Each training example is one chunk:
prompt + entity lines + up to max_probes packed probes (all related pairs, then sampled unrelated ones). Up to half
the examples use the generator's own entity lines on the training chunks (own_results: its saved results; targets
through the entity alignment), so the head trains on what it sees at run time; the rest use gold entity lines, and
for those the same forward pass also gives the language-model loss on the entity lines, so generation stays intact.
loss = ATLOP loss on the probes + lm_weight * LM loss on the gold entity lines.

The output directory gets adapter/ (PEFT adapter and tokenizer), head.safetensors, head_config.json and
train_summary.json: a head directory in the published layout. A head for a new label set (a schema the earlier head
was not trained on) starts fresh on top of the continued adapter.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

DEFAULT_MODEL = "unsloth/qwen3-4b-unsloth-bnb-4bit"


def train_head(train_pool: list[Path], out: Path, generator: Path, model_id: str = DEFAULT_MODEL,
               own_results: list[Path] = (), base_head: Path | None = None, schema=None, n_examples: int = 2400,
               max_probes: int = 150, lm_weight: float = 0.5, grad_accum: int = 8, seed: int = 0,
               list_repeat: int = 0, upcast: bool = True, reserve_mb: int = 700, layers=(-1, -9),
               cut: float = -2.0, margin: float = 0.5) -> dict:
    """train_pool: labelled chunk files; generator: the generator directory (its adapter is the starting point and
    its settings say whether the schema is in the prompt); own_results: the generator's results on train_pool
    chunks (relweave.formats.save_results files); base_head: an earlier head directory or head.pt to continue;
    schema: name, Schema or path.py:NAME (default: the training chunks' own schema). upcast=False skips PEFT's
    fp32 copies of embeddings and lm_head (needed for a 4B model on 8 GB). cut, margin: union settings recorded in
    head_config.json. Returns the train summary."""
    import torch
    torch.manual_seed(seed)
    free, _ = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction((free - reserve_mb * 2**20) / torch.cuda.get_device_properties(0).total_memory)
    out, generator = Path(out), Path(generator)
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from relweave.extract import generator_settings
    from relweave.formats import FORMATS, chat_messages, generation_prompt, load_results
    from relweave.head import Head, adapter_dir, aligned_targets, at_loss, entity_lines, labels, load_head, pack_ids
    from relweave.schema import ExtractionOutput, load_schema, registered_schema
    from relweave.training.eval import align as _align

    out.mkdir(parents=True, exist_ok=True)
    adapter = adapter_dir(generator)
    tok = AutoTokenizer.from_pretrained(str(adapter))
    prequantized = "bnb-4bit" in str(model_id)  # a 4-bit checkpoint carries its own quantization config
    base = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", dtype=torch.bfloat16,
        **({} if prequantized else {"quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)}))
    if upcast:  # PEFT's standard k-bit preparation: also copies non-quantized weights (embeddings, lm_head) to fp32
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    else:  # the same checkpointing without the fp32 copies: an untied 4B vocabulary in fp32 alone is about 3 GB
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
    model.config.use_cache = False
    pool = [json.loads(line) for p in train_pool for line in Path(p).read_text().splitlines() if line.strip()]
    if schema is None:  # the training chunks' own schema when they share one, else business (registered only)
        names = {d.get("schema", "business") for d in pool}
        schema = registered_schema(names.pop() if len(names) == 1 else "business")
    else:
        schema = load_schema(schema)
    name = schema.name
    try:
        conditioned = generator_settings(generator).get("conditioned", False)
    except FileNotFoundError:
        conditioned = False
    labs = labels(schema)
    dim = model.config.hidden_size * len(layers)
    head = Head(dim, len(labs)).cuda()
    sd = None
    if base_head is not None and Path(base_head).is_dir():
        sd = load_head(base_head)[0].state_dict()
    elif base_head is not None and Path(base_head).exists():
        sd = torch.load(base_head, weights_only=True)
    if sd is not None and sd["net.4.weight"].shape[0] == len(labs) + 1:
        head.load_state_dict(sd)  # continue from a head with the same label set
    else:  # a new schema: a fresh head on top of the (continued) LoRA
        print(f"stage 1: new head for schema {name} ({len(labs)} labels)", flush=True)
    fmt = FORMATS["sentences"].with_schema(schema)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731

    # training examples: (prompt ids, entity-line ids, number of entity -> type, targets fn, lm?)
    rng = random.Random(seed)
    examples = []
    train = {d["chunk_id"]: d for d in pool}
    own = {r.chunk_id: r for p in own_results  # earlier files first: they fill the "own" half first
           for r in load_results(Path(p), {k: v["text"] for k, v in train.items()}, schema=schema)
           if r.output is not None}
    for cid, r in list(own.items())[:n_examples // 2]:  # the generator's own entity lines: half the examples
        c = train.get(cid)
        if c is None:
            continue
        gold = ExtractionOutput.model_validate(c["gold"], context={"schema": schema})
        ents, types = entity_lines(r.raw)
        align = _align(c["text"], gold, r.output)
        examples.append(("own", c["text"], ents, types, gold, align, {e.id: int(e.id[1:]) for e in r.output.entities}))
    rest = [k for k in train if k not in own]
    rng.shuffle(rest)
    for cid in rest[:max(0, n_examples - len(examples))]:  # gold entity lines (+ LM loss)
        c = train[cid]
        gold = ExtractionOutput.model_validate(c["gold"], context={"schema": schema})
        ents, types = entity_lines(fmt.encode(gold, c["text"]))
        examples.append(("gold", c["text"], ents, types, gold, {e.id: e.id for e in gold.entities},
                         {e.id: k + 1 for k, e in enumerate(gold.entities)}))
    if list_repeat:  # chunks whose gold has a list (one source or target, one type, 2+ partners): repeated
        from collections import Counter
        def has_list(gold):
            c = Counter((r.source, r.type) for r in gold.relations) + Counter((r.target, r.type) for r in gold.relations)
            return any(v >= 2 for v in c.values())
        examples += [e for e in examples if has_list(e[4])] * list_repeat
    rng.shuffle(examples)
    print(f"stage 1: {len(examples)} examples ({sum(e[0] == 'own' for e in examples)} with a model's own entity lines)", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": params, "lr": 1e-4}, {"params": head.parameters(), "lr": 5e-4}], weight_decay=0.0)
    total = math.ceil(len(examples) / grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, total // 10)) * max(0.05, 1 - s / total))
    model.train(); head.train()
    t0, step, log, skipped = time.time(), 0, [], 0
    for n, (kind, text, ents, types, gold, align, number_of) in enumerate(examples, 1):
        from itertools import combinations
        pairs = list(combinations(sorted(types), 2))
        if not pairs:
            continue
        y_all = aligned_targets(gold, align, number_of, pairs, schema=schema)
        pos = [k for k in range(len(pairs)) if y_all[k].sum() > 0]
        neg = [k for k in range(len(pairs)) if y_all[k].sum() == 0]
        rng.shuffle(neg)
        keep = sorted(pos[:max_probes] + neg[:max(0, max_probes - len(pos))])
        pairs = [pairs[k] for k in keep]
        y = y_all[keep].cuda()
        prompt = enc(generation_prompt(tok, chat_messages(text, "sentences", compact=True, schema=schema, conditioned=conditioned)))
        ent_ids = enc(ents)
        probes = [enc(f"P E{i} {types[i]} E{j} {types[j]}\n") for i, j in pairs]
        ids, posn, mask, last = pack_ids(prompt + ent_ids, probes)
        m = mask  # boolean, True = may attend: valid whatever dtype the attention runs in
        a, b = len(prompt), len(prompt) + len(ent_ids)
        # vocabulary logits only where the LM loss needs them (entity-line positions of gold examples)
        keep_logits = torch.arange(a - 1, b - 1, device="cuda") if kind == "gold" else 1
        try:  # one unusually long example must not end the run: skip it (a failed backward may leave part of
            # its gradient in the accumulation, a negligible error next to losing the run)
            fwd = model(input_ids=torch.tensor([ids], device="cuda"), position_ids=torch.tensor([posn], device="cuda"),
                        attention_mask=m[None, None].cuda(), output_hidden_states=True, logits_to_keep=keep_logits)
            hs = torch.cat([fwd.hidden_states[k][0, last] for k in layers], dim=-1).float()
            loss_rel = at_loss(head(hs), y)
            loss = loss_rel
            loss_lm = torch.zeros((), device="cuda")
            if kind == "gold":  # entity-line tokens of the gold answer: keep generation intact
                logits = fwd.logits[0].float()
                loss_lm = torch.nn.functional.cross_entropy(logits, torch.tensor(ent_ids, device="cuda"))
                loss = loss + lm_weight * loss_lm
            (loss / grad_accum).backward()
            log.append((loss_rel.item(), loss_lm.item()))
        except torch.OutOfMemoryError:
            skipped += 1
            print(f"skipped example {n}: out of memory ({len(ids)} tokens, {len(pairs)} probes)", flush=True)
            fwd = hs = loss = loss_rel = loss_lm = logits = None  # free the failed example's tensors
            torch.cuda.empty_cache()
        fwd = hs = None
        if n % grad_accum == 0 or n == len(examples):
            torch.nn.utils.clip_grad_norm_(params + list(head.parameters()), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 10 == 0 or step == total:
                recent = log[-grad_accum * 10:]
                print(f"progress: stage1 step {step}/{total} rel {sum(a for a, _ in recent) / len(recent):.3f} "
                      f"lm {sum(b for _, b in recent) / len(recent):.3f} elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    from relweave.head import head_config, save_head
    model.save_pretrained(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))
    save_head(head, out, head_config(head, schema, model.config.hidden_size, layers, compact_prompt=True,
                                     conditioned=conditioned, cut=cut, margin=margin))
    summary = {"format": "sentences", "compact_prompt": True, "examples": len(examples), "max_probes": max_probes,
               "lm_weight": lm_weight, "model": str(model_id), "train_pool": [str(p) for p in train_pool],
               "own_res": [str(p) for p in own_results], "base_adapter": str(adapter),
               "base_head": str(base_head) if base_head else None, "seed": seed, "list_repeat": list_repeat,
               "skipped_oom": skipped, "schema": name, "conditioned": conditioned, "labels": [list(x) for x in labs]}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"stage 1 trained ({skipped} examples skipped for memory)", flush=True)
    return summary
