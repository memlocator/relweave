"""QLoRA fine-tuning of the generator (Qwen3 + LoRA) on labelled chunks, via Unsloth and TRL (relweave[train]).

The output directory holds adapter/ (PEFT adapter, tokenizer and relweave_config.json, so adapter/ is a generator
directory in the published layout) and train_summary.json."""

from __future__ import annotations

import json
from pathlib import Path

from relweave.formats import FORMATS, STUDENT_FORMAT, messages, render_chat
from relweave.training.variants import training_view

DEFAULT_MODEL = "unsloth/qwen3-4b-unsloth-bnb-4bit"


def build_examples(chunks_path: Path, fmt: str = STUDENT_FORMAT, compact: bool = False, conditioned: bool = False,
                   variants: bool = False, held_out: dict | None = None, seed: int = 0) -> list[dict]:
    """Chat-format examples: identical prompt to inference, gold encoded in the format as the answer.
    Each chunk is shown under its own schema (record key "schema", default business). conditioned: the
    rendered schema is in the system prompt. variants: a random variant of the schema per example (gold
    mapped to it). held_out ({"relations", "entities", "schemas"}): those types are removed from every
    schema and gold, and chunks of the held-out schemas are skipped. Each example records its root schema
    name under "schema"."""
    import random
    rng = random.Random(seed)
    examples = []
    for line in chunks_path.read_text().splitlines():
        c = json.loads(line)
        if "messages" in c:  # a ready-made example: used as it is
            examples.append({"messages": c["messages"], "chunk_id": c["chunk_id"]})
            continue
        view = training_view(c, rng, variants, held_out)
        if view is None:
            continue
        gold, schema, root = view
        f = FORMATS[fmt].with_schema(schema)
        msgs = messages(f, c["text"], compact=compact, conditioned=conditioned)
        answer = f.encode(gold, c["text"])
        if c.get("task") == "relations_only":  # entities given in the prompt, relations are the whole answer
            entity_part, relation_part = split_entities(answer)
            msgs[-1] = {**msgs[-1], "content": msgs[-1]["content"] + "\n\nEntities:\n" + entity_part}
            answer = relation_part
        msgs.append({"role": "assistant", "content": answer})
        examples.append({"messages": msgs, "chunk_id": c["chunk_id"], "schema": root})
    return examples


def split_entities(answer: str) -> tuple[str, str]:
    """A sentences-format answer -> (entity lines, the rest: sentence headers and relation lines)."""
    import re
    lines = answer.split("\n")
    k = next((i for i, line in enumerate(lines) if re.fullmatch(r"S\d+( \d+)?", line)), len(lines))
    return "\n".join(lines[:k]), "\n".join(lines[k:])


PROGRESS_EVERY = 10  # steps between progress lines


def _progress_callback():
    """One flushed line per PROGRESS_EVERY steps: step, loss, elapsed, ETA. Progress bars do not
    reach a remote job's log, and a run that cannot be watched cannot be stopped in time."""
    import time
    from transformers import TrainerCallback

    class Progress(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            self.t0, self.s0 = time.time(), None  # s0: first step seen, so a resumed run's ETA is right
            # leading newline: progress bars redraw with carriage returns and leave the cursor
            # mid-line, so without it this line is glued onto a bar and line filters miss it
            print(f"\nprogress: step 0/{state.max_steps}", flush=True)

        def on_log(self, args, state, control, logs=None, **kw):
            if self.s0 is None:  # before the early return: the first logged step, not the first printed one
                self.s0 = state.global_step - 1
            if state.global_step % PROGRESS_EVERY and state.global_step != state.max_steps:
                return
            el = time.time() - self.t0
            eta = el / max(state.global_step - self.s0, 1) * (state.max_steps - state.global_step)
            loss = (logs or {}).get("loss")
            print(f"\nprogress: step {state.global_step}/{state.max_steps}"
                  + (f" loss {loss:.3f}" if loss is not None else "")
                  + f" elapsed {el / 60:.1f} min eta {eta / 60:.1f} min", flush=True)

    return Progress()


def _epoch_adapter_callback(out_dir: Path, tokenizer):
    """Save the adapter at the end of every epoch to out_dir/adapter_epoch<n>, so the best epoch
    can be chosen on a validation set afterwards."""
    from transformers import TrainerCallback

    class EpochAdapters(TrainerCallback):
        def on_epoch_end(self, args, state, control, model=None, **kw):
            path = out_dir / f"adapter_epoch{round(state.epoch)}"
            model.save_pretrained(str(path))
            tokenizer.save_pretrained(str(path))
            print(f"\nprogress: saved {path.name}", flush=True)

    return EpochAdapters()


def train(train_chunks: Path, out_dir: Path, model_id: str = DEFAULT_MODEL,
          max_seq_length: int = 3072, epochs: int = 3, lr: float = 2e-4, r: int = 32,
          grad_accum: int = 8, seed: int = 0, lora_dropout: float = 0.05,
          fmt: str = STUDENT_FORMAT, last_response_only: bool = True, fast: bool = False,
          save_steps: int = 0, checkpoint_dir: Path | None = None, resume: bool = False,
          compact: bool = False, epoch_adapters: bool = False, init_adapter: Path | None = None,
          conditioned: bool = False, schema_variants: bool = False, held_out: dict | None = None,
          examples: list[dict] | None = None, on_trainer=None, summary_extra: dict | None = None) -> dict:
    """QLoRA by default (fits an 8 GB card). With fast=True, for large GPUs: bf16 base
    weights instead of 4-bit, no gradient checkpointing, and the whole effective batch in
    one step instead of accumulated. With save_steps, a checkpoint (adapter, optimizer,
    scheduler) is written every save_steps steps to checkpoint_dir, keeping only the latest;
    resume=True continues from the latest checkpoint there. The effective batch
    size is the same either way, so results stay comparable. Examples are not packed:
    last_response_only masking would keep only one answer per packed sequence.
    init_adapter: continue from a trained adapter instead of a fresh LoRA (its rank and targets must
    match r and the target modules here); every adapter weight must load, or training stops.
    conditioned, schema_variants and held_out: see build_examples.
    examples: ready chat examples instead of build_examples(train_chunks, ...); on_trainer(trainer, tokenizer):
    called before training (e.g. to replace the loss); summary_extra: more train_summary.json fields."""
    from unsloth import FastLanguageModel  # must be imported before transformers/trl
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id, max_seq_length=max_seq_length, load_in_4bit=not fast, dtype=torch.bfloat16)
    model = FastLanguageModel.get_peft_model(
        model, r=r, lora_alpha=r, lora_dropout=lora_dropout, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing=False if fast else "unsloth", random_state=seed)
    if init_adapter is not None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        state = load_file(str(init_adapter / "adapter_model.safetensors"))
        result = set_peft_model_state_dict(model, state)
        missing = [k for k in result.missing_keys if "lora_" in k]
        if missing or result.unexpected_keys:
            raise ValueError(f"adapter {init_adapter} does not match: {len(missing)} LoRA weights missing, "
                             f"{len(result.unexpected_keys)} unexpected (rank or target modules differ?)")
        print(f"continuing from {init_adapter}: {len(state)} adapter tensors loaded", flush=True)

    if examples is None:
        examples = build_examples(train_chunks, fmt, compact, conditioned, schema_variants, held_out, seed)

    def to_text(ex):
        return {"text": render_chat(tokenizer, ex["messages"])}

    ds = Dataset.from_list(examples)
    ds_columns = list(ds.column_names)
    ds = ds.map(to_text, remove_columns=ds_columns)
    lengths = [len(tokenizer(t)["input_ids"]) for t in ds["text"]]

    cfg = SFTConfig(
        output_dir=str(checkpoint_dir or out_dir / "checkpoints"),
        per_device_train_batch_size=grad_accum if fast else 1,
        gradient_accumulation_steps=1 if fast else grad_accum, num_train_epochs=epochs, learning_rate=lr,
        lr_scheduler_type="cosine", warmup_ratio=0.05, logging_steps=1,
        save_strategy="steps" if save_steps else "no", save_steps=save_steps or 500, save_total_limit=1,
        optim="adamw_8bit", weight_decay=0.01, seed=seed, max_length=max_seq_length,
        dataset_text_field="text", report_to="none", bf16=True, fp16=False,
        dataset_num_proc=4)  # Unsloth's default spawned 64 workers: 2+ minutes to tokenize 118 examples
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, args=cfg,
                         callbacks=[_progress_callback()] + ([_epoch_adapter_callback(out_dir, tokenizer)] if epoch_adapters else []))
    trainer = train_on_responses_only(
        trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n",
        last_response_only=last_response_only)  # the worked example's answer is prompt, not target
    if on_trainer is not None:
        on_trainer(trainer, tokenizer)
    stats = trainer.train(resume_from_checkpoint=True if resume else None)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))
    summary = train_summary(
        examples, lengths, init_adapter=init_adapter, conditioned=conditioned, schema_variants=schema_variants,
        held_out=held_out, fmt=fmt, compact=compact, model_id=model_id, last_response_only=last_response_only,
        fast=fast, epochs=epochs, lr=lr, r=r, lora_dropout=lora_dropout, train_loss=stats.training_loss,
        runtime_s=stats.metrics.get("train_runtime"), peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        **(summary_extra or {}))
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "adapter" / "relweave_config.json").write_text(json.dumps(relweave_config(summary), indent=1) + "\n")
    return summary


def train_summary(examples: list[dict], lengths: list[int], *, init_adapter, conditioned, schema_variants, held_out,
                  fmt, compact, model_id, last_response_only, fast, epochs, lr, r, lora_dropout, train_loss, runtime_s,
                  peak_vram_gb, task="extract", pair_chunks=None, pair_distance=False, relation_weight=None) -> dict:
    """The train_summary.json content. "schemas" lists root schema names (looked up in SCHEMAS), never variant
    names; "schema" is the one schema when all examples share it."""
    schemas = sorted({ex["schema"] for ex in examples if "schema" in ex})
    return {
        **({"schema": schemas[0]} if len(schemas) == 1 else {}),
        "task": task, "pair_chunks": str(pair_chunks) if pair_chunks else None, "pair_distance": pair_distance,
        "init_adapter": str(init_adapter) if init_adapter else None,
        "conditioned": conditioned, "schema_variants": schema_variants, "held_out": held_out,
        "schemas": schemas,
        "examples": len(examples), "max_tokens": max(lengths), "mean_tokens": sum(lengths) / len(lengths),
        "format": fmt, "compact_prompt": compact, "model_id": model_id, "last_response_only": last_response_only,
        "fast": fast, "epochs": epochs, "lr": lr, "r": r, "lora_dropout": lora_dropout,
        "relation_weight": relation_weight, "train_loss": train_loss, "runtime_s": runtime_s,
        "peak_vram_gb": peak_vram_gb,
    }


def relweave_config(summary: dict, max_new_tokens: int = 2500) -> dict:
    """relweave_config.json of a generator directory: what the extractor needs to prompt it the way it was trained."""
    return {"schema": summary.get("schema", "business"), "format": summary["format"],
            "compact_prompt": summary["compact_prompt"], "conditioned": summary["conditioned"],
            "base_model": summary["model_id"], "lora_rank": summary["r"], "max_new_tokens": max_new_tokens}
