"""Assemble a model folder in the published layout from a generator run and a head run (nothing is uploaded).

    <out>/generator/   adapter_config.json, adapter_model.safetensors, tokenizer.json, tokenizer_config.json,
                       chat_template.jinja, relweave_config.json
    <out>/head/        the same adapter and tokenizer files, head.safetensors, head_config.json
    <out>/schema.json  the schema (Schema.json_schema())
    <out>/README.md    model card skeleton (fill in the results; check the licences)

The generator run holds adapter/ and train_summary.json (relweave train-generator); the head run holds adapter/,
head.safetensors + head_config.json or a pickled head.pt, and train_summary.json (relweave train-head). When the
generator adapter has no tokenizer files, the base model's tokenizer is saved (the one the generator is then
prompted with), keeping the adapter's own chat_template.jinja.

Usage: uv run python scripts/export_model.py --generator RUN --head RUN --out DIR [--schema NAME|path.py:NAME]
       [--base-model unsloth/qwen3-4b-unsloth-bnb-4bit] [--cut -2.0] [--margin 0.5] [--repo-id chrullis/relweave-4b-base]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
BASE_MODEL = "unsloth/qwen3-4b-unsloth-bnb-4bit"


def _adapter(run: Path) -> Path:
    from relweave.head import adapter_dir
    return adapter_dir(run)


def _copy_adapter(src: Path, dst: Path, base_model: str) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ADAPTER_FILES:
        if not (src / name).exists():
            raise FileNotFoundError(f"{src}: missing {name}")
        shutil.copy2(src / name, dst / name)
    if not all((src / n).exists() for n in TOKENIZER_FILES[:2]):
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(base_model).save_pretrained(str(dst))
    for name in TOKENIZER_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    missing = [n for n in TOKENIZER_FILES if not (dst / n).exists()]
    if missing:
        raise FileNotFoundError(f"{dst}: no {', '.join(missing)}")
    for extra in set(p.name for p in dst.iterdir()) - set(ADAPTER_FILES) - set(TOKENIZER_FILES):
        (dst / extra).unlink()  # only the published files


def export(generator: Path, head: Path, out: Path, schema=None, base_model: str = BASE_MODEL, cut: float = -2.0,
           margin: float = 0.5, max_new_tokens: int = 2500, repo_id: str = "chrullis/relweave-4b-base") -> Path:
    from relweave.extract import generator_settings
    from relweave.head import LAYERS, head_config, load_head, save_head
    from relweave.schema import load_schema, registered_schema

    settings = generator_settings(generator)
    schema = load_schema(schema) if schema else registered_schema(settings.get("schema", "business"))
    if settings.get("schema", "business") != schema.name:
        raise ValueError(f"the generator was trained on {settings.get('schema')!r}, not {schema.name!r}")
    gen_adapter, head_adapter = _adapter(generator), _adapter(head)
    _copy_adapter(gen_adapter, out / "generator", base_model)
    rank = json.loads((gen_adapter / "adapter_config.json").read_text()).get("r")
    (out / "generator" / "relweave_config.json").write_text(json.dumps({
        "schema": schema.name, "format": settings.get("format", "sentences"),
        "compact_prompt": settings.get("compact_prompt", False), "conditioned": settings.get("conditioned", False),
        "base_model": base_model, "lora_rank": rank, "max_new_tokens": max_new_tokens}, indent=1) + "\n")

    _copy_adapter(head_adapter, out / "head", base_model)
    net, config = load_head(head)
    if config.get("schema", "business") != schema.name:
        raise ValueError(f"the head was trained on {config.get('schema')!r}, not {schema.name!r}")
    layers = tuple(config.get("layers", LAYERS))
    prompt = config.get("prompt", {})
    save_head(net, out / "head", head_config(net, schema, net.net[1].in_features // len(layers), layers,
                                             compact_prompt=prompt.get("compact_prompt", True),
                                             conditioned=prompt.get("conditioned", settings.get("conditioned", False)),
                                             cut=cut, margin=margin))
    (out / "schema.json").write_text(json.dumps(schema.json_schema(), indent=1) + "\n")
    (out / "README.md").write_text(model_card(schema, base_model, repo_id, cut, margin))
    return out


def _base_name(base_model: str) -> str:
    """"Qwen3-4B" for unsloth/qwen3-4b-unsloth-bnb-4bit; the id itself when it does not name a Qwen3 size."""
    import re
    m = re.search(r"qwen3-([\d.]+)b", base_model.lower())
    return f"Qwen3-{m.group(1)}B" if m else base_model


def model_card(schema, base_model: str, repo_id: str, cut: float, margin: float) -> str:
    original = _base_name(base_model)
    relations = "\n".join(f"| {e} | {'|'.join(sorted(schema.sources(e)))} -> {'|'.join(sorted(schema.targets(e)))} | "
                          f"{schema.definition(e)} |" for e in schema.edge_names())
    return f"""---
base_model: {base_model}
library_name: relweave
license: apache-2.0
language: en
tags:
- relation-extraction
- named-entity-recognition
- knowledge-graph
- peft
- lora
---

# {repo_id.split("/")[-1]}

Two LoRA adapters on `{base_model}` (Qwen/{original} quantized to 4 bits) that turn English text into a typed
knowledge graph with the [relweave](https://github.com/memlocator/relweave) library: a generator that writes entities
(with all their mentions) and relations, and a pair-classification head that scores every entity pair; the library
keeps a generated relation unless the head scores it at or below {cut} and adds every relation the head scores above
{margin:+}.

## How to use

```python
# pip install git+https://github.com/memlocator/relweave
from relweave import Extractor

g = Extractor(weights="{repo_id}").run(open("report.txt").read())
print(g.to_json())  # JSON Graph Format v2
```

Command line: `relweave run report.txt --out graph.json --weights {repo_id}`.

Fast generation with vLLM (after `scripts/merge_for_vllm.py` has put the merged generator at the root):
`vllm serve {repo_id}`, then `Extractor(weights="{repo_id}", generator_url="http://localhost:8000")`; the pair head
runs locally.

## Files

- `generator/`: PEFT adapter, tokenizer and `relweave_config.json` (schema, output format, prompt settings).
- `head/`: PEFT adapter, tokenizer, `head.safetensors` (the MLP over the probe hidden states) and `head_config.json`
  (schema, label order, layers read, union settings).
- `schema.json`: the schema `{schema.name}`.
- root (`config.json`, `model.safetensors`, tokenizer): the generator as a plain bf16 model for vLLM or other
  servers: the 4-bit base dequantized with the adapter merged (`scripts/merge_for_vllm.py`).

## Schema

Entity types: {", ".join(schema.node_names())}.

| relation | endpoints | definition |
|---|---|---|
{relations}

## Results

TODO: fill in from docs/methodology.md (typed relation F1, strict, validation and test).

## Limits

TODO: from docs/methodology.md, Section 8 (English business text, the schema above only, at most 40 entities and
40 relations per chunk, list items are the main source of misses).

## Training data and licences

The code (relweave) and these weights are licensed under the Apache License 2.0. The adapters were trained on English
Wikipedia passages (CC BY-SA 4.0; attribution: Wikipedia contributors, https://en.wikipedia.org), labelled by a
language model; redistributing the training text or labels derived from it falls under CC BY-SA 4.0. The base model
{original} is Apache 2.0. Details: docs/methodology.md, Section 9.

## Citation

TODO
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--generator", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--schema", default=None)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--cut", type=float, default=-2.0)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--repo-id", default="chrullis/relweave-4b-base")
    a = ap.parse_args()
    out = export(a.generator, a.head, a.out, a.schema, a.base_model, a.cut, a.margin, repo_id=a.repo_id)
    print(f"exported to {out}: " + ", ".join(sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())))


if __name__ == "__main__":
    main()
