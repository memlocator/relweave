"""Rewrite the Deployment and Files sections of the published model cards (relweave-4b-base, relweave-1.7b-base)
and the Models section of this repository's README from one template, so the cards, the README and their numbers
stay in step.
Usage: python scripts/card_sections.py CARD.md 4b|1.7b
       python scripts/card_sections.py README.md readme"""
import sys
from pathlib import Path

MODELS = """| model | system F1 (validation / test) | generator on vLLM, 8 GB GPU | generator, transformers backend | vLLM weights in memory |
|---|---|---|---|---|
| [relweave-4b-base](https://huggingface.co/chrullis/relweave-4b-base) | 0.781 / 0.783 | about 42 chunks/min (fp8) | about 4.5 chunks/min | 4.2 GB (fp8) |
| [relweave-1.7b-base](https://huggingface.co/chrullis/relweave-1.7b-base) | 0.737 / 0.722 | about 96 chunks/min (bf16) | about 6 chunks/min | 3.3 GB (bf16) |

Speeds on one RTX 3070 Ti (8 GB), 64 test chunks of about 200 words. Over 377 held-out chunks the 1.7B scores 0.048
below the 4B (95% interval 0.030 to 0.067)."""

SERVE = {
    "4b": ("vllm serve chrullis/relweave-4b-base --quantization fp8 --max-model-len 4096",
           "--model chrullis/relweave-4b-base --quantization fp8 --max-model-len 4096",
           "On an 8 GB card the 4B needs `--max-model-len 3072 --max-num-batched-tokens 2048 --gpu-memory-utilization "
           "0.84` and then leaves no room for the pair head: run the head after the server stops, or on another GPU."),
    "1.7b": ("vllm serve chrullis/relweave-1.7b-base --max-model-len 4096",
             "--model chrullis/relweave-1.7b-base --max-model-len 4096",
             "On an 8 GB card the 1.7B server and the pair head fit together with `--gpu-memory-utilization 0.52 "
             "--max-model-len 3072 --max-num-batched-tokens 1024 --enforce-eager`."),
}


def deployment(size: str) -> str:
    serve, docker, small = SERVE[size]
    repo = f"chrullis/relweave-{size}-base"
    return f"""## Deployment

relweave runs in two steps per chunk: the generator writes entities and relations, then the pair head scores every
entity pair. The generator is most of the run time, and it can be served by vLLM:

```bash
{serve}
# or the official container (same arguments):
docker run --gpus all -p 8000:8000 --ipc=host vllm/vllm-openai {docker}
```

```python
from relweave import Extractor
ex = Extractor(weights="{repo}", generator_url="http://localhost:8000")
graph = ex.run(open("report.txt").read())
```

- The model at the repository root is the generator with its LoRA already merged (see Files), so vLLM needs no
  `--enable-lora` and no adapter arguments. Tested with vLLM 0.31.
- The endpoint is a component, not a chat model: it answers in relweave's line format only when prompted the way it
  was trained and decoded under the per-chunk grammar relweave sends with each request. Call it through relweave
  (`generator_url`), which also runs the pair head and merges entities across chunks.
- The pair head always runs locally in relweave (it needs the model's hidden states, which vLLM does not return), on
  a GPU with about 3 GB free.
- {small}
- Without a CUDA toolkit installed, start vLLM with `VLLM_USE_FLASHINFER_SAMPLER=0`.

### Choosing a model

{MODELS}
"""


def files(size: str) -> str:
    base = {"4b": "Qwen3-4B", "1.7b": "Qwen3-1.7B"}[size]
    return f"""## Files

| path | contents | used by |
|---|---|---|
| `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json` (root) | the generator as one plain bf16 model: the 4-bit {base} base the adapter was trained on, dequantized, with the generator LoRA merged in. The pair head is not in it. | vLLM and other servers |
| `generator/` | the generator LoRA adapter (PEFT) and tokenizer, unmerged; `relweave_config.json` holds the base model id and prompt settings | relweave, transformers backend (adapter on the 4-bit base) |
| `head/` | the pair head: its own LoRA adapter (PEFT) on the 4-bit base, `head.safetensors` (the classifier over the probe hidden states) and `head_config.json` (label set, layers read, union settings) | relweave, always local |
| `schema.json` | the business schema as JSON (types, definitions, endpoints) | relweave |

relweave downloads only `generator/`, `head/` and `schema.json`. vLLM loads only the root model; its download also
fetches the adapters' `.safetensors` files (a few hundred MB), which it does not use.
"""


def readme_models() -> str:
    return f"""## Models

{MODELS}

Each model repository holds the generator twice: as a LoRA adapter in `generator/` (used by relweave's default
transformers backend on the 4-bit base) and, at the root, merged into the dequantized 4-bit base as one plain bf16
model for vLLM. The pair head (`head/`, its own LoRA plus a classifier) is never merged and always runs in relweave.

"""


def main(path: str, size: str) -> None:
    s = Path(path).read_text()
    if size == "readme":
        a, b = s.index("## Models"), s.index("## Serving the generator with vLLM")
        Path(path).write_text(s[:a] + readme_models() + s[b:])
        return
    a = s.index("### Serving the generator with vLLM")
    b = s.index("## Results")
    s = s[:a] + deployment(size) + "\n" + s[b:]
    s = s[:s.index("## Files")] + files(size)
    Path(path).write_text(s)


if __name__ == "__main__":
    main(*sys.argv[1:])
