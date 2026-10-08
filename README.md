# relweave

relweave turns English text of any length into a typed knowledge graph: entities (people, organisations, places,
objects, events) with every mention located in the text, and typed relations between them with the sentence that
states each one. The graph comes out as JSON Graph Format (v2), networkx or GraphML.

It runs a fine-tuned 4B language model on one consumer GPU (8 GB is enough). The text is cut into chunks; per chunk a
generator writes the entities and a first set of relations, a pair-classification head scores every entity pair, and
the two are combined (the union rule below). Entities are then merged across chunks.

## Install

```
pip install git+https://github.com/memlocator/relweave            # running the model
pip install "relweave[train] @ git+https://github.com/memlocator/relweave"   # also training
```

A PyPI release will follow. relweave needs Python 3.12 and a CUDA GPU with about 5 GB free (the base model is 4-bit; the generator and the head
each take about 3.6 GB and are loaded one after the other below 9 GB free).
On first use it downloads the model `chrullis/relweave-4b-base` and the base model
`unsloth/qwen3-4b-unsloth-bnb-4bit` from the Hugging Face Hub.

## Quick start

```python
from relweave import Extractor

ex = Extractor()                      # chrullis/relweave-4b-base, business schema
g = ex.run(open("report.txt").read())

for e in g.entities:
    print(e.id, e.type, e.name, [m.text for m in e.mentions])
for r in g.relations:
    print(r.source, r.type, r.target, r.modality, r.score)

open("graph.json", "w").write(g.to_json())   # JSON Graph Format v2
nx_graph = g.to_networkx()
```

Several documents: `ex.run_many(texts)` returns one graph per text, and `ex.iter_run(texts)` yields
`(index, graph)` as each document finishes; each model is loaded once per call.

Command line:

```
relweave run report.txt --out graph.json
relweave run a.txt b.txt c.txt --out graphs/ --graphml
```

Useful options (`relweave run --help`): `--weights` (a Hub repository or a local directory with `generator/` and
`head/`; `chrullis/relweave-1.7b-base` for the smaller model), `--generator-url` (see below), `--schema`, `--cut` and `--margin` (the union rule), `--batch-size` (default `auto`), `--max-words` (chunk
size). If a model cannot be loaded (for example not enough GPU memory) the command fails with an error and a non-zero
exit code; a single chunk that fails becomes a warning in the graph and the document continues.

## Models

| model | system F1 (validation / test) | generator on vLLM, 8 GB GPU | generator, transformers backend | vLLM weights in memory |
|---|---|---|---|---|
| [relweave-4b-base](https://huggingface.co/chrullis/relweave-4b-base) | 0.781 / 0.783 | about 42 chunks/min (fp8) | about 4.5 chunks/min | 4.2 GB (fp8) |
| [relweave-1.7b-base](https://huggingface.co/chrullis/relweave-1.7b-base) | 0.737 / 0.722 | about 96 chunks/min (bf16) | about 6 chunks/min | 3.3 GB (bf16) |

Speeds on one RTX 3070 Ti (8 GB), 64 test chunks of about 200 words. Over 377 held-out chunks the 1.7B scores 0.048
below the 4B (95% interval 0.030 to 0.067).

Each model repository holds the generator twice: as a LoRA adapter in `generator/` (used by relweave's default
transformers backend on the 4-bit base) and, at the root, merged into the dequantized 4-bit base as one plain bf16
model for vLLM. The pair head (`head/`, its own LoRA plus a classifier) is never merged and always runs in relweave.

## Serving the generator with vLLM

Generation is most of the run time. Each model repository holds, at its root, the generator as an ordinary bf16
model (the 4-bit base the adapter was trained on, dequantized, with the adapter merged in), so a standard vLLM
server or container serves it directly:

```
vllm serve chrullis/relweave-4b-base --quantization fp8 --max-model-len 4096     # 4B: fp8 weights, ~4.5 GB
vllm serve chrullis/relweave-1.7b-base --max-model-len 4096                      # 1.7B: bf16, ~3.5 GB
docker run --gpus all -p 8000:8000 --ipc=host vllm/vllm-openai --model chrullis/relweave-4b-base --quantization fp8
```

The LoRA is already merged into that model, so vLLM needs no `--enable-lora` (tested with vLLM 0.31). The endpoint is
a component, not a chat model: it writes relweave's line format only under the prompt and per-chunk grammar relweave
sends, so call it through relweave.

relweave then sends each chunk to the server with its decoding grammar and runs the pair head locally:

```python
ex = Extractor(weights="chrullis/relweave-4b-base", generator_url="http://localhost:8000")
```

```
relweave run report.txt --out graph.json --generator-url http://localhost:8000
```

On one 8 GB GPU (test set of 64 chunks), the generator ran at about 42 chunks per minute for the 4B (fp8) and 96 for
the 1.7B against 4.5 and 6 with the default transformers backend, at the same F1 (4B system 0.771 vs 0.770, 1.7B 0.749
vs 0.736). On an 8 GB card the 4B needs `--max-model-len 3072 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.84`
(and then leaves no room for the pair head: run the head afterwards or on another GPU). Without a CUDA toolkit
installed, start vLLM with `VLLM_USE_FLASHINFER_SAMPLER=0`. Serving the adapter
on the original full-precision Qwen3 instead of the merged model loses about 0.03 F1 (the adapter learned against the
4-bit weights); `scripts/merge_for_vllm.py` builds the merged model from your own trained weights.

The pair head needs the model's hidden states, which vLLM does not return, so it runs locally in transformers. On a
GPU shared with the server, leave it about 3 GB (for example `--gpu-memory-utilization 0.5`).

## Output

Every entity node carries its type, a name, and every located mention (text, character offsets, the chunk it was read
from and the containing sentence). Every relation edge carries its type, modality (`asserted`, `negated`, `hedged`,
`reported`), the pair head's score, whether the generator or the head proposed it, and its evidence sentences.
Symmetric relation types are undirected edges. The document node links to each entity with `MENTIONED_IN` edges.

## Schema

The model is trained on a business schema of 6 entity types (Person, Org, Object, Place, Coordinate, Event) and 21
relation types (EMPLOYED_BY, EXECUTIVE_OF, BOARD_MEMBER_OF, OWNS_STAKE_IN, SUBSIDIARY_OF, ACQUIRED,
HEADQUARTERED_IN, ...). The full list with definitions is in `docs/methodology.md`.

A schema is written as Python classes; the docstring is the definition the model reads:

```python
from relweave.schema import Entity, Relation, Schema

class Ship(Entity):
    """A vessel."""

class Port(Entity):
    """A harbour."""

class DockedAt(Relation[Ship, Port]):
    """The Ship lies in the Port."""

SHIPPING = Schema(name="shipping", entities=[Ship, Port], relations=[DockedAt])
```

Prompts, decoding grammars, validation and scoring are derived from the classes. A model only knows the types it was
trained on: for your own schema, label data and train both parts as described in `docs/recipe-own-schema.md`, then
run with `--schema path/to/module.py:SHIPPING --weights <your directory>`.

## Your own relation types without training (experimental)

With `zeroshot=True`, relweave scores relation types you define at run time with
[relweave-4b-zeroshot](https://huggingface.co/chrullis/relweave-4b-zeroshot): the generator still finds the entities
(Person, Org, Place, Object, Event, Coordinate), and for every entity pair and each of your relation types a yes/no
adapter reads the type's definition (the class docstring) and scores it.

```python
class DonatedTo(Relation[Person | Org, Org]):
    """The source has given money, goods or other gifts to the target organisation."""

ex = Extractor(schema=Schema(name="charity", entities=[Person, Org], relations=[DonatedTo]), zeroshot=True,
               threshold=0.8)
```

On 12 relation types it never saw, it scores 0.82 F1 on short synthetic passages and 0.74 on dense ones with a
threshold calibrated on labelled passages, and 0.56 on 15 unseen Re-DocRED types. The best threshold depends on the
text (about 0.7 for short passages, about 0.9 for dense ones); calibrate it on a few labelled chunks with
`relweave.zeroshot.calibrate`. Details, numbers and limits: `docs/zero-shot.md`.

## How it decides relations

- **Generator.** Qwen3-4B with a LoRA adapter writes entity lines and relation lines under a grammar derived from the
  schema, so every output parses and every relation is legal for its endpoint types.
- **Pair head.** A second LoRA adapter on the same base plus a small MLP reads the text, the generator's entity lines
  and one probe per entity pair, and scores every relation type and direction against a learned threshold.
- **Union.** A relation the generator wrote is kept unless the head scores it at or below the cut (-2.0); every
  relation the head scores above the margin (+0.5) is added when it is legal. Both values were chosen on validation.

## Results

Typed relation F1 (strict: both endpoints aligned to gold entities, type and direction right) on held-out English
business Wikipedia chunks from articles not used in training (some names and facts recur; see
`docs/methodology.md`, Section 4):

| system | validation | test |
|---|---|---|
| 4B generator alone | 0.650 | 0.684 |
| relweave-4b-base (generator + pair head, union) | 0.781 | 0.783 |

The test set has 105 chunks, so one test score is uncertain by about +-0.045. Two independent labelling
passes agree at about 0.88. Details: `docs/methodology.md`.

## Limits

- English only; trained on business Wikipedia text, which is cleaner than news or filings.
- The published weights know only the business schema above.
- At most 40 entities and 40 relations per chunk (chunks are about 200 words, so this rarely binds).
- Long coordinated lists are the main error source: later list items are often missed.
- Entity merging across chunks is rule-based (names, overlap, descriptions); it can join a company with a renamed
  predecessor or keep two spellings apart.

## Training on your own data

`docs/recipe-own-schema.md` walks through it: chunk documents (`relweave label ingest`), have a frontier model label
them (`relweave label prompts`, `relweave label import`), train the generator (`relweave train-generator`) and the
pair head (`relweave train-head`), and evaluate (`relweave generate`, `relweave score`, `relweave eval`).
`docs/methodology.md` documents how `relweave-4b-base` itself was trained.

## Licence

The code and the model weights (chrullis/relweave-4b-base, chrullis/relweave-1.7b-base) are licensed under the Apache License 2.0 (`LICENSE`). The
weights were trained on English Wikipedia passages (CC BY-SA 4.0; attribution: Wikipedia contributors,
https://en.wikipedia.org); redistributing the training text or labels derived from it falls under CC BY-SA 4.0. The base
models Qwen3-4B and Qwen3-1.7B are Apache 2.0. Details: `docs/methodology.md`, Section 9.

## Links

- Models: https://huggingface.co/chrullis/relweave-4b-base, https://huggingface.co/chrullis/relweave-1.7b-base
- Training and evaluation data: https://huggingface.co/datasets/chrullis/relweave-business-data (CC BY-SA 4.0)
- Methodology: `docs/methodology.md`
- Training on your own schema: `docs/recipe-own-schema.md`
