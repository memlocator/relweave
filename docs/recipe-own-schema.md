# Recipe: train relweave on your own entity and relation types

This recipe takes you from a list of entity and relation types to a working generator, pair head and `relweave`
extractor for that schema.

**Status.** Run end to end on a new schema on 2026-10-14: this recipe's own `projects` example, 24 written documents
(14 prose, 10 markdown-style), labelled through steps 2 and 3a by a language-model annotator, every command of steps 5
to 8 run as written on a 24 GB GPU. Results on 6 test chunks: generator alone relation F1 0.83 (P 0.76, R 0.91), entity
F1 0.92; the generator trained 3 epochs in 3 minutes at 4.7 GB. Two problems it found are fixed here: the labelling
prompts of a new schema carried no description of the JSON answer (fixed in relweave), and `--examples` sized to the
number of training chunks starved the head (step 7). With so little data the head did not help (union 0.83 at cut -12,
0.08 at cut -2): see step 8. The numbers are from a toy set and only show that the steps work.

**New types need data and training.** relweave-4b-base reads the 6 entity types and 21 relation types of its business
schema. It does not read new types zero-shot. An experimental schema-conditioned head reached 0.29 F1 on unseen
relation types, against about 0.80 on trained ones, and is not part of the model (see `docs/whitepaper.md`, Section 5.5).
You must label text with your types and train.

Install with the training extra: `pip install "relweave[train] @ git+https://github.com/memlocator/relweave"`. Every
command has `--help`.

## 0. What you need

| item | value |
|---|---|
| GPU, generator training | One GPU with 8 GB or more. QLoRA training of the 4B generator peaked at 4.84 GB (the business run: 4-bit base, LoRA rank 32). |
| GPU, head training | More than the generator. The head runs one forward pass per chunk with up to `--max-probes` packed probes. On a 24 GB L4 the business head used 200 probes. An 8 GB card fits only about 80 probes, which measurably hurt the business head. |
| GPU, inference | 4-bit 4B base. About 3.6 GB per loaded model (generator and head), loaded one at a time below 9 GB free. |
| time | Generator: the business run trained 10,765 examples for one epoch in 1.33 hours. A few hundred chunks for 2 to 3 epochs is minutes to an hour on a comparable GPU (an estimate from that rate, not a measurement). Head, measured on the government-report schema (A10G): generating on the 540 training records it needs took 95 minutes (about 10 s per record), training the head 17 minutes; the generator's 3 epochs took 25 minutes. |
| labelled data, starting point | A few hundred labelled chunks (about 1,200 characters each) from at least 10 to 20 documents, plus a validation set and a test set of 50 to 100 chunks each. |
| labelled data, what helps | More chunks and more documents. The business model used about 2,900 labelled chunks plus 500 list-rich chunks, expanded to 10,765 training examples with sentence windows. The last 500 chunks added +0.003 validation F1, which is not significant: the curve had flattened there. No learning curve exists for a new schema. |
| labeller | A large language model annotator (or human annotators) that follows written conventions. Budget is about 6,100 tokens per chunk from scratch. |

Test noise matters when you plan. With 105 test chunks the business score is uncertain by about +-0.045.

## 1. Write the schema module

A schema is a Python module anywhere on disk. Entity types are `Entity` subclasses. Relation types are
`Relation[Source, Target]` subclasses. The rules (from `relweave.schema.base`):

- The docstring of a class is its definition. The model reads it. Write it as a rule: what counts, what does not, and
  which neighbouring type to use instead.
- A relation name comes from the class name: `DepartmentOf` becomes `DEPARTMENT_OF`.
- `Relation[Person, Department | Org]` allows several endpoint types.
- Optional fields on a relation (`title: str | None = None`) are its attributes. Required fields on an entity class are
  its required attributes.
- `symmetric = True` marks a relation with no direction.
- `aliases` and `paraphrases` are used only when training with `--schema-variants`. Leave them out otherwise.
- Validation rules: entity names are unique; relation names are unique and differ from entity names; every endpoint
  type must be in the schema; an alias may not equal an existing entity or relation name.
- One chunk holds at most 40 entities and 40 relations.

Create `projects.py`:

```python
"""Projects and departments in annual reports."""

from relweave.schema import Entity, Relation, Schema


class Org(Entity):
    """A company, public agency, university, foundation or other organised body, including a partner organisation."""


class Department(Entity):
    """An internal unit of an Org: a division, department, team, laboratory or business area."""


class Project(Entity):
    """A named programme, project, initiative or product development effort with a defined goal."""


class Person(Entity):
    """A human being: executive, manager, researcher or employee."""


class DepartmentOf(Relation[Department, Org]):
    """The Department is an internal unit of the Org; a separate legal company is not a department"""


class Heads(Relation[Person, Department | Org]):
    """Person leads the Department or Org as its head, director, manager or chief; an ordinary employee or a board member without a leading role is not HEADS"""
    title: str | None = None


class RunsProject(Relation[Department | Org, Project]):
    """The Department or Org carries out, owns or is responsible for the Project; a partner that only takes part is PARTNER_IN, and a person working on a project is no relation"""


class PartnerIn(Relation[Org, Project]):
    """The Org takes part in the Project as a partner, co-funder or collaborator without running it; the Org that runs the Project is RUNS_PROJECT"""


class CollaboratesWith(Relation[Org, Org]):
    """The two Orgs are stated to cooperate, in a joint venture, alliance or partnership agreement; ownership and subsidiary ties are not collaboration"""
    symmetric = True


PROJECTS = Schema(
    name="projects",
    entities=[Org, Department, Project, Person],
    relations=[DepartmentOf, Heads, RunsProject, PartnerIn, CollaboratesWith])
```

Every command below takes it as `--schema projects.py:PROJECTS` (a file path and the variable name; `package.module:NAME`
works too), and the Python API as `schema="projects.py:PROJECTS"`. Nothing in relweave needs editing. The schema name
`projects` is what the chunk records, the training summaries, the head and the extractor carry; they must all agree.
A name read from a data or config file is only looked up among schemas already loaded, so always pass `--schema` to
commands that read your chunk files.

Check that it loads and see what the model will read:

```bash
python -c "from relweave.schema import load_schema; print(load_schema('projects.py:PROJECTS').render())"
```

Keep the definitions short and mutually exclusive. Most label errors in the business schema came from two types that
both seemed to fit. A "not X, that is Y" clause in each definition removes most of them.

## 2. Collect and chunk texts

Put one plain-text file per document in `data/projects/texts/` (`*.txt`). The file name without `.txt` is the document
id. Documents in one language, in the register you will run on. If your source is another language, translate it first;
relweave-4b-base reads English.

Chunk and split by document:

```bash
mkdir -p data/projects
relweave label ingest data/projects/texts --out data/projects/chunks.jsonl --test-frac 0.3 --seed 0
```

Options: `--out`, `--test-frac` (share of documents held out, default 0.2), `--seed`, `--max-words` (default 200).
By default the documents are chunked exactly as `relweave run` chunks them (200 words, one paragraph of overlap), the
size the published models were trained on: train on the chunks you will run on. If you change `--max-words` here, pass
the same `--max-words` to `relweave run`. A chunk may hold at most 40 entities and 40 relations (the model's output
caps); `relweave label import` leaves out over-cap chunks and warns. For dense lists and tables, use smaller chunks.
(`--max-chars` gives the older sentence windows of a fixed character count.)

`relweave label split chunks.jsonl --schema projects --out data/projects` does the split below in one command (with
`--group` to keep related documents on one side). By hand: `relweave label ingest` marks documents as `train` or `test`,
and writes `gold: null`. Split the held-out documents into validation and test, and tag every record with the schema name (the draft-correction step reads the `schema` field):

```python
import json, pathlib, shutil

root = pathlib.Path("data/projects")
rows = [json.loads(line) for line in (root / "chunks.jsonl").read_text().splitlines()]
held = sorted({r["doc_id"] for r in rows if r["split"] == "test"})
val_docs = set(held[::2])  # half of the held-out documents validate, half test
out = {"train": [], "val": [], "test": []}
for r in rows:
    r["schema"] = "projects"
    name = "train" if r["split"] == "train" else ("val" if r["doc_id"] in val_docs else "test")
    out[name].append(r)
for name, rs in out.items():
    (root / f"{name}_unlabelled.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs))
(root / "texts_train").mkdir(exist_ok=True)  # the training documents only, for the overlap check in step 4
for d in {r["doc_id"] for r in out["train"]}:
    shutil.copy(root / "texts" / f"{d}.txt", root / "texts_train" / f"{d}.txt")
print({k: len(v) for k, v in out.items()})
```

Save it as `split.py` and run it with `python split.py`. You need at least about 10 documents for this split to give a
non-empty test set.

## 3. Label

You label the validation and test chunks first. Step 4 then removes training documents that overlap them, so you do not
pay to label chunks you will throw away. The procedure below is the same for all three sets.

**What the labels leave out, the model learns to leave out.** Measured on our business schema: where the training
labels skipped relations written as modifiers ("the Danish shipping group Nordvik", "Republican state senator X"), the
model learned to skip them too, and later training on more examples did not undo it. Three habits keep labels useful:

- **One written convention, settled before labelling.** Decide the hard cases in the schema definitions (is the head
  of an office EXECUTIVE_OF it or EMPLOYED_BY it? is a party adjective a membership?). Labels that disagree with each
  other teach the model the average of two conventions.
- **Two independent passes, then adjudication.** Two blind passes with differently worded instructions, and a third
  pass that judges only the relations one pass has and the other lacks. One pass misses about a third of what the
  text states; two passes plus adjudication agreed at 0.75-0.88 on our documents.
- **Audit completeness on a sample.** Before training, have a judge list what a random 40 chunks state but the labels
  lack, and group the misses by how the text writes them (verb, title before the name, modifier, possessive, list). A
  group that is systematically missing is a convention gap to fix in the labels, not in the model.

### 3a. From scratch with teacher prompts

Write one prompt file per chunk. The schema text in the prompt is rendered from your classes:

```bash
relweave label prompts data/projects/test_unlabelled.jsonl --schema projects.py:PROJECTS \
  --out runs/projects/prompts_test
```

Each `<name>.json` has `chunk_id` and `messages`. Have a large language model annotator (or human annotators) answer
each one. The answer is only the JSON extraction, written to `<raw dir>/<name>.txt`, where `<name>` is the prompt file's
name without `.json`. Entity records have `id`, `type`, `name`, `mentions`; relation records have `type`, `source`,
`target`, `modality`, `evidence`. Mentions and evidence are copied verbatim from the chunk.

`relweave label guide` prints the labelling guide: the agent prompt and the conventions that carried over to every
domain. Copy that agent prompt and replace the business-specific lines (for example the headquarters and board-chair
rules) with conventions for your types. Rules worth keeping:

- List every distinct surface string of an entity; every occurrence counts as a mention.
- If a string refers to different entities in the chunk, leave it out of all of them.
- A list gets one relation per item when a lead sentence or heading states the relation for every item.
- Copy evidence character for character.
- Do not infer relations from names alone or from names that merely sit next to each other.
- Extract what the text states, even if it seems implausible.
- Prefer the most specific type.

Import the answers. Answers are parsed and span-checked like model output; anything unverifiable is dropped and counted
in the command output:

```bash
relweave label import data/projects/test_unlabelled.jsonl --schema projects.py:PROJECTS \
  --raw-dir runs/projects/raw_test --out data/projects/gold_test
```

Options: `--raw-dir`, `--out`, `--schema`, `--train-windows` (default 500: training chunks are added again as sentence
windows of about 500 characters with their labels projected, the augmentation the business model used; 0 turns it off).
Chunks marked `test` land in `<out>/test.jsonl`, so the file you use later is `data/projects/gold_test/test.jsonl`.
Records carry `"schema": "projects"`. You can correct labels by editing the JSONL by hand.

Read 30 to 50 labelled chunks yourself before going on. Fix the conventions, not just the chunks.

Repeat for validation (`val_unlabelled.jsonl`, output `gold_val`). Do not label training chunks yet; go to step 4.

### 3b. Cheaper: correct a draft (after a first model exists)

Once you have a first generator and head (steps 5 to 7), a model can draft labels for new chunks and the annotator only
corrects them. On the business set this cost 1,940 tokens per chunk against 6,100 from scratch, and corrected labels
agreed with from-scratch labels at 0.876 typed F1. The risk is anchoring: correctors keep some borderline draft
relations.

1. Run the generator on the new chunks and score its entity pairs with the head (the `relweave generate` and
   `relweave score` commands of steps 6 and 8).
2. Export drafts and instructions:

```bash
relweave label draft data/projects/new_unlabelled.jsonl runs/projects/new.jsonl runs/projects/new_scores.json \
  --schema projects.py:PROJECTS --out runs/projects/correct --n 100 --per-batch 25
```

`--n` chunks are sampled and written in batches of `--per-batch`; `--cut` and `--margin` set the union for the drafts
(default -2.0 and +1.0: fewer head additions than at run time). The chunk records need `"schema": "projects"` (the
split script in step 2 sets it).

3. `runs/projects/correct/instructions.md` contains the schema (rendered from your classes) and a Conventions section
   copied from the labelling guide. Replace that Conventions section with your own before giving the file to the
   annotator.
4. The annotator reads `instructions.md` and one `batch_NN.txt` and writes `answers_NN.txt` into the same folder, with
   one line per change: `drop`, `retype`, `flip`, `entity`, `mention`, `add` (the formats are in `instructions.md`).
5. Merge:

```bash
relweave label correct runs/projects/correct data/projects/new_unlabelled.jsonl --schema projects.py:PROJECTS
```

This writes `runs/projects/correct/labels.jsonl`, in the same layout as `train.jsonl`. Concatenate it with your other
training labels.

## 4. Split by document with no shared entities

Splitting by document is not enough. If a test document names the same people, organisations or projects as a training
document, the model can recall facts instead of reading. Remove training documents that share entities with the
evaluation sets: a training document should go when it names three or more Person or Org entities of the evaluation
gold, or the subject of an evaluation document, and the same holds for your own types (search the training texts for
each project and department name in the evaluation gold). The experiment repository relweave was split from has a
script for the Person and Org part (`scripts/holdout_overlap.py` in `entity-rel-extraction`); relweave itself has no
command for it.

If nearly all training documents are excluded, your documents share entities (for example the same company over
several years). Split by company, not by year, and run steps 2 to 4 again.

Write the excluded document ids to `data/projects/excluded.json` (a JSON list), drop them from the training chunks, then
label what is left with the step 3a procedure (prompts, answers, import):

```python
import json, pathlib
root = pathlib.Path("data/projects")
skip = set(json.loads((root / "excluded.json").read_text()))
rows = [line for line in (root / "train_unlabelled.jsonl").read_text().splitlines() if json.loads(line)["doc_id"] not in skip]
(root / "train_kept.jsonl").write_text("\n".join(rows) + "\n")
print(len(rows), "training chunks kept")
```

```bash
relweave label prompts data/projects/train_kept.jsonl --schema projects.py:PROJECTS --out runs/projects/prompts_train
# ... annotator answers into runs/projects/raw_train ...
relweave label import data/projects/train_kept.jsonl --schema projects.py:PROJECTS \
  --raw-dir runs/projects/raw_train --out data/projects/gold_train
```

Training labels are `data/projects/gold_train/train.jsonl`, sentence windows included.

## 5. Train the generator

Continue from the generator adapter of relweave-4b-base. Download it once:

```bash
hf download chrullis/relweave-4b-base --local-dir models/relweave-4b-base
```

Its LoRA rank is 32, so `--r 32` (the default) is required: the rank and target modules must match or training stops.
Use the conditioned prompt, which puts your rendered schema in the system prompt. The business model never saw a prompt
that lists types, so the first steps learn the new prompt form; the schema in the prompt is how the model learns your
types.

```bash
relweave train-generator data/projects/gold_train/train.jsonl --schema projects.py:PROJECTS \
  --init-adapter models/relweave-4b-base/generator \
  --lr 1e-4 --epochs 3 --conditioned --epoch-adapters \
  --out runs/projects/gen
```

- The compact prompt (a short system prompt without a worked example) is the default. With `--conditioned` it still
  lists your types and definitions.
- `--epochs 3` and `--epoch-adapters` (saves `adapter_epoch<n>`) are a starting point for a few hundred chunks. The
  business continuation used 1 epoch at lr 1e-4 on 10,765 examples. Compare epochs on validation (step 8) and keep the
  best.
- Do not use `--held-out` or `--schema-variants` for this recipe. They exist for schema-change experiments; the
  held-out option removes types from training.
- The run writes `runs/projects/gen/adapter` (with `relweave_config.json`) and `runs/projects/gen/train_summary.json`.
  Both record `"schema": "projects"`, `"conditioned": true` and `"compact_prompt": true`. Inference and the head read
  them; `runs/projects/gen` is the generator directory every later command takes.
- Long runs: `--save-steps N` and `--resume` give checkpoints.

Whether continuing from the business adapter beats a fresh LoRA on a new schema is not measured. Continuing is the
default advice (start from a good adapter unless format, prompt or base change). The prompt does change here, so if
validation is poor after 3 epochs, try a fresh LoRA (leave out `--init-adapter`, use `--lr 1.5e-4 --epochs 2`, the
settings of the business pilot stage).

## 6. Run the generator on the training chunks

The head trains on the entity lines the generator itself writes (half of its examples), not only on gold lines. In the
business system this was worth +0.014 test and +0.024 validation F1 together with 200 probes per chunk. Generate on the
training chunks:

```bash
relweave generate data/projects/gold_train/train.jsonl --schema projects.py:PROJECTS \
  --generator runs/projects/gen --out runs/projects/own.jsonl
```

The prompt settings come from the generator directory. `--batch-size` defaults to `auto` (the largest of 1, 2, 4 or 8
that fits the free GPU memory, halved on out-of-memory). Repetition loops are cut off as in production. With
`--engine vllm` (run from an environment with vLLM, README "Faster: the generator on vLLM") the same command runs on
vLLM, several times faster, in the format and parallelism your GPU allows; `--vllm-format`, `--vllm-memory` and
`--vllm-max-seqs` override. The results are the same within noise (whitepaper, Section 5.12).

Do the same for validation and test. These runs are the generator-alone baseline and the input to head scoring:

```bash
for s in val test; do
  relweave generate data/projects/gold_$s/test.jsonl --schema projects.py:PROJECTS \
    --generator runs/projects/gen --out runs/projects/$s.jsonl
done
```

Note on the own lines: the generator was trained on these chunks, so its entity lines are cleaner on them than they
will be on new text. The business head had the same property.

## 7. Train the pair head

`relweave train-head` trains a second LoRA and the head together. The label set follows the schema (for `projects`: 9
labels, one per relation and direction, one for the symmetric relation), so a head from another schema cannot be
reused: without `--base-head` it prints `stage 1: new head for schema projects (9 labels)` and starts a fresh head on top
of the generator's LoRA. The prompt (compact, conditioned or not) follows the generator directory, so head and
generator read the same context.

```bash
relweave train-head data/projects/gold_train/train.jsonl --schema projects.py:PROJECTS \
  --generator runs/projects/gen --own-results runs/projects/own.jsonl \
  --max-probes 200 --examples 600 --out runs/projects/head
```

| option | meaning |
|---|---|
| `--generator` | The generator directory; its adapter is where the head's LoRA starts. |
| `--own-results` | The generator's results on the training chunks (step 6); repeat the option for several files. Read under your schema. |
| `--base-head` | A head to continue from (same label set). Leave it out for a new schema. |
| `--model` | Base model (default `unsloth/qwen3-4b-unsloth-bnb-4bit`). |
| `--examples` | Training examples (default 2400), sampled with repetition from the training chunks. Keep the default, or at least several hundred: with 110 the dry run trained 7 steps and the head learned nothing. Up to half use the generator's own entity lines, the rest use gold entity lines, which also add a language-model loss that keeps generation intact. |
| `--max-probes` | Probes per chunk (default 150): all related pairs first, then sampled unrelated ones. Lower it on a small GPU. |
| `--upcast` | PEFT's fp32 copies of embeddings and `lm_head`; off by default (an untied fp32 vocabulary of the 4B is about 3 GB). |
| `--seed`, `--list-repeat` | Seed; extra copies of chunks that contain a list. |

The output directory holds `adapter/`, `head.safetensors`, `head_config.json` (schema, label order, layers, union
settings) and `train_summary.json`.

The business run continued from a head trained alone on the frozen generator first (stage 0, experiment repository).
That step is optional here. A frozen-model head barely helped (0.667 test against 0.759 after joint training), so the
joint stage is the one that matters. Stage 0 was not tried on a new schema.

Out of memory: lower `--max-probes` (for example 80). Chunks that still fail are skipped and counted as `skipped_oom` in
`train_summary.json`.

## 7b. Train the entity tagger

The generator writes each chunk's entities as a list and may end it early, so in long lists and table columns it
leaves names out, and nothing later can recover a name that was never listed. The entity tagger tags every name in
one pass (BERT-style, over the hidden states of your trained head; it reads each window twice so every token has seen
the whole window). Its names are added to the generator's entity lines before the pair head scores pairs.

```bash
relweave train-tagger data/projects/gold_train/train.jsonl --schema projects.py:PROJECTS \
  --head runs/projects/head --out runs/projects/tagger
```

Options: `--types` (a comma-separated subset of the schema's entity types; default all), `--epochs` (default 2),
`--layers` (default `-1,-14`). The 4B stays frozen and only a small classifier trains: about 12 minutes for 2,700
chunks on an 8 GB GPU. It learns from the entities' name mentions only (the name and mentions sharing a word with it),
not pronouns or descriptions, so coreference stays with the generator.

Use it with `--entity-tagger runs/projects/tagger` (step 9). Measured on the business model: entity recall in long
lists and tables 0.96-1.00 against 0.78-0.92 for the generator, relation F1 +0.03 to +0.08 there, no measurable
change on prose. Label titles that are not entities as nothing: a tagger trained on data that names "the inquiry" as
an entity learned to tag "the" and "The".

## 8. Evaluate, per type

Score the head on validation and test, then evaluate the generator alone and the union:

```bash
for s in val test; do
  relweave score data/projects/gold_$s/test.jsonl runs/projects/$s.jsonl --schema projects.py:PROJECTS \
    --generator runs/projects/gen --head runs/projects/head --out runs/projects/${s}_scores.json
done
relweave eval data/projects/gold_val/test.jsonl runs/projects/val.jsonl --schema projects.py:PROJECTS \
  --scores runs/projects/val_scores.json
```

`relweave eval` prints the generator alone and the union (default cut -2.0 and margin +0.5), each chunk scored under its
own schema. Pick the two cut-offs on validation, for example over a small grid:

```bash
for cut in -12 -8 -4 -2; do for margin in -1 -0.5 0 0.5 1; do
  echo "cut $cut margin $margin: $(relweave eval data/projects/gold_val/test.jsonl runs/projects/val.jsonl \
    --schema projects.py:PROJECTS --scores runs/projects/val_scores.json --cut=$cut --margin=$margin | grep union)"
done; done
```

Then report test once, per type:

```bash
relweave eval data/projects/gold_test/test.jsonl runs/projects/test.jsonl --schema projects.py:PROJECTS \
  --scores runs/projects/test_scores.json --cut=-2 --margin=0.5 --by-type --out runs/projects/test_metrics.json
```

The metric is strict typed relation F1: a predicted relation counts only if both endpoints align to gold entities (by
mention-span overlap and type), and the type and direction match.

**With little data, start from the generator alone.** The head needs far more labelled chunks than the generator: in
the dry run (110 training records) the generator reached 0.83 and the head scored every pair below the default cut, so
the union at cut -2 vetoed almost everything (0.08); at cut -12 the union equals the generator. Use the head only once
the union beats the generator alone on validation (on the business schema, about 3,000 training chunks, it added +0.10).

How to read it:

- Pick every setting (epoch, cut, margin, `--max-probes`) on validation. Report test once.
- The union rule keeps a generator relation unless the head scores it at or below the cut, and adds every legal head
  relation above the margin. The business values were cut -2.0 and margin +0.5. Use the values you picked at run time
  (`--cut`, `--margin`, step 9), or record them in the exported head (step 9).
- The per-type table shows which definitions are weak. A type with many false positives and few false negatives needs a
  tighter definition or more negative examples. A type with many false negatives is either rare or lists.
- Business numbers for orientation: generator alone 0.684, union 0.783 test F1 with 105 test chunks. Do not expect these
  on another schema.
- The generator alone can be scored before the head exists: `relweave eval` without `--scores`.
- If your test set is small (under 100 chunks), differences of 0.02 to 0.03 are within noise. Use a paired bootstrap
  over chunks before you accept a change.

To compare epochs, make a generator directory per epoch (a copy of `runs/projects/gen/train_summary.json` and that
epoch's `adapter_epoch<n>` renamed to `adapter/`), generate validation with it and score the generator alone.

## 9. Use it from relweave

```python
from relweave import Extractor

extractor = Extractor(
    schema="projects.py:PROJECTS",
    generator="runs/projects/gen",     # adapter/ and train_summary.json
    head="runs/projects/head",         # adapter/, head.safetensors, head_config.json
    cut=-2.0, margin=0.5,              # the values picked on validation
)
graph = extractor.run(open("report.txt").read(), doc_id="report-2024")
for r in graph.relations:
    print(r.type, r.source, r.target, r.score, r.origin)
graph.to_json()  # JSON Graph Format; see also graph.to_graphml(path)
```

Command line:

```bash
relweave run report.txt --out graph.json --schema projects.py:PROJECTS \
  --generator runs/projects/gen --head runs/projects/head --cut -2 --margin 0.5
```

Several documents: `extractor.run_many(texts, doc_ids)` or `extractor.iter_run(texts, doc_ids)`; each model loads once
per call. To package both parts in the published layout (a folder you can pass as `--weights`, or upload to the Hugging
Face Hub yourself), use `scripts/export_model.py --generator runs/projects/gen --head runs/projects/head --schema
projects.py:PROJECTS --cut <cut> --margin <margin> --out models/projects`; the cut-offs go into `head_config.json`, so
the extractor uses them by default.

To serve the generator with vLLM (much faster generation, see the README), add the merged model to that folder:
`uv run python scripts/merge_for_vllm.py models/projects models/projects` writes the dequantized 4-bit base with your
adapter merged into the folder's root, so `vllm serve models/projects` loads it and `Extractor(weights="models/projects",
generator_url="http://localhost:8000")` uses it. Do not serve your adapter on the original full-precision base: it was
trained against the 4-bit weights and loses about 0.03 F1 there.

Schema checks. The extractor reads the generator's settings (`relweave_config.json`, or `train_summary.json` of a
training run) and the head's `head_config.json` (or `train_summary.json`). Each must name `projects`, equal to the
schema's name. Otherwise it raises `ValueError` ("generator was trained on schema ..." or "pair head was trained on
schema ..."). Without `schema=`, the extractor takes the generator's schema, which must then be one already loaded.

### When the gold itself is made by a model

If your validation and test labels were written by a language model, a relation the model under test writes that the
gold lacks is not necessarily wrong: on our real documents a third of such "errors" were true facts the labels had
stated at another level or type, or left out, and a fifth of the gold's own relations were not stated by the text.
Score by pooled judging instead (`relweave judge`; the commands are in quickstart-new-domain.md, step 6): collect every relation from the gold and from each system you compare, have two
independent judges decide each one against the text (true only with a verbatim quote, true only if both agree), and
compute precision against that and recall against all judged-true relations of the pool (the gold is then a floor for
recall, not the truth). Test the judges before trusting them: plant decoys (direction swapped, another entity of the
same type, a sibling type, a true-in-the-world fact the text does not state) and require that they reject almost all
of them; have a third reader check a random sample blind. Score every system you compare in the same pool, since adding
a system adds true relations and lowers everyone's pooled recall.

## 10. Troubleshooting

**Recall does not rise when you add training examples.** Check whether the model already knows the relations it
misses: score each missed relation's answer line in place, given the model's own answer up to that point, against the
lines it does write, and sample a few answers per chunk. Read the first token of the line (whether the model would
begin another relation line there at all), not the average over the line, which near-certain tokens such as entity
numbers dominate once a line has begun. If missed lines are nearly as likely
and appear in samples, the model has learned to omit them (usually from incomplete labels), and more examples of the
same kind will not help; complete the labels for that construction, and consider training on the model's own judged
outputs.

**Out of memory in head training.** Lower `--max-probes` (200 to 80). Keep `--upcast` off for the 4B. Close other GPU
processes: training reserves 700 MB (`--reserve-mb`) and caps its own memory to the rest of the free memory. Chunks too
dense for the cap are skipped and counted in `skipped_oom`.

**Out of memory at inference.** Generation batches shrink by themselves on out-of-memory (`--batch-size auto`), and the
head halves its probe groups down to 1. With under 7 GB free, `Extractor` runs sequentially (generate all chunks,
unload, then score). A chunk that still fails becomes an entry in `graph.warnings` and the document continues. If a
model cannot be loaded at all, the extractor raises an error naming the free memory, and `relweave run` exits non-zero.

**Out of memory in generator training.** `relweave train-generator` is already 4-bit with checkpointing. Lower
`--max-seq-length` or shorten chunks. Do not pass `--fast`; it is for large GPUs and uses a bf16 base.

**Low recall on lists.** Lists were the largest error source in the business system: of 250 missed test relations, 159
were in long sentences, mostly coordinated lists where later items were dropped. Remedies, from cheap to costly: (a)
state in your definitions and conventions that a list gets one relation per item, and label that way; (b) add chunks
that contain lists (the 500 list-rich chunks of the business model are the reason for its last gain); (c)
`--list-repeat 2` in head training repeats chunks that have a list; (d) draft correction (step 3b) finds missed items
cheaply.

**Entities missing or the 40-entity warning.** `graph.warnings` carries "hit the 40-entity cap". Use shorter chunks
(`--max-words`, the same for `relweave label ingest` and `relweave run`).

**Invented relations.** The usual causes are entities that merely co-occur or share a neighbour. Check the per-type
table in step 8 for the type that causes it. Tighten its definition with a "not X, that is Y" clause, relabel, and
retrain.

**Schema mismatch errors.**

- `pair head was trained on schema 'business', not 'projects'`: you pointed `head` at the business head (or a head
  directory without your schema). Train a head for your schema (step 7).
- `generator was trained on schema ...`: `generator` points to another schema's run, or its settings are missing.
- `unknown schema 'projects'; known: ...`: a command read the name `projects` from a file but was not given
  `--schema projects.py:PROJECTS`.
- `entity type ... not in schema business`: something validated under the business schema (the default when no schema
  is named). Check that the chunk records have `"schema": "projects"` and that the command got `--schema`.
- `the head does not match the base model's hidden size`: the head was trained on a different base model. Train and run
  with the same base.

**Training looks flat.** Check on a handful of validation chunks by eye before changing hyperparameters. The commonest
cause is labels that disagree with each other, not the optimiser.

## What this recipe does not cover

- Zero-shot use of new types.
- Changing the types after training. A new relation type needs new labels and a new head.
- Licences of your own texts and labels. relweave's code and the relweave-4b-base weights are Apache 2.0; the base
  model Qwen3-4B is Apache 2.0.
