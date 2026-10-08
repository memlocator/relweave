# Zero-shot relation model of relweave (relweave-4b-zeroshot)

This document describes how the working model `relweave-4b-zeroshot` is trained and evaluated. It is written for an ML
practitioner who wants to understand, reproduce or adapt it. The run-by-run records are in the experiment repository the
library was split from, `entity-rel-extraction` (`docs/findings-2026-09-28-filters.md`); numbers here are copied from
there. The model is published as an experimental release,
[chrullis/relweave-4b-zeroshot](https://huggingface.co/chrullis/relweave-4b-zeroshot), with an evaluation set,
[chrullis/relweave-zeroshot-eval](https://huggingface.co/datasets/chrullis/relweave-zeroshot-eval). Training and
benchmark scripts live in the experiment repository; using the model goes through relweave (Section 2a).

## 1. What the model does

Input is a text, a list of entities found in it, and a question: does this relation, given by its definition, hold from
this source entity to this target entity? Output is yes or no, read as a probability. Because the relation is written
into the question, a user can define a new relation type at run time, in plain words, and extract it without
retraining.

The model is a LoRA adapter (rank 16) on Qwen3-4B in 4-bit (QLoRA). It is trained from a fresh adapter on the plain
base model, not on top of `relweave-4b-base`.

| | relweave-4b-base | relweave-4b-zeroshot |
|---|---|---|
| relation types | fixed: 21 business types, learned as output classes | any: given by a definition in the question |
| new type | new labels and retraining of generator and head | write a definition, no training |
| decision | pair-classification head over probe states | yes/no answer to one question per pair and type |
| cost per chunk | one forward pass for all pairs | one question per ordered pair and fitting type (Section 6) |
| quality | 0.783 typed F1 on trained types (test) | 0.74-0.82 on 12 unseen types in synthetic text, 0.56 on 15 unseen Re-DocRED types (Section 4) |

The two are complementary. The fixed model is better on the types it was trained for. The zero-shot model is for types
that do not exist yet.

## 2. Question format and score

One prefix per passage, then one short question per probe. The prefix is the chat template with thinking disabled and
this user message (the text, the entity lines, the instruction):

```
<text>

Entities:
E1 Person: <name> | <other surface string> ...
E2 Org: <name> | ...

For each question below, answer yes if the text states or clearly implies that the relation holds from the source to the target, otherwise no.
```

Each question, appended after the generation prompt, is exactly:

```
Q: Source: <source name> | Target: <target name> | Relation: <definition> A:
```

The model's next token should be " yes" or " no". The score is

    p(yes) = softmax over the logits of the two tokens " yes" and " no" at the last token of the question

and a relation is accepted when p(yes) is above a threshold. All questions of one passage are packed into one forward
pass. An attention mask lets each question see the prefix but not the other questions, and all questions share one
position range, so a question's answer does not depend on the others.

An entity pair is only asked about a relation type when the endpoint types fit: the entity types of source and target
must be among the types seen as that relation's source and target (Section 4 for how this gate is built for the
benchmark).

## 2a. Using it from relweave

Write the relation types as a schema, reusing the generator's entity types (Person, Org, Place, Object, Event,
Coordinate), and run with `zeroshot=True`. The docstring is the definition the model reads: say what the relation is
from "the source" to "the target", and for near types what it is not.

```python
from relweave import Extractor
from relweave.schema import Relation, Schema
from relweave.schema.business import Org, Person

class DonatedTo(Relation[Person | Org, Org]):
    """The source has given money, goods or other gifts to the target organisation."""

class PledgedTo(Relation[Person | Org, Org]):
    """The source has promised a future gift to the target organisation that has not been given yet."""

CHARITY = Schema(name="charity", entities=[Person, Org], relations=[DonatedTo, PledgedTo])
ex = Extractor(schema=CHARITY, zeroshot=True)       # density-calibrated threshold by default
graph = ex.run(open("annual_report.txt").read())   # relations carry the adjusted score, origin "zeroshot"
```

Command line: `relweave run report.txt --schema charity.py:CHARITY --zeroshot --out graph.json`.

The generator finds the entities (as for the fixed model); for every pair of them and every type of your schema whose
endpoint types fit, the adapter answers the question and relations scoring above the threshold are kept. A symmetric
type (`symmetric = True` on the class) is asked once per pair.

**Threshold.** The best raw threshold depends on the text: about 0.7 on short passages with few entities, about 0.9
on dense text, because each extra pair is another chance for a false yes. relweave therefore calibrates by default:
each score is adjusted for the number of questions in its chunk (logit(p) - 0.5 ln(n / 50)) and compared with one
threshold (0.6; constants in `relweave.zeroshot`). Fitted on the synthetic benchmark, this matched the best per-set
threshold on both short and dense passages (0.830 vs 0.829, 0.746 vs 0.744) and gave 0.483 on held-out Re-DocRED
against 0.512 for its best fixed threshold and 0.474 for a raw 0.8: it removes the density effect, not differences
between text types. With a few labelled chunks of your own text, pick thresholds per type on the adjusted scores the
graph holds: `relweave.zeroshot.calibrate([(type, score, is_true), ...])` returns `{type: threshold, "*": pooled}`,
which `Extractor(threshold=...)` accepts. `evaluate.py` in the model repository scores labelled passages and reports F1
tuned and untuned.

## 3. Training data

The training loss is cross-entropy over the " yes" and " no" logits at the last token of each question. One epoch,
learning rate 1e-4, 8 passages per optimiser step, AdamW without weight decay, linear warmup over the first tenth of the
steps then linear decay (floor 5% of the learning rate). LoRA rank 16, alpha 32, dropout 0.05, on all attention and MLP
projections.

### Sources

| source | what | definitions |
|---|---|---|
| Re-DocRED training documents | relation types except the held-out types and their near neighbours (below) | one of three plain-English definitions per type, drawn at random per question (`data/redocred/relations.json`) |
| business chunks | our labelled chunks, all business types except MEMBER_OF, FAMILY_OF and OWNS_STAKE_IN | the type's schema docstring or one of its paraphrases |
| open-type passages | Re-DocRED documents annotated by a large language model with free-form relation names and definitions, minus held-out neighbours | the definition written with the relation |

The first run used 1,693 passages: 600 Re-DocRED documents, 600 business chunks and 493 open-type passages. That gave
80,233 questions, 17,007 of them yes.

### Held-out protocol

The benchmark types must never be seen, nor anything close to them, or the score would measure memory.

- **15 held-out Re-DocRED types**, never in training: director, composer, author, performer, record label, next to
  water, mouth of watercourse, league, platform, developer, publisher, present in work, cast member, religion, official
  language.
- **Near neighbours**, also never in training: member of, member of political party, member of sports team, father,
  mother, spouse, child, sibling, owned by.
- **Business held-out types**: MEMBER_OF, FAMILY_OF, OWNS_STAKE_IN.

Open-type passages whose relation names are held-out neighbours are dropped. The earlier result of 0.25-0.30 on our
three business held-out types was flattering because they have close trained neighbours; the 15 Re-DocRED types are the
benchmark.

### Negatives

A model trained on positives alone would say yes to everything. Each passage gets its gold triples as yes-questions
(capped) and these no-questions.

**Version 1 (the run reported in Section 4), at most 48 questions per passage:**

- the same pair with another relation type whose endpoint types fit;
- the reversed direction of a positive, when the reversed pair is not itself gold;
- random pairs with random fitting types, up to the cap.

**Version 2 (under evaluation, no result yet):**

- a ratio of 1 positive to 8 negatives (`NEG_RATIO=8`);
- half of the negatives are sibling types on the same pair: the five types whose definitions share the most content
  words with the positive's;
- the other negatives are the reversed direction and random fitting pairs;
- Re-DocRED documents use the original entity types (person, organisation, location, miscellaneous) for the endpoint
  gate, as a schema would state them;
- definitions that carry one positive and one negative example (`DEFS_V2=<file>`), after GoLLIE, where examples in the
  guidelines were the largest single factor in the ablation.

## 4. Evaluation

Scored on Re-DocRED development documents with gold entities, asking every directed entity pair with a fitting type
about each of the 15 held-out types. Metric: F1 over (source, target, type) triples; a gold relation whose pair is never
asked counts as a miss.

### Benchmark v1

First 50 development documents, 182 gold relations. The first training run, 1 epoch, 212 steps.

| system | F1 | precision | recall |
|---|---|---|---|
| untrained Qwen3-4B, same questions (`scripts/zs_entail.py`) | 0.152 (at 0.99, still rising) | 0.09 | 0.73 |
| GLiREL, `glirel-large-v0`, relation names as labels (CPU, default threshold 0.5 = its best) | 0.180 | 0.13 | 0.32 |
| **yes/no LoRA, threshold 0.9** | **0.434** | 0.34 | 0.59 |
| yes/no LoRA, threshold 0.95 | 0.415 | 0.42 | 0.41 |
| yes/no LoRA, threshold 0.8 | 0.369 | 0.25 | 0.70 |

The trained model scores 2.4 times GLiREL on this all-pairs extraction task.

**Caveat.** The threshold was chosen on the same 50 documents that are scored, so 0.434 is slightly optimistic. Fifty
documents and 182 relations are also few: differences of a few points between variants cannot be told apart.

### Benchmark v2

`scripts/zs_qa_eval.py` fixes the weaknesses of v1:

- **Separate documents for the threshold.** The threshold is chosen on development documents 50 to 150 and F1 is
  reported on documents 150 to 300. The scored documents are never used for tuning.
- **Endpoint gating with original types.** A type is asked only for pairs whose original Re-DocRED entity types
  (PER, ORG, LOC, MISC) occur as its endpoints in the training labels, counting a type only when it makes up at least
  10% of that side's labels. The collapsed types of v1 (Person, Org, Place, Object) made the gate nearly a no-op.
- **Name matching.** Gold and predictions are compared as (source name, target name, type) with case and a trailing
  plural s normalised, so "Catholic" and "Catholics" count once.
- **Per-type and macro F1.** Pooled F1, per-type precision and recall, and macro F1 over the types with gold.
- **Paired bootstrap** over test documents against another run's saved scores.
- **Optional pair prefilter** (`PREFILTER=<fraction>`): one question per ordered pair, "any relation stated from the
  source to the target", and only the top fraction of pairs per document get typed questions. Its recall of gold pairs
  is reported.

Results on benchmark v2 (v1 adapter): global threshold 0.93 -> F1 0.497 (P 0.487, R 0.507), macro 0.483;
per-type thresholds chosen on the development documents -> 0.561. Version 2 training (more and harder negatives,
definitions with examples) 0.529 at threshold 0.5, not significantly different; the pair prefilter loses gold pairs
without a gain (0.496 at 50% kept, 0.349 at 15%).

An audit of the 257 false positives on the test documents by blind LLM judges (with known-true and known-false
controls; the judges accepted none of 40 known-false items) found about a third stated in the text but missing from the
gold, and only 57% of the gold itself stated in the text: the official number understates how well the model reads.

### Synthetic unseen types

[relweave-zeroshot-eval](https://huggingface.co/datasets/chrullis/relweave-zeroshot-eval): 12 relation types in 4 sets
of near siblings that occur in none of the training data, in fictional passages with planted gold and near misses.

| split | tuned threshold (other half of the passages) | untuned 0.5 |
|---|---|---|
| short (100 passages, 4-8 entities) | 0.82 | 0.77 |
| dense (50 passages, 18-24 entities, incl. list-heavy) | 0.74 | 0.57 (0.73 at 0.93) |

The highest-scoring type of a pair's sibling set is the gold one 96-98% of the time and reversed directions almost
never pass; list-heavy passages reach 0.74 but find about two thirds of list members.

### Format comparison (Qwen3-1.7B)

Multiple choice per pair (all candidate types plus "none" in one question) against yes/no, same data: Re-DocRED 0.405
vs 0.313, synthetic short 0.608 vs 0.728. Multiple choice's best threshold is stable across both benchmarks (0.5-0.7)
but it predicted the reverse of 18% of directed gold triples, because each ordered pair was a separate question; a
version with both directions as options in one question is being tested. The 1.7B is far below the 4B in zero-shot
(Re-DocRED 0.31 vs 0.50 for yes/no).

## 5. What we learned on the way

- **A matching head over frozen states does not carry definitions to unseen types.** A head that scores a pair against
  a type probe built from the definition, on frozen in-context states, reached at most about 0.10 F1 on the 15 held-out
  Re-DocRED types, whatever type-diverse data was added (Re-DocRED types 0.004 to 0.066, open-type passages 0.019 to
  0.063, ZeroRel texts 0.096). The layer matters: transfer to unseen types peaked at layer -14 and trained types are
  best at the top layers.
- **Joint LoRA training makes definitions in context usable**, but zero-shot stayed at about 0.3 on our three business
  held-out types (0.297) and the variants swung by 0.15 in both directions, so three types cannot rank them. That is why
  the 15-type benchmark exists.
- **Asking the model works because the knowledge is there and the calibration is not.** The untrained model, asked the
  question directly, finds 73% of the unseen relations at 9% precision. Training teaches it when to say yes, not what
  the relations are.
- **The gain comes early.** On 10 development documents the untrained model scored 0.200; after 50 steps 0.436; at 100,
  150 and 200 steps 0.40, 0.37 and 0.40. A quarter of the data probably suffices. `zs_qa_train.py` prints this check
  before training and every 50 steps, next to the untrained score.
- **Closest neighbours cost the most.** Types that compete with trained types (OWNS_STAKE_IN against SUBSIDIARY_OF and
  ACQUIRED) transfer worst. Version 2 negatives target exactly this.

## 6. Limits

- **Base rate and the threshold.** Most pairs have no relation. Precision depends on how rare a relation is and on how
  many entities a text has; the best threshold moves from about 0.7 (short passages) to about 0.9 (dense text, Re-DocRED)
  and a fixed one without labelled examples loses 0.05-0.17 F1. Calibrate on a few labelled chunks.
- **Sibling confusion.** Near-synonymous types (for example author, director and developer of a creative work) are
  confused unless the definitions separate them.
- **Label noise in Re-DocRED.** Gold is incomplete. Some "wrong" yes-answers are true relations that the labels omit, so
  precision is understated.
- **Small benchmark.** 182 gold relations (v1) from 50 documents. Benchmark v2 uses 150 documents but is still one
  dataset, one domain (Wikipedia) and 15 types.
- **Cost.** One question per ordered pair and fitting type, so cost grows with pairs times types: a passage with 20
  entities and 12 types is about 1,800 questions, about 35 seconds on an 8 GB GPU. A cheap "any relation" prefilter
  lost gold pairs without a gain.
- **Entity types are fixed.** Entities come from the generator of `relweave-4b-base` (Person, Org, Place, Object,
  Event, Coordinate); new entity types need training.

## 7. Licences and attribution

| component | licence | what it requires |
|---|---|---|
| relweave code and the relweave-4b-zeroshot weights | Apache 2.0, like relweave-4b-base | keep the licence and notices |
| Re-DocRED (training documents, definitions drawn from its relation types) | MIT | keep the copyright notice |
| Wikipedia text (Re-DocRED documents and our business chunks) | CC BY-SA 4.0 | attribution (Wikipedia contributors, https://en.wikipedia.org); share-alike for any redistributed text or data derived from it |
| open-type passages | Re-DocRED / Wikipedia text with labels from a large language model | as Re-DocRED and Wikipedia above |
| Qwen3-4B (base model) | Apache 2.0 | keep the licence and notices |

The weights were trained on Wikipedia passages. Redistributing the training text, or labels derived from it, falls under
CC BY-SA 4.0. GLiREL was used only as a comparison and its data is not part of training; ZeroRel texts were tried in
experiments (Section 5) and are not in the training set of this model.

## 8. How to reproduce

Run from the experiment repository root (`uv run`). The scripts currently live there and will move into the `relweave`
command line. Local runs used one 8 GB GPU; the first run used a Hugging Face Job on an L4 GPU (24 GB).

1. **Baseline, untrained model on benchmark v1.** `uv run python scripts/zs_entail.py` (`N_DOCS=50`).
2. **Train and score on benchmark v1.** `uv run python scripts/zs_qa_train.py all`
   (or `train`, then `eval`). Environment: `OUT` (default `runs/zs_qa`), `N_DOCRED` and `N_OURS` (1500 each; the
   reported run used 600 and 600), `OPEN` (1), `MAX_PROBES` (48), `EPOCHS` (1), `LR` (1e-4), `EVAL_DOCS`,
   `EARLY_DOCS` (10), `EARLY_EVERY` (50). `SMOKE=1` runs a tiny check. The Hugging Face Job wrapper is
   `scripts/hf_zs_qa_job.sh`.
3. **Version 2 training.** `NEG_RATIO=8 DEFS_V2=<definitions file> uv run python scripts/zs_qa_train.py train`.
4. **Benchmark v2.** `ADAPTER=<adapter dir> OUT=<scores.json> uv run python scripts/zs_qa_eval.py`
   (`CAL=50:150`, `TEST=150:300`; `NO_ADAPTER=1` for the plain model; `PREFILTER=<fraction>`).
   Compare two runs with `COMPARE=<other scores.json> uv run python scripts/zs_qa_eval.py report <scores.json>`.

The question format and prompt are defined once, in `question()` and `INSTRUCTION` in `scripts/zs_qa_train.py`;
the evaluation imports them, so training and scoring cannot disagree.

## References

- Boylan, J. et al. (2025). GLiREL: Generalist Model for Zero-Shot Relation Extraction. https://arxiv.org/abs/2501.03172
- Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS. https://arxiv.org/abs/2305.14314
- Hu, E. J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. https://arxiv.org/abs/2106.09685
- Qwen Team (2025). Qwen3 Technical Report. https://arxiv.org/abs/2505.09388
- Sainz, O., Lopez de Lacalle, O., Labaka, G., Barrena, A. and Agirre, E. (2021). Label Verbalization and Entailment for
  Effective Zero and Few-Shot Relation Extraction. EMNLP. https://aclanthology.org/2021.emnlp-main.92/
- Sainz, O. et al. (2024). GoLLIE: Annotation Guidelines Improve Zero-Shot Information-Extraction. ICLR.
  https://arxiv.org/abs/2310.03668
- Tan, Q. et al. (2022). Revisiting DocRED: Addressing the False Negative Problem in Relation Extraction. EMNLP.
  https://arxiv.org/abs/2205.12696
