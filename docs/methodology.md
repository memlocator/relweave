# Training methodology of relweave-4b-base

This document describes how `relweave-4b-base` and its library `relweave` were trained and evaluated. It is written for
an ML practitioner who wants to understand, reproduce or adapt the training. The experiments behind it (the full
argument, run-by-run records and the experiment scripts named below) are in the experiment repository the library was
split from, `entity-rel-extraction` (`docs/whitepaper.md`, `docs/findings-2026-09-28-filters.md`); numbers here are
copied from those documents. Commands in this document use the `relweave` command line; scripts marked
"(experiment repository)" exist only there.

## 1. What the model does

Input is a chunk of English business text (about 1,000 characters). Output is a small knowledge graph:

- **Entities.** Each entity has a type and every surface string that refers to it: names, pronouns and descriptions
  ("Inditex", "the company", "it"). Grouping the surface strings of one entity is coreference.
- **Relations.** Each relation has one of 21 types, a source entity, a target entity and a modality (`asserted`,
  `negated`, `hedged` or `reported`). A relation is legal only for certain endpoint types.

The system has two trained parts, which share one 4B base model:

1. **Generator.** Qwen3-4B with a LoRA adapter. It writes the entity lines and a first set of relation lines as text.
2. **Pair-classification head.** A second LoRA adapter on the same base model plus a small MLP. One forward pass reads
   the text, the generated entity lines and one short probe per entity pair. The MLP turns each probe's hidden states
   into one score per relation type and direction, plus a "no relation" threshold score.

**Union rule.** A relation the generator wrote is kept unless the head rejects it firmly (head margin at or below -2).
Every relation the head scores above +0.5 on its own is added, if it is legal for the endpoint types. The two cut-offs
were chosen on the validation set.

The relation decision is taken by the classifier, not by text generation. This is the main design choice: it raised
typed F1 from 0.684 (4B generator alone) to 0.783 on the test set.

## 2. Schema

The schema is written once, as Python classes in `src/relweave/schema/business.py`. The docstring of each class is its
definition. Prompts, decoding grammars, label validation and scoring are derived from it. The tables below are
generated from those classes.

### Entity types (6)

| type | definition |
|---|---|
| PERSON | A human being. |
| ORG | A company, institution, family acting as owner, or other organised body; publications, imprints and brands are Orgs. |
| OBJECT | A physical thing: vehicle, vessel, building, artwork. |
| PLACE | A country, region, city or other location. |
| COORDINATE | A numeric map coordinate. |
| EVENT | Something that happened at a time: a sale, meeting, election, ceremony. |

### Relation types (21)

Edge attributes such as dates, shares and roles exist in the classes but are not part of the scored output. Symmetric
relations have no direction.

| relation | endpoints (source -> target) | definition |
|---|---|---|
| EMPLOYED_BY | Person -> Org | Person works for Org in a non-executive role |
| EXECUTIVE_OF | Person -> Org | Person holds an executive title at Org (CEO, CFO, managing director) or chairs its board or supervisory board (title chairman, vice chairman, honorary chairman) |
| BOARD_MEMBER_OF | Person -> Org | Person sits on the board of Org as a member; a board chair is EXECUTIVE_OF instead |
| MEMBER_OF | Person -> Org | Person is a member of Org (club, party, association) |
| FAMILY_OF | Person -> Person (symmetric) | Family tie; kind is what the target is to the source: spouse, parent, child or sibling |
| ASSOCIATE_OF | Person -> Person (symmetric) | Persons described as associates, partners or close contacts |
| MET_WITH | Person -> Person (symmetric) | Two persons met in person |
| COMMUNICATED_WITH | Person -> Person (symmetric) | Two persons communicated (call, email, message, letter) |
| LOCATED_IN | Object or Org or Person or Place -> Place | Source is or was resident or situated in the Place (dated if the text says when), or a Place lies within another; not a birthplace, not a market; an Org that is based in a Place is HEADQUARTERED_IN |
| BORN_IN | Person -> Place | Person was born in the Place |
| OPERATES_IN | Org -> Place | Org has operations, offices, plants, stores or sales in the Place |
| HEADQUARTERED_IN | Org -> Place | Org has its headquarters in, or is based in, the Place |
| FOUNDED | Org or Person -> Org | Person or Org founded or co-founded the Org; when Orgs merge to form a new Org, each merging Org FOUNDED it |
| OWNS_STAKE_IN | Org or Person -> Org | Person or Org owns shares in, a stake in, or controls through ownership the Org; share is the stated fraction |
| SUBSIDIARY_OF | Org -> Org | Source Org is a subsidiary, division or brand of the target Org; publications, magazines, newspapers, imprints and product brands are Orgs linked this way, never Objects; in a chain of owners, link only to the nearest parent the text states |
| ACQUIRED | Org or Person -> Org | Person or Org bought or took over the target Org; only when the text names the buyer |
| HAS_COORDINATE | Place -> Coordinate | Place has the stated numeric coordinate |
| OWNS_OBJECT | Org or Person -> Object | Person or Org owns the Object (a physical thing: vehicle, vessel, building, artwork); brands and publications are Orgs |
| TRANSFERRED_OBJECT | Event -> Object | The Event transferred the Object (sale, delivery, seizure) |
| PARTICIPATED_IN | Org or Person -> Event | Person or Org took part in the Event; role is buyer, seller, host, attendee |
| HELD_AT | Event -> Place | The Event took place at the Place |

## 3. Data

### Sources

- **Training, validation and test text.** English Wikipedia articles on Nordic and European business, cut into chunks
  of about 1,000 characters. The articles are in the `data/` directories of the experiment repository (see Section 9
  for licence); they are not distributed with the library.
- **Re-DocRED** (3,053 / 500 / 500 Wikipedia documents, 96 Wikidata relation types) was used only for experiments, for
  example pre-training the head's LoRA and the changeable-schema probes. It is not part of the labels of
  `relweave-4b-base`; the pre-training experiment scored -0.007 and was not adopted.

### How labels were made

Labels were annotated by a large language model following written annotation conventions, then spot-checked. Later
rounds corrected model drafts instead of labelling from scratch.

The conventions are the schema definitions in Section 2 plus a few rules that follow from them:

- A list gets one relation per member. If a lead sentence says "acquired A, B, C and D", all four get the relation.
- In a chain of owners, a subsidiary is linked only to its nearest parent that the text states.
- A board chair or honorary chairman is `EXECUTIVE_OF`; an ordinary board seat is `BOARD_MEMBER_OF`.
- A person is `LOCATED_IN` a place only where they live; a birthplace is `BORN_IN`.
- An organisation based in a place is `HEADQUARTERED_IN`, not `LOCATED_IN`.
- Publications, imprints and product brands are Orgs linked by `SUBSIDIARY_OF`, never Objects.
- `ACQUIRED` is written only when the text names the buyer. When Orgs merge into a new Org, each merging Org `FOUNDED` it.

The first training labels were completed by a second pass after an audit showed the first pass missed relations. The test
labels were additionally audited relation by relation.

**Cheaper labelling by correction.** The current system drafts each chunk. An annotator reads the schema once and
returns only corrections: drop, retype, flip or add (`relweave label draft`, then `relweave label correct`). On list-rich text this costs 1,940
tokens per chunk against 6,100 from scratch. Corrected labels agree with independent from-scratch labels at 0.876
typed F1 on ordinary chunks (0.803 on dense list chunks); two independent from-scratch passes agree at about 0.88.
The risk is anchoring: correctors kept some borderline draft relations that a from-scratch labeller would not write.

Wikidata was tried as a source of candidate labels and rejected: its facts cover only 11-12% of gold relations,
because Wikidata records what is true, while the labels record what a passage states.

| split | chunks | articles | labels |
|---|---|---|---|
| training | 2,859 + 500 list-rich chunks | 460 + 210 | completed (new chunks: corrected drafts) |
| validation | 208 | | completed |
| test | 105 | | completed and audited |
| fresh test | 64 | 13 | one pass; scored once |

## 4. Evaluation protocol

- **Split by document.** No document appears in more than one split.
- **No shared entities or facts.** Evaluation text shares no entities or facts with training text, not just no
  documents. Training articles that name three or more evaluation people or companies, or the subject of an evaluation
  article, are excluded. Candidate test articles are checked for relations already present in training
  (`scripts/holdout_overlap.py`, `scripts/fact_overlap.py`, experiment repository).
- **Metric.** Typed relation F1, strict. A predicted relation counts only if both endpoints align to gold entities (by
  mention-span overlap and type) and the type and direction match.
- **Ceiling.** Two independent labelling passes agree at about 0.88. An audit found about 8% of the model's "invented"
  relations to be true, so gold is incomplete.
- **Decisions.** All decisions (thresholds, model choices) are made on validation, with a paired bootstrap over chunks.
  The test set is reported, never tuned on. With 105 chunks one test score is uncertain by about +-0.045.

## 5. Generator training

| setting | value |
|---|---|
| base model | Qwen3-4B, 4-bit (NF4) base, LoRA adapter (QLoRA) |
| LoRA | rank 32, alpha 32, dropout 0.05 |
| optimiser | AdamW 8-bit, cosine schedule, warmup ratio 0.05, weight decay 0.01, gradient accumulation 8 |
| pilot stage | fresh LoRA on 2,000 completed chunks, lr 1.5e-4, 2 epochs (size pilot) |
| main stage | continued from the pilot adapter, 1 epoch on 10,765 completed training chunks (windows of the training articles), lr 1e-4 |
| max sequence length | 3,072 tokens (longest training example 1,251) |
| loss | on the response only |

- **Output format ("sentences").** Compact text lines, not JSON: entity lines with all surface strings, then relation
  lines grouped by the sentence they come from.
- **Compact prompt.** A one-line system prompt without a worked example, three times fewer tokens than the long prompt.
- **Constrained decoding.** A grammar generated from the schema (XGrammar) makes every output parse and every
  relation legal for its endpoint types. A stop-on-repeat check ends repetition loops.

## 6. Pair head training

- **Probes.** After the entity lines, one probe line per entity pair is appended (`P E1 Org E2 Place`). All probes
  are packed into one forward pass. An attention mask lets each probe see the text and the entity lines but not the
  other probes, and all probes share one position id (packed levitated markers).
- **Readout.** The last-token hidden states at layers -1 and -9 are concatenated and fed to an MLP: LayerNorm,
  Linear to 512, GELU, dropout 0.1, Linear to one score per relation type and direction plus a threshold class
  (about 2M parameters).
- **Loss.** The adaptive-threshold loss of ATLOP, plus an auxiliary language-model loss on the entity lines
  (weight 0.5) for examples with gold entity lines.
- **Stage 0.** The head is trained alone on the frozen generator's probe states (experiment repository; `relweave
  train-head` starts a fresh head when no earlier one is given).
- **Stage 1 (joint training).** The head and a second LoRA adapter, initialised from the generator adapter, are trained
  together (`relweave train-head`): lr 1e-4 for the LoRA, 5e-4 for the head, gradient accumulation 8, up to 200
  probes per chunk (all related pairs, then sampled unrelated ones). A head on a frozen model barely helps (0.667
  test); joint training reaches 0.759.
- **Own entity lines.** Half of the examples use the generator's own entity lines for the same chunk, aligned to gold;
  the other half use gold entity lines. This matters because the head reads the generator's lists at run time. Training
  on another model's lists and capping probes at 80 cost 0.024 on validation.
- **Union cut-offs.** Cut -2 and margin +0.5, chosen on validation (grid: cut -12 to -2, margin -1 to +1).

## 7. Results

| system | test F1 | validation F1 |
|---|---|---|
| 1.7B generator, all real data | 0.650 | 0.614 |
| 1.7B generator + 1.7B pair head, union | 0.716 | 0.718 |
| 4B generator alone | 0.684 | 0.650 |
| 4B generator + 4B pair head, union | 0.759 | 0.754 |
| + head trained on the generator's own entity lists, 200 probes | 0.773 | 0.778 |
| **+ 500 list-rich chunks of new labels (relweave-4b-base)** | **0.783** | **0.781** |
| 1.7B, same data and recipe as relweave-4b-base (relweave-1.7b-base) | 0.722 | 0.737 |

Typed relation F1 is strict (Section 4). Test precision 0.818, recall 0.751 for the final system; the union adds
+0.099 over the generator alone (95% interval +0.076 to +0.125).

| evaluation | validation | test |
|---|---|---|
| end to end (generator entities) | 0.781 | 0.783 |
| relation step with gold entities | 0.839 | 0.822 |

On a fresh test set, scored once, the system before the last two steps scored 0.751.

Contribution of each step:

| step | gain |
|---|---|
| pair head instead of generated relations | +0.07 to +0.10 at every model size; invented relations on test 340 to 161 |
| joint training of LoRA and head | 0.667 to 0.759 (test, 4B) |
| head trained on the generator's own entity lists, 200 probes | +0.014 test, +0.024 validation |
| Qwen3-4B instead of 1.7B | +0.05 to +0.07 at equal data, about +0.04 for the full system |
| 500 more list-rich chunks | +0.003 validation, not significant: the learning curve has flattened |

relweave-1.7b-base is the same recipe on Qwen3-1.7B (generator continued from a 2,000-chunk pilot, 1 epoch over all
10,765 chunks; head stage 1 on its own entity lists). Over validation, test and the fresh test set (377 chunks) it
scores 0.732 against the 4B's 0.780: -0.048, 95% interval -0.067 to -0.030.

Serving. The adapters are trained on the 4-bit base, and they learn corrections for its rounding: the same adapter on
the original full-precision Qwen3 loses about 0.03 generator F1 (1.7B, fresh test set: 0.617 on the 4-bit base, 0.583
on the original). The published repositories therefore hold, at their root, the 4-bit base dequantized to bf16 with the
generator adapter merged (`scripts/merge_for_vllm.py`), which reproduces the trained model: on vLLM, 1.7B generator
0.626 and system 0.749 (transformers 0.617 / 0.736); 4B with vLLM's fp8 weights, system 0.771 (transformers 0.770), on
64 chunks.

Measured negative results (filters, loss weighting, DPO-style training on invented lines, grounded decoding, pair-to-pair
attention, a second pass on the same data, Re-DocRED pre-training) are listed in Section 7 of the whitepaper
(experiment repository).

## 8. Known limitations

- **At most 40 entities per chunk.** The output grammar and schema cap a chunk at 40 entities. Long text must be
  chunked. `relweave` chunks documents into whole paragraphs of up to 200 words with one paragraph of overlap, and
  merges entities and relations across chunks.
- **Repetition loops in decoding.** The generator can fill relation blocks until a cap. A stop-on-repeat check
  reduced capped blocks from 109 to 7 on one run, but loops are not eliminated.
- **Business-only schema.** The published weights know the 6 entity types and 21 relation types above. Other types
  need new labels and retraining of both parts (`docs/recipe-own-schema.md`). The changeable-schema head (whitepaper
  Section 5.5) reads unseen types at 0.29 F1 against about 0.80 for trained types and is not part of this model.
- **Coreference is weaker than entity finding.** Proper names are found reliably. On validation 150 of 1,981 gold
  entities (7.6%) are missed: 60% never written (divisions and brands with ordinary-word names), 25% written with
  another type, 15% grouped differently (a company merged with its predecessor).
- **Lists are the biggest error source.** Of 250 missed test relations, 159 sit in long sentences, mostly coordinated
  lists where later items are dropped (375 of 522 list items found). Invented relations mostly link co-occurring
  entities (49 of 161) or entities that share a neighbour (34).
- **Labels and test size.** Labels come from one annotator model under one convention. The test set has 105 chunks.
  All text is English business Wikipedia, cleaner than news or filings.

## 9. Licences and attribution

| component | licence | what it requires |
|---|---|---|
| relweave code and the relweave-4b-base weights | Apache 2.0 | keep the licence and notices |
| Wikipedia article text (training, validation and test chunks) | CC BY-SA 4.0 | attribution (Wikipedia contributors, https://en.wikipedia.org); share-alike for any redistributed text or data derived from it |
| Re-DocRED (experiments only) | MIT | keep the copyright notice |
| Qwen3-4B (base model) | Apache 2.0 | keep the licence and notices |

The weights were trained on English Wikipedia passages. Redistributing the training text, or labels derived from it,
falls under CC BY-SA 4.0. Other data tried in experiments was not used to train `relweave-4b-base`.

## 10. How to reproduce

Install with the training extra (`pip install "relweave[train]"`); every `relweave` command has `--help`. Local runs
used one 8 GB GPU; paid runs used Hugging Face Jobs on an L4 GPU (24 GB). The labelled chunk files are JSONL with
`chunk_id`, `doc_id`, `text`, `gold` and `schema` (see `relweave.training.data`).

1. **Schema, formats, grammars and scoring.** Defined in `src/relweave/schema/`, `src/relweave/formats.py`,
   `src/relweave/generator.py` and `src/relweave/training/eval.py`. Nothing to run.
2. **Data and labels.** Chunk the documents: `relweave label ingest <texts dir> --out chunks.jsonl`. Teacher prompts
   and answers: `relweave label prompts chunks.jsonl --out prompts/`, answers following `relweave label guide`, then
   `relweave label import chunks.jsonl --raw-dir raw/ --out labelled/`. Correction labelling: `relweave label draft`
   then `relweave label correct`. Chunk selection, leakage checks and the list audit are experiment scripts
   (`scripts/select_chunks.py`, `scripts/holdout_overlap.py`, `scripts/fact_overlap.py`, `scripts/list_audit.py`,
   experiment repository).
3. **Generator.**
   `relweave train-generator <pilot chunks> --out runs/pilot --lr 1.5e-4 --epochs 2` (rank 32 and the compact prompt
   are the defaults), then
   `relweave train-generator <training chunks> --out runs/gen --init-adapter runs/pilot/adapter --lr 1e-4 --epochs 1`.
4. **Generator answers** on validation, test and training chunks:
   `relweave generate <chunks.jsonl> --generator runs/gen --out gen_<set>.jsonl`.
5. **Pair head.** `relweave train-head <training chunks> --generator runs/gen --own-results gen_train.jsonl --max-probes 200 --out runs/head`
   (the published head continued from a stage-0 head: `--base-head`).
6. **Score and union.** `relweave score <chunks.jsonl> gen_<set>.jsonl --generator runs/gen --head runs/head --out scores_<set>.json`,
   then `relweave eval <chunks.jsonl> gen_<set>.jsonl --scores scores_<set>.json` (generator alone and union; pick
   `--cut` and `--margin` on validation, report test once).
7. **Relation step with gold entities, error analysis.** `scripts/gold_entity_eval.py`, `scripts/error_modes.py`,
   `scripts/diagnostics.py` (experiment repository).

The library entry point is `relweave.Extractor` (`relweave run` on the command line): chunking, generator, head and
union per chunk, then the merge across chunks.

## References

- Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS. https://arxiv.org/abs/2305.14314
- Dong, Y. et al. (2024). XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models.
  https://arxiv.org/abs/2411.15100
- Hu, E. J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. https://arxiv.org/abs/2106.09685
- Qwen Team (2025). Qwen3 Technical Report. https://arxiv.org/abs/2505.09388
- Tan, Q. et al. (2022). Revisiting DocRED: Addressing the False Negative Problem in Relation Extraction. EMNLP.
  https://arxiv.org/abs/2205.12696
- Ye, D. et al. (2022). Packed Levitated Marker for Entity and Relation Extraction. ACL. https://arxiv.org/abs/2109.06067
- Zhou, W. et al. (2021). Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling.
  AAAI. https://arxiv.org/abs/2010.11304
