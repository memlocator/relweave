# Changelog

## 0.4.1

### Fixed

- **Typed relation decoding on vLLM wrote no relations.** The relation request continued from the entity request's
  tokens including its end-of-turn token, so the model answered after its own turn had ended. Found on a 24 GB card
  (validation relation F1 0.000, entities unaffected); typed decoding on transformers was not affected. Checked after
  the fix on the same 24 GB card at 256 parallel sequences: 10 chunks, 243 relation lines typed against 159 untyped.
- **vLLM on a large card could run out of memory while sampling.** The plan gave vLLM up to 0.95 of the card, but vLLM's
  start-up profile does not count the logits and grammar bitmasks of 256 parallel sequences (about 0.3 GB). The plan
  now keeps 3 MB per parallel sequence out of vLLM's share and caps it at 0.92.
- **A request vLLM could not finish now raises** instead of returning its truncated text as an answer.

## 0.4.0

### Changed

- **The generator runs on vLLM by default** (`engine="auto"`: vLLM when it is installed, else transformers). Same
  results (judged F1 0.719 against 0.702 on 26 unseen documents, paired +0.017, 95% 0.000 to +0.045), about 3x faster
  end to end. The weight format (bf16, fp8, or the 4-bit base with the generator as a LoRA) and parallelism are chosen
  from the card (`relweave.serving.plan`); knobs `--engine`, `--vllm-format`, `--vllm-memory`, `--vllm-max-seqs` on
  `run` and `generate`, the same on `Extractor`. On a large card the pair head stays loaded next to vLLM; on a small one
  vLLM stops after generating. vLLM lives in its own environment (README, "Faster: the generator on vLLM").
- **One base model, adapters attached by name** (`share_base`, default): the generator and head share one 4-bit copy.
  `--separate-models` keeps the old loading.
- **The tagger reads the text once** and keeps only the layers it reads: a 7,000-token document fits 8 GB.
- **The whitepaper lives here** (`docs/whitepaper.md`); `docs/methodology.md` is folded into its appendices.

### Added

- Typed relation decoding, `--typed-relations` on `relweave run` and `generate`, `Extractor(typed_relations=True)`
  (both engines; a head config may save it as `union.typed_relations`): after the entity lines, only relation types whose endpoint
  types fit the listed entities can be written. On court opinions it removed every illegal relation line (35% before)
  and raised judged F1 by 0.02-0.03. `TYPED=1` in `scripts/train_domain.sh`.
- A remote generator (`--generator-url`, e.g. a vLLM docker server) is asked for the LoRA named `generator` when it
  serves one; README examples for merged and LoRA serving.
- `relweave schema`: a schema's types, endpoints and definitions as markdown tables, generated from the classes.
- `relweave train-head --readout label`: a pair head read from the model's own next-token logits (one label token per
  type plus a threshold token), servable as one-token requests; F1 0.823 against 0.838 for the MLP head.
- `relweave run --dense-entities-only` / `--dense-margin`: dense chunks generated as entity lines only, relations from
  the pair head (off by default).
- `scripts/train_domain.sh`: resumes where a run stopped (a step whose output exists is skipped); `ENGINE` and
  `RELWEAVE_GEN` generate on vLLM from its own environment while training stays in the train environment; the
  allocator setting that avoids 8 GB out-of-memory errors applies to training steps only.

### Results

- A new domain through the recipe (US court opinions, whitepaper Section 5.13): judged F1 0.50, level with a Sonnet
  labelling pass, below an Opus pass (0.74) and below the pass mark set in advance (0.65). The model did not exceed
  its labeller; relabelling with the more complete one did not measurably lift it at 29 training documents.

## 0.3.0

### Changed

- **Entity tagger on by default.** When the weights include `tagger/` (relweave-4b-base does from this release), the
  entity tagger (`relweave.tagger`) tags every name in one pass and adds the names the generator left out before the
  pair head scores pairs. Long lists and tables: entity recall 0.78-0.92 -> 0.96-1.00, relation F1 +0.03 to +0.08;
  prose unchanged (26 unseen documents, judged, paired +0.005). Each chunk is read a second time, so extraction is
  slower. `--no-entity-tagger` / `Extractor(entity_tagger=False)` turns it off; `--entity-tagger DIR` uses another.
- **The union's own cap.** The pair head's additions are no longer stopped by the generator's 40-relation cap.
- **Layout-aware chunking.** A paragraph that is mostly table rows, list items or key-value lines is cut into chunks
  of at most `DENSE_WORDS` (80) words at line ends, and each such chunk carries context the model reads first: the
  heading above it and, inside a table, the table's header row and the nearest one-cell group row ("| Ministry of
  Justice |"). Prose keeps 200-word chunks. A chunk stays a verbatim slice of the document (offsets unchanged); the
  model reads `Chunk.model_text`. Labelling (`relweave label ingest`) writes the same text, so training matches run
  time. `chunk(..., dense_words=None)` gives the old behaviour.
- **`relweave label ingest` chunks as `relweave run` does** (200 words, `--max-words`), not 4,000-character windows. The
  old default trained a model on chunks three times the size it reads, and dense chunks went over the per-chunk caps.
  `--max-chars` keeps the old chunking. The chunk size lives in one constant, `relweave.chunk.MAX_WORDS`.
- **`relweave label import` warns about over-cap chunks.** A label with more than 40 entities or relations was skipped
  with a one-line note; it is now counted, and the command ends with a warning saying how many were left out.

### Added

- `relweave train-tagger`: train an entity tagger for your own schema's entity types over your trained head's states
  (it then runs with the head, also when the generator is served by vLLM). Recipe step 7b, quickstart step 5b, and a
  step in scripts/train_domain.sh.

- `--head` (and `Extractor(head=...)`) takes a Hub repository id as well as a directory. A head's `head_config.json`
  may set `chunking.dense_words`, the window size it reads tables and lists in; `--dense-words` overrides it.

- `relweave judge`: pooled judging for any schema. `pool` builds the claims of the gold and each system with planted
  decoys, the judging brief rendered from the schema and a task file for each judge; `check` tests the judges (decoys
  rejected, agreement); `score` gives judged precision, pooled recall and F1 with a document-bootstrap interval, per
  source and per relation type (`--without` the dominant type); `pair` gives a paired difference; `add`/`merge` add a
  system to a judged pool; `export` turns `relweave generate` output (or the union at a cut) into a labels file;
  `gold` does the same for a labelled chunk file.
- `relweave label split`: train, validation and test by document, with `--group` to keep related documents together.
- `relweave label task`: the labelling agent's task for your own schema, domain-free rules plus your conventions.
- `scripts/train_domain.sh`: training and validation of the generator and head in one run, with a timing estimate.
- `docs/quickstart-new-domain.md`: a new domain step by step, every step a command with a check.

## 0.2.0

Results differ from 0.1.x with the same input and settings: the default behaviour changed in the ways listed under
"Changed". Each can be switched back.

### Changed

- **Union cut 0 for heads without a configured cut** (was -2). The pair head vetoes a generated relation it scores at
  or below the cut. On real documents (press releases, news) it removes a third of the false relations (F1 0.624 to
  0.648); on Wikipedia it changes nothing (0.755 to 0.756). A head's own `head_config.json` still decides; the
  published heads (relweave-4b-base, relweave-1.7b-base) now set 0. Pass `cut=-2.0` to `Extractor` for the old setting.
- **Relations of entities never named are left out.** An entity that no chunk calls by a name, only "the company" or
  "she", is usually a back-reference the merge could not link; its relations used to count as wrong twice. Long
  back-reference documents 0.472 to 0.566, others unchanged. `Extractor(keep_unnamed=True)` or `--keep-unnamed` keeps
  them (some are true facts about a thing the text never names).
- **An entity left without relations is re-tested as each other of Person, Org and Place.** The pair head scores its
  pairs again with its type rewritten; if a relation then passes the margin, the entity takes that type. This fixes an
  unfamiliar company read as a person ("Kappahl's sustainability director, Sandra Roos"). `relweave.union.RETEST =
  False` switches it off.
- **The parser retypes an entity when its relations say so** (Person/Org/Place, only when strictly more of its
  relations become legal and none breaks). `relweave.schema.output.RETYPE = False` switches it off.
- **Event is defined as named or pinned to a date** ("the 2023 Halvik Cup final"), not a vague description ("the
  scheme"); PARTICIPATED_IN lists more roles. This changes the schema text used in labelling and conditioned prompts.
- **Markdown table rows are sentences.** A table row (as Word or HTML tables dump to text) is its own sentence for
  evidence and grounding, and `tidy` never joins table lines.

### Added

- `relweave.tidy`: cleans text dumped from PDFs, web pages, Word exports and scans before extraction (hard wraps,
  hyphenation, page numbers, running headers, HTML entities, table-of-contents lines, field codes, footnote numbers).
- `relweave.ontology`: a grouped view of the Person/Org/Place relation types (roles as AFFILIATED_WITH, place relations
  as LOCATED_IN, ...) and `coarsen(graph)`; the official types are unchanged.
- `relweave.tables.verbalize`: table rows rewritten as sentences with their column headers. Not used by default: the
  model reads raw rows better.
- Carry-forward inference (`Extractor(carry=...)`) for generators trained with earlier chunks' entities as input.

### Fixed

- Item unpacking in `auto_batch` and the vLLM generator for carry-forward items.
