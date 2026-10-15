# Recipe: tuning relweave to a new domain

New entity and relation types, text that is not prose (markdown, wiki and CMS dumps, notes, tables), and longer
documents. This guide is the decision layer on top of [recipe-own-schema.md](recipe-own-schema.md), which has the
commands for writing a schema, labelling, training the generator and the pair head, and evaluating. Read this one first
to decide what to do and in which order; follow that one to do it, or [quickstart-new-domain.md](quickstart-new-domain.md)
for every step as a command.

Everything below is either measured on the business schema (with the number) or marked as not yet measured.

**Status.** recipe-own-schema.md was run end to end on a new schema on 2026-10-14, first on a toy set (its own
`projects` example), then on real text: Swedish government inquiry reports, translated, with new entity types (Inquiry,
Report) and new relation types. The trained model reached the labellers' level on that domain (judged F1 0.92 against
0.91 and 0.93 for two labelling passes); section 11 has the run, the scores per type and what it did not test.

## 1. What changes, and what it costs

| your domain needs | what relweave-4b-base does | what you need |
|---|---|---|
| new relation types between Person, Org and Place | reads only its 21 trained types | labels and training (recipe-own-schema), or the zero-shot adapter for a first look |
| new entity types (products, genes, contracts, ...) | writes only its 6 trained entity types | labels and training; nothing zero-shot covers new entity types |
| non-prose text | trained almost only on prose | text preparation (section 3) and training documents in the same layout |
| long documents | reads 200-word chunks and merges them | usually nothing; see section 6 |

The zero-shot adapter (relweave-4b-zeroshot) reads a relation type's definition at run time: 0.65-0.71 F1 end to end
on new relation types in four domains, 0.50-0.56 on held-out Re-DocRED types. It does not add entity types. Use it to
see whether your relations are readable at all before you label.

## 2. Measure before you train

Most of what we learned is that training on a guess fails and measuring first does not. Build the evaluation first.

1. **An unseen set.** 20-30 documents of your real kind that share no entities or facts with anything you will train
   on, frozen before any model output is seen. Report a document-bootstrap interval with every score: at 26 documents
   it is about +-0.08.
2. **Score by pooled judging, not against one labelling.** If your gold is written by a language model (even two passes
   and an adjudication), it is narrow and partly wrong: on our real documents a frozen gold scored the model at 0.54,
   while two LLM judges checking every relation against the text put it at 0.75, and found a fifth of the gold's own
   relations unstated. Pool the relations of the gold and of every system you compare, have two judges rule on each
   against the text with a verbatim quote, plant decoys (wrong direction, wrong entity, wrong type, true-in-the-world
   but not in the text) and require that the judges reject almost all of them, and score every system in the same
   pool (recipe-own-schema, "When the gold itself is made by a model").
3. **Run the base model first.** On new layouts and genres it tells you how far the text itself is from what the model
   knows. On new types it cannot write them (it was trained on fixed type names; the zero-shot adapter reads new
   relation definitions but adds no entity types); how low it scores there is not yet measured.
4. **Report scores per type, and without the dominant type.** One easy, frequent relation can carry the headline: on
   the government reports, lists of appointed experts were 60% of the relations and scored 0.94-0.98, while HEADS
   scored 0.4-0.5. Type-average F1 and F1 without the dominant type (`relweave judge score --without TYPE`)
   show what the headline hides. A type with no relation in the test set is untested, not passed.

## 3. Text that is not prose

Measured on 12 markdown-style documents (headings, nested bullets, key-value lines, tables, link syntax, dump clutter;
written for the test, business types, 163 relations of which 75% are carried by layout alone), recall by how the
relation is laid out:

| layout | relations | raw | after `relweave.tidy` |
|---|---|---|---|
| sentence | 52 | 0.73 | 0.71 |
| table row | 57 | 0.72 | 0.72 |
| bullet | 22 | 0.68 | 0.77 |
| key-value line ("**CEO:** X") | 32 | 0.59 | 0.69 |
| all (F1 against the written gold) | 163 | 0.73 | 0.74 |

Layout costs a little, mostly in key-value lines, and `tidy` recovers part of it; the model does not fall apart on
markdown. These documents were written to look like dumps; real files may be messier, so measure on yours.

Other measurements:

- **Tables.** One row per line, as Word and HTML tables dump to markdown: relation recall 0.97-1.00 on 4- and 10-row
  tables. Rewriting rows as sentences with their headers ("City: Lyon; Subsidiary: ...") made it much worse (0.20):
  keep the raw row. `relweave.tidy` keeps table rows apart.
- **Lists.** Bulleted and coordinated lists are recalled completely up to 15 items; lists where each item has its own
  clause, and "respectively" pairings, fall to about 0.5 at 10 items.
- **Dump clutter.** `relweave.tidy` removes PDF hard wraps, hyphenation, page numbers, running headers, HTML entities,
  table-of-contents lines, Word field codes and footnote numbers. Use it on any dumped text.
- In prose, relations written before a name without a verb are the model's weakest (titles before a name 0.43,
  modifiers 0.37, possessives 0.45 recall, against 0.68-0.79 for verbs, appositives and prepositions).

## 4. Adding new entity and relation types

New types need labelled data and training; nothing in relweave-4b-base reads a new entity type. The plan that the
dry run tested, with costs for one domain:

| step | what | cost |
|---|---|---|
| 0 | Schema module: every type with a one-line definition that says what it is and what it is not (the nearest other type) | an hour of thought |
| 1 | 30-40 documents of the real kind; set 8 aside, frozen, as the unseen test | collection |
| 2 | Test labels: two blind passes and an adjudication | ~200k annotator tokens |
| 3 | Training labels: ~300 chunks, one pass, and a completeness audit of 40 grouped by how relations are written | ~1M tokens |
| 4 | Train the generator from relweave-4b-base with the conditioned prompt (3 epochs: minutes) | ~$0.30 of GPU |
| 5 | Evaluate the generator alone, judged, per type; add the pair head only when there is enough data that the union beats the generator on validation | ~$0.20 |
| 6 | Weak types: sharper definitions or more labelled examples of the forms they miss, then retrain | ~$0.50 a round |

A model that reads any types at run time, without retraining, is a different project: it needs training across many
schemas (dozens or more), so that the model can only succeed by reading the definitions. Training on one schema with
its types renamed or dropped did not teach that here (a wrong definition moved 2 of 58 decisions).

## 5. Other languages and unusual domains

- **Translate first** when an upstream translation step exists: the English model then needs no new training for
  types it already knows. Keep proper names in the original language in the translation (agencies, laws, inquiry
  names), so the graph's names can be traced back to the source text.
- **Translation rules for extraction.** What the translator does decides what the extractor can find. Instruct it
  (and check a sample) to:
  1. keep proper names exactly as written in the source: people, organisations, agencies, places, names of laws,
     reports and inquiries; no glosses in brackets (a second string makes a second entity);
  2. render every term the same way throughout a document (variants become separate entities);
  3. translate titles and roles, and keep them next to the name as in the source ("director-general Charlotte
     Ingvar-Nilsson");
  4. keep the layout: line breaks, headings, lists, and tables one row per line (raw rows read at 0.97, rewritten
     rows at 0.20);
  5. translate sentence for sentence, so evidence maps back to the source;
  6. add and drop nothing: numbers, dates, footnote marks, and hedges such as "alleged" or "according to" (claims
     read as facts are already a sixth of the model's errors).
  Measured on six Swedish government-report excerpts with a business-schema model, both judges agreeing: names kept,
  41 true of 47 written; names translated, 39 of 47; the Swedish original read directly, 29 of 33. No measured
  difference between keeping and translating names (2 relations on six documents). Rule 1 is kept for traceability
  to the source, not for score. Translating at all gained about 40% more true relations.
- **Train in the language** when the original names must be kept or translation is unavailable: the base model
  (Qwen3) is multilingual, so labelled chunks in that language (route above) are the main cost.
- **Continued pre-training** on unlabelled text of the domain or language (domain-adaptive pre-training) comes last:
  only when the fine-tuned model still misreads the domain's names and terms. It needs millions of tokens of raw text
  and hours of GPU.

## 6. Long documents

relweave reads 200-word chunks with one paragraph of overlap and merges them into one graph per document (entities by
name and mention, relations deduplicated). Measured:

- Longer windows at inference are worse for a model trained on 200-word chunks (role recall 0.42 at 200 words, 0.33
  at 500).
- Few relations depend on crossing a chunk: 4 of 26 missed role relations on our real documents did.
- Relations of things never named ("the company", "she") are dropped by default (`keep_unnamed=True` keeps them).
- A coreference pre-pass helps long back-reference narratives (0.57 to 0.75) and costs on encyclopaedic text (-0.02).

Measured again in a new domain (Swedish government reports, translated; section 11): 11 whole documents of 900-4,700
words, 283 chunks, labelled per document by two blind passes. Only 2 of 226 labelled relations had ends that never
share a chunk (the model missed both); within chunks it found 87-91% of the labelled relations, as on short excerpts.
Judged per document it scored 0.92 (precision 0.96), against 0.97 and 0.99 for the labelling passes, which read each
document whole.

So long documents mostly need nothing beyond the default chunking. Train with chunks of the same size you run with.
Check how many of your relations cross chunks before building anything for them: in both domains it was a few percent.

## 7. Labels

The model learns what the labels leave out. Settle one written convention for every hard case before labelling, run
two blind passes and an adjudication, and audit completeness on a sample grouped by how the text writes each relation
(recipe-own-schema, section 3). For non-prose text, label the relations the layout carries (a heading, list nesting, a
table row, a key-value line), or the model will learn that layout carries none.

## 8. Training

- Start from the trained adapter (`--init-adapter`), one epoch, with a replay of the original training data so the
  model keeps what it has.
- Label completeness matters more than volume. On our real documents, three rounds of adding examples of the missed
  forms, and one round of completing the labels for them in part of the data, did not raise recall: the model writes
  what it already writes more confidently, and its decision to begin another relation line at a missed relation did
  not move (start probability about 0.19 against 0.70 for the lines it writes). The literature's measured remedy is
  completing the labels of the whole training set (Re-DocRED: recall 0.32 to 0.69 on a re-labelled training set).
- Write training examples that look like the text you will run on, in the same layout, with the same conventions.

## 9. Settings at run time

- `cut` (pair head veto of generated relations, default 0): stricter cuts trade recall for precision one for one.
- `margin` (pair head adding relations the generator skipped, default +0.5): 0 adds a few true relations at a small
  precision cost (26 unseen documents: recall 0.63 to 0.66, precision 0.81 to 0.79, F1 +0.013, within noise). Lower it
  when recall matters more than precision.
- `stop_bias` (decoding bias against closing a relation block): no measurable effect once chunks are merged.
- Neither changes what the model has learned to leave out; recall that will not move needs training (section 10).

## 10. When recall will not move

Find out where relations are lost before changing the data: is the entity missing, is the pair in one sentence but not
linked, how is the relation written? Then probe the model's own answer at the place the missing line belongs: if the
first token of the line is unlikely, the model has learned to stop there, and more examples of the same kind will not
change it (recipe-own-schema, section 10).

## 11. Worked example: Swedish government reports (2026-10-14)

The full route on real text, with the pass mark fixed before any model output: judged F1 at least 0.65 and within 0.10
of a single labelling pass.

- **Text.** 40 inquiry reports (SOU) from the Riksdag's open data: the cover letter and the start of the summary,
  translated to English by the translation rules of section 5 (names kept). Split by report, volumes of one report
  together: 23 training, 4 validation, 8 test reports (199 / 33 / 92 chunks).
- **Schema.** Person, Org, Inquiry, Report, Place; CHAIRS, APPOINTED_TO, HEADS, EMPLOYED_BY, UNDER, COMMISSIONED_BY,
  SUBMITS, OWNS, SUBSIDIARY_OF, LOCATED_IN, each definition naming where its neighbours begin.
- **Labels.** By language-model agents with a written convention; the test set twice, blind (the two passes agree at
  F1 0.76); 540 training records (1,758 relations). A completeness audit of 30 training chunks found no false relation
  of 160 and 12 missing.
- **Training and cost.** The generator from relweave-4b-base (`--init-adapter`), 3 epochs: 25 minutes on an A10G. The
  head needs the generator's output on every training record: 95 minutes, most of the run. The whole job took 2 h 50 min
  (about $2.85 on Hugging Face Jobs). Time `relweave generate` on a few records first (about 10 s per chunk here).
- **Generator or union.** Unlike the toy run, the head helped: on validation the union at cut -4 scored 0.95 against
  0.91 for the generator alone, and it held on test (0.92-0.94 against 0.86-0.89 by the label files).
- **Judged result** (test, 92 chunks, pool of both passes and the model, two judges agreeing on 97% of claims, 0 of 42
  decoys accepted): model 0.92 (precision 0.95), pass A 0.91, pass B 0.93. Pass mark met.

| relation | judged-true relations | model F1 |
|---|---|---|
| APPOINTED_TO | 230 | 0.94 |
| EMPLOYED_BY | 102 | 0.94 |
| CHAIRS | 14 | 0.88 |
| SUBMITS | 11 | 0.95 |
| COMMISSIONED_BY | 14 | 0.55 |
| HEADS | 6 | 0.46 |
| UNDER | 3 | 0.00 |
| all but APPOINTED_TO | 150 | 0.88 |

Entities against the label files: Person 0.93, Inquiry 0.76-0.91, Org 0.73-0.86, Report 0.48-0.65.

**What this does not show.**
- The headline rests on lists of appointed experts, a formulaic part of these reports. HEADS, COMMISSIONED_BY and UNDER
  are weak or barely present; OWNS, SUBSIDIARY_OF, LOCATED_IN and the Place type do not occur in the test reports and
  are untested.
- Report is the weak entity type: laws and conventions written as Report although the definition excludes them, and
  the two labelling passes disagreed on whether a page header "SOU 2026:18" is a mention. Sharpen a definition like
  this before labelling, with examples of what is excluded.
- Pooled recall counts only relations some source found; the model's true recall is lower by an unmeasured amount.
  Labellers, judges and translators are all language models of one family, so their shared blind spots are not
  measured either. A human pass over a few documents would measure both.
- The untrained model on this schema cannot be run: relweave refuses a generator on a schema it was not trained on
  ("generator was trained on schema 'business', not 'govreports'"). Extending a schema is not zero-shot; new types
  need training (the zero-shot adapter reads new relation definitions between Person, Org and Place only).
- Whole documents (section 6): judged 0.92 against 0.97-0.99 for labellers reading the whole document; without the
  expert lists 0.80 against 0.93-0.99. The model never wrote UNDER from a cover letter's "To the Minister and Head of
  the Ministry of X" (0 of 10), a form rare in its training chunks.
- The model was trained on labels written with the same convention as test pass A; it reaches the labellers' level on
  their convention, not on a reader's independent one.
