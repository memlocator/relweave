# Quickstart: relweave on a new domain

Step by step, for a person or an agent: your own entity and relation types, your own documents, labels made by
language-model agents, training, and scoring by judges. Every step is a command; every step ends with a check. The
reasons behind the steps are in [recipe-new-domain.md](recipe-new-domain.md); the details of each command are in
[recipe-own-schema.md](recipe-own-schema.md). Before you report a result, check it against
[evaluation-standards.md](evaluation-standards.md).

Worked through once on Swedish government reports (new types Inquiry and Report, ten relation types): judged F1 0.92,
level with two labelling passes (recipe-new-domain.md, section 11).

**You need:** 30-50 documents of your kind (more is better), a GPU with 8 GB or more for the generator (24 GB for the
head with 200 probes; or a rented one), an agent that can read and write files (labelling and judging), and Python 3.11
with `pip install "relweave[train]"`.

All commands run in one directory, `mydomain/` below, holding your schema module and your data.

## 1. Write the schema

`mydomain/mydomain.py`. One class per type; the docstring is the definition the model, the labellers and the judges
read. Each definition says where its neighbours begin.

```python
from relweave.schema import Entity, Relation, Schema


class Person(Entity):
    """A human being."""


class Contract(Entity):
    """A named agreement between parties: a supply contract, a licence, a lease. A law is not a Contract."""


class Org(Entity):
    """A company, agency or other organised body."""


class PartyTo(Relation[Org | Person, Contract]):
    """The Org or Person is a party to the Contract; a signatory acting for an Org is SIGNED, not PARTY_TO."""


class Signed(Relation[Person, Contract]):
    """The Person signed the Contract on behalf of a party"""
    role: str | None = None


MYDOMAIN = Schema(name="mydomain", entities=[Person, Contract, Org], relations=[PartyTo, Signed])
```

Check: `relweave judge brief mydomain.py:MYDOMAIN` prints every type with its definition. Read it as a stranger
would: could two people label the same sentence differently? Then sharpen that definition now.

## 2. Texts and split

One `.txt` per document in `texts/`. Text that is not prose (markdown, tables, CMS dumps) is fine; keep table rows one
per line. Text in another language: translate first, keeping names as written (recipe-new-domain.md, section 5).

```bash
relweave label ingest texts --out chunks.jsonl   # chunks as `relweave run` reads them (200 words)
relweave label split chunks.jsonl --schema mydomain --out . --val-frac 0.15 --test-frac 0.2
# documents that belong together (volumes of one report, years of one company): --group '<regex on the document id>'
```

Check: the counts printed for train, val and test; test should have at least 8 documents. Test documents should share
no people or organisations with training documents beyond the unavoidable (recipe-own-schema.md, step 4).

## 3. Labelling conventions

Write `conventions.txt`: how to label the hard cases of your types, in plain sentences with an example each ("A list
of people under 'were appointed experts' gives each one APPOINTED_TO, role expert"; "Where the report was signed gives
no relation"). Label 5 chunks yourself first; every hesitation is a convention to write down.

## 4. Label

For each split (`test` first, then `val`, then `train`):

```bash
relweave label prompts test_unlabelled.jsonl --schema mydomain.py:MYDOMAIN --out prompts_test
relweave label task prompts_test raw_test --conventions conventions.txt --out task_test.md
```

Give `task_test.md` to a labelling agent (split the prompt files across several agents for large sets). Then:

```bash
relweave label import test_unlabelled.jsonl --schema mydomain.py:MYDOMAIN --raw-dir raw_test --out gold_test
```

Check: the import reports few dropped mentions and relations. Read 20 labelled chunks yourself.

The test set gets a second, independent pass by a different agent (or model) with the same task, imported to
`gold_test_b`: two passes show how far labellers agree, which is the level a model can be expected to reach.

## 5. Train and evaluate on validation

```bash
bash path/to/relweave/scripts/train_domain.sh mydomain.py:MYDOMAIN
```

It trains the generator from relweave-4b-base, runs it on validation and test, trains the pair head, and prints the
validation scores of the generator alone and of the union at several cuts (`runs/summary.txt`). It times generation
first and prints how long the head's part will take (on an A10G about 10 s per training chunk; `HEAD=0` skips the head
and halves the run). On a rented GPU (a Hugging Face job, a cloud box), copy the directory there and run the same line.

Check: pick the generator alone or the union at the best cut by **validation**, never by test.

## 6. Score the test set by judges

The label files are one labeller's reading; judges ruling each claim against the text score everyone, labellers
included, on equal terms.

```bash
relweave judge gold gold_test/test.jsonl gold.jsonl --schema mydomain.py:MYDOMAIN
relweave judge gold gold_test_b/test.jsonl passb.jsonl --schema mydomain.py:MYDOMAIN
relweave judge export runs/test.jsonl model.jsonl --chunks gold_test/test.jsonl \
  --scores runs/test_scores.json --cut -4 --margin 0.5 --schema mydomain.py:MYDOMAIN   # omit --scores for the generator alone
relweave judge export runs/base_test.jsonl base.jsonl --schema mydomain.py:MYDOMAIN   # optional, see below
relweave judge pool pool gold=gold.jsonl passb=passb.jsonl model=model.jsonl base=base.jsonl --schema mydomain.py:MYDOMAIN
```

Give `pool/task_a.md` and `pool/task_b.md` to two different judges (two models, or a model and a person). Then:

```bash
relweave judge check pool        # judges usable? decoys rejected, little disagreement
relweave judge score pool --without <your most frequent relation type>
relweave judge pair pool base model   # the gain from training, with a paired interval
```

Check: `judge check` says "judges usable". Read the per-type table, not only the headline: a frequent easy type can
carry the total, and a type with no judged-true relation in the test set is untested. Pooled recall counts only
relations some source found; true recall is lower by an unmeasured amount (a person labelling two or three documents
completely measures it).

**Optional baseline**, to see what training bought: run the untrained generator on the test set,
`relweave generate gold_test/test.jsonl --schema mydomain.py:MYDOMAIN --generator models/base/generator --out runs/base_test.jsonl`,
and include it in the pool as above.

## 7. Use it

```bash
relweave run new_document.txt --out graph.json --generator runs/gen --head runs/head --schema mydomain.py:MYDOMAIN \
  --cut -4   # the cut you chose on validation in step 5
```

When it misses things, read recipe-new-domain.md section 10 before adding data: find where relations are lost first.
