# Evaluation standards for a new domain

The checklist every result on a new domain should pass before it is reported. Each point says what to do and why,
and links to where the recipe measured it. The commands are in [quickstart-new-domain.md](quickstart-new-domain.md).

## Before any model output

1. **A frozen, unseen test set.** Documents of your real kind that share no people, organisations or facts with
   training, chosen and frozen before you look at any model's output. Eight documents is a minimum, 20-30 is better;
   every score comes with its document-bootstrap interval (`relweave judge score` prints it).
2. **Two blind labelling passes on the test set**, by different labellers (two models, or a model and a person), with
   the same written conventions. Their agreement is the level a model can be expected to reach; report it.
3. **The pass mark, written down first.** For example: judged F1 at least 0.65 and within 0.10 of a single labelling
   pass. A bar set after seeing the numbers is not a test.
4. **One chunking everywhere.** Label, train and run on the same chunks (`relweave label ingest` and `relweave run`
   share `--max-words`, default 200). For dense lists and tables use smaller chunks in all three
   ([recipe-new-domain.md](recipe-new-domain.md), "Dense lists and tables"). An import warning about over-cap chunks
   means re-chunk, not ignore.

## Scoring

5. **Pooled judging, not one label file.** Pool the gold, the second pass and every system you compare; two judges
   rule each claim against the text with a verbatim quote (`relweave judge pool`). Trust the scores only if
   `relweave judge check` says the judges are usable: at most 10% of planted decoys accepted, at most 15% disagreement.
6. **Every compared system in the same pool**, and differences as paired intervals (`relweave judge pair`), not as two
   separate scores.
7. **Per relation type, and without the dominant type** (`relweave judge score --without TYPE`). One frequent, easy
   type can carry the headline (on the government reports 60% of relations, 0.94, while HEADS scored 0.46). A type
   with no relation in the test set is untested: list it as such.
8. **Settings chosen on validation only.** The union cut, chunk size and any threshold are picked on the validation
   set; the test set is scored once with them.

## Reporting

9. **Say what the score cannot see.** Pooled recall counts only relations some source found, so true recall is lower
   by an unmeasured amount; a person labelling two or three test documents completely measures it. When labellers,
   judges and translators are all language models of one family, their shared blind spots are not measured either.
10. **Cost and time with the result**: GPU, hours, money, and which step took the time (generating the head's
    training input was most of the government run).
