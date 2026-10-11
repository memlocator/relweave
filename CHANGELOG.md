# Changelog

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
