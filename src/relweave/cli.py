"""relweave command line.

  relweave run <file.txt> --out graph.json          text to a graph (JSON Graph Format v2)
  relweave label ingest|prompts|import|draft|correct  build labelled chunk files for your own schema
  relweave generate / score                         run the generator / the pair head over a chunk file
  relweave eval                                     score saved results against gold
  relweave train-generator / train-head / train-tagger   train the parts (pip install "relweave[train]")
  relweave judge pool|check|score|...                pooled judging: score systems against the text
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from relweave.chunk import DENSE_WORDS, MAX_WORDS
from relweave.zeroshot import DEFAULT_THRESHOLD

from relweave.serving import DEFAULT_ENGINE  # noqa: E402  (light: no torch at import)

ENGINE_HELP = "auto (vLLM when installed, else transformers), vllm (the generator on vLLM in this process; vllm and vllm-bnb-plugin installed next to relweave) or transformers"
VLLM_FORMAT_HELP = "vLLM weight format: auto (the most precise that fits), bf16, fp8 or 4bit"
VLLM_MEMORY_HELP = "vLLM's share of the GPU (default: what the card allows)"
VLLM_SEQS_HELP = "sequences vLLM generates at once (default: as many as fit)"

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def _root() -> None:
    """relweave: text to a typed graph."""


def _batch(value: str) -> int | str:
    if value == "auto":
        return value
    if not value.isdigit() or int(value) < 1:
        raise typer.BadParameter("a positive integer or 'auto'")
    return int(value)


@app.command()
def run(files: list[Path] = typer.Argument(..., help="text files"),
        out: Path = typer.Option(..., help="graph JSON (JSON Graph Format v2) for one file; a directory for several"),
        graphml: bool = typer.Option(False, help="also write GraphML next to each JSON"),
        no_text: bool = typer.Option(False, help="leave the document text out of the JSON"),
        weights: str | None = typer.Option(None, help="Hub repository or local directory with generator/ and head/ "
                                           "(default chrullis/relweave-4b-base)"),
        schema: str | None = typer.Option(None, help="schema: a registered name or path/to/module.py:NAME "
                                          "(default: the one the generator was trained for)"),
        model: str | None = typer.Option(None, help="base model path or hub id"),
        generator: Path | None = typer.Option(None, help="generator directory, overriding the one in --weights"),
        head: str | None = typer.Option(None, help="pair head: a directory or a Hub repository id (e.g. a table head), overriding the one in --weights"),
        cut: float | None = typer.Option(None, help="drop a generator relation the head scores at or below this "
                                         "(default from the head's config, -2.0)"),
        margin: float | None = typer.Option(None, help="add a head relation scoring above this (default from the "
                                            "head's config, +0.5)"),
        batch_size: str = typer.Option("auto", help="chunks generated together: 1, 2, 4, 8 ... or auto"),
        max_words: int = typer.Option(MAX_WORDS, help="words per chunk"),
        entity_tagger: Path | None = typer.Option(None, help="entity tagger directory (tagger.pt, tagger_config.json); default: the weights' own tagger/ when present. It adds the names the generator did not list before the relation head scores pairs"),
        separate_models: bool = typer.Option(False, help="load the base model twice (generator and head) instead of once with both adapters"),
        dense_entities_only: bool | None = typer.Option(None, "--dense-entities-only/--dense-full",
                                                        help="on list, table and key-value chunks the generator writes "
                                                             "entity lines only and the pair head every relation "
                                                             "(default: the head's saved setting, else off)"),
        dense_margin: float | None = typer.Option(None, help="head margin on those chunks (default: the head's saved "
                                                             "setting, else --margin)"),
        engine: str = typer.Option(DEFAULT_ENGINE, help=ENGINE_HELP),
        vllm_format: str = typer.Option("auto", help=VLLM_FORMAT_HELP),
        vllm_memory: float | None = typer.Option(None, help=VLLM_MEMORY_HELP),
        vllm_max_seqs: int | None = typer.Option(None, help=VLLM_SEQS_HELP),
        no_entity_tagger: bool = typer.Option(False, help="run without the entity tagger (faster; long lists and tables lose names)"),
        dense_words: int | None = typer.Option(None, help="words per chunk for tables and lists (default: the head's own setting, else 80; 0: chunk them as prose)"),
        reserve_mb: int = typer.Option(500, help="GPU memory left free"),
        generator_url: str | None = typer.Option(None, help="generate on this HTTP server (vLLM serving the merged "
                                                 "generator, e.g. http://localhost:8000); the head runs locally"),
        zeroshot: bool = typer.Option(False, help="score the relation types of --schema with the zero-shot adapter "
                                      "(types defined at run time; entities still come from the generator)"),
        threshold: float | None = typer.Option(None, help=f"zero-shot: keep relations whose density-adjusted score is above this (default {DEFAULT_THRESHOLD}; "
                                        "scores are adjusted for the number of entity pairs per chunk)"),
        zeroshot_weights: str | None = typer.Option(None, help="zero-shot adapter (default chrullis/relweave-4b-zeroshot)"),
        keep_unnamed: bool = typer.Option(False, help="keep relations of entities never named, only described "
                                          "(\"the business\", \"she\"); off by default: most are unlinked back-references")) -> None:
    """Extract a graph from each file; all files go through one Extractor, so each model loads once."""
    import relweave
    try:
        ex = relweave.Extractor(weights=weights, schema=schema, model=model, generator=generator, head=head, cut=cut,
                                margin=margin, batch_size=_batch(batch_size), max_words=max_words, dense_words=dense_words, entity_tagger=False if no_entity_tagger else entity_tagger, share_base=not separate_models, reserve_mb=reserve_mb,
                                dense_entities_only=dense_entities_only, dense_margin=dense_margin,
                                engine=engine, vllm_format=vllm_format, vllm_memory=vllm_memory,
                                vllm_max_seqs=vllm_max_seqs,
                                generator_url=generator_url, zeroshot=zeroshot, threshold=threshold,
                                zeroshot_weights=zeroshot_weights, keep_unnamed=keep_unnamed)
        many = len(files) > 1
        if many:
            out.mkdir(parents=True, exist_ok=True)
        for k, g in ex.iter_run([f.read_text() for f in files], [f.stem for f in files]):
            path = out / f"{files[k].stem}.json" if many or out.is_dir() else out
            path.write_text(g.to_json(include_text=not no_text))
            if graphml:
                g.to_graphml(path.with_suffix(".graphml"))
            typer.echo(f"{len(g.entities)} entities, {len(g.relations)} relations, {len(g.warnings)} warnings -> {path}")
            for w in g.warnings:
                typer.echo(f"warning: {w}", err=True)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e


# ------------------------------------------------------------------------------------------------------------
# Labelling, running over chunk files, evaluation and training
# ------------------------------------------------------------------------------------------------------------

label_app = typer.Typer(add_completion=False, no_args_is_help=True,
                        help="Labelled chunk files: chunk texts, teacher prompts and answers, draft correction.")
app.add_typer(label_app, name="label")

SCHEMA_HELP = "schema: a registered name or path/to/module.py:NAME"




def _schema(spec: str | None):
    """Load (and register) the schema, so chunk records naming it resolve; None when not given."""
    if spec is None:
        return None
    from relweave.schema import load_schema
    try:
        return load_schema(spec)
    except (ValueError, TypeError, AttributeError, ImportError, FileNotFoundError) as e:
        raise typer.BadParameter(str(e)) from e


def _parts(weights: str | None, generator: Path | None, head: Path | None, need_head: bool) -> tuple[Path, Path | None]:
    from relweave.extract import DEFAULT_WEIGHTS, resolve_weights
    if generator is None or (need_head and head is None):
        root = resolve_weights(weights or DEFAULT_WEIGHTS)
        generator, head = generator or root / "generator", head or root / "head"
    return generator, head


@label_app.command("ingest")
def label_ingest(texts_dir: Path = typer.Argument(..., help="directory of plain-text documents (*.txt)"),
                 out: Path = typer.Option(..., help="unlabelled chunk file (JSONL)"),
                 test_frac: float = typer.Option(0.2, help="share of documents in the test split"),
                 seed: int = 0,
                 max_words: int = typer.Option(MAX_WORDS, help="words per chunk, as `relweave run` reads them (keep the "
                                                               "two equal: a model reads best what it was trained on)"),
                 max_chars: int | None = typer.Option(None, help="instead of --max-words: sentence windows of this "
                                                                "many characters (the old chunking)"),
                 overlap_sentences: int = 2) -> None:
    """Chunk a directory of documents into unlabelled chunks with a train/test split by document, by default exactly
    as `relweave run` chunks them."""
    from relweave.training.data import ingest
    files = sorted(texts_dir.glob("*.txt"))
    if not files:
        raise typer.BadParameter(f"no *.txt files in {texts_dir}")
    n, n_test = ingest(files, out, test_frac, seed, max_chars, overlap_sentences, max_words)
    typer.echo(f"wrote {n} chunks from {len(files)} documents ({n_test} test docs) -> {out}")


@label_app.command("split")
def label_split(chunks: Path = typer.Argument(..., help="chunk file from `relweave label ingest`"),
                schema: str = typer.Option(..., help="schema name the records are tagged with (the NAME of module.py:NAME)"),
                out: Path = typer.Option(..., help="directory for train_, val_ and test_unlabelled.jsonl"),
                val_frac: float = 0.15, test_frac: float = 0.2, seed: int = 0,
                group: str | None = typer.Option(None, help="regex on the document id: its match names a group kept "
                                                           "on one side (e.g. '^[^_]+' for report_a, report_b)")) -> None:
    """Split chunks into train, validation and test by document (or document group), tagged with the schema."""
    from relweave.training.data import split
    try:
        r = split(chunks, out, schema, val_frac, test_frac, seed, group)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    for k, (d, c) in r.items():
        typer.echo(f"{k:5} {d:4} documents {c:5} chunks -> {out}/{k}_unlabelled.jsonl")


@label_app.command("prompts")
def label_prompts(chunks: Path = typer.Argument(..., help="chunk file"),
                  out: Path = typer.Option(..., help="directory for one prompt file per chunk"),
                  schema: str = typer.Option("business", help=SCHEMA_HELP)) -> None:
    """One teacher prompt per chunk (schema text rendered from the classes); answer each into <raw dir>/<name>.txt
    following the guide `relweave label guide` prints."""
    from relweave.training.labels import write_prompts
    sch = _schema(schema)
    recs = [json.loads(line) for line in chunks.read_text().splitlines() if line.strip()]
    paths = write_prompts([(d["chunk_id"], d["text"]) for d in recs], out, schema=sch)
    typer.echo(f"wrote {len(paths)} prompts to {out}")


@label_app.command("guide")
def label_guide() -> None:
    """Print the labelling guide (prompt for an agent answering the prompt files, and the conventions)."""
    from relweave.training.labels import conventions_file
    typer.echo(conventions_file())


@label_app.command("task")
def label_task(prompt_dir: Path = typer.Argument(..., help="prompt directory from `relweave label prompts`"),
               raw_dir: Path = typer.Argument(..., help="directory the labeller writes its answers to"),
               conventions: Path | None = typer.Option(None, help="your domain's labelling conventions (text file)"),
               out: Path | None = typer.Option(None, help="write the task here instead of printing it")) -> None:
    """The task for a labelling agent on your own schema: domain-free rules plus your conventions."""
    from relweave.training.labels import agent_task
    n = len(list(prompt_dir.glob("*.json")))
    if n == 0:
        raise typer.BadParameter(f"no prompt files in {prompt_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    t = agent_task(prompt_dir, raw_dir, n, conventions.read_text() if conventions else "")
    if out:
        out.write_text(t)
        typer.echo(f"task for {n} prompt files -> {out}")
    else:
        typer.echo(t)


@label_app.command("import")
def label_import(chunks: Path = typer.Argument(..., help="the chunk file the prompts were written from"),
                 raw_dir: Path = typer.Option(..., help="directory with one <chunk>.txt answer per chunk"),
                 out: Path = typer.Option(..., help="directory for train.jsonl and test.jsonl"),
                 schema: str = typer.Option("business", help=SCHEMA_HELP),
                 train_windows: int = typer.Option(500, help="also add training chunks as sentence windows of this "
                                                   "many characters (0: none)")) -> None:
    """Teacher answers to gold chunk files; anything unverifiable is dropped and counted."""
    from relweave.training.labels import import_labels
    c = import_labels(chunks, raw_dir, out, _schema(schema), train_windows, echo=typer.echo)
    typer.echo(f"missing answers: {c['missing']}; unverifiable items dropped from labels: {c['dropped']}; "
               f"chunks skipped: {c['skipped']}")
    if c["over_cap"]:
        typer.echo(f"WARNING: {c['over_cap']} chunks hold more entities or relations than the model writes per chunk "
                   f"and were left out: chunk smaller (`relweave label ingest --max-words`, and the same --max-words "
                   f"at run time), then label again", err=True)


@label_app.command("draft")
def label_draft(chunks: Path = typer.Argument(..., help="chunk file"),
                results: Path = typer.Argument(..., help="generator results on it (relweave generate)"),
                scores: Path = typer.Argument(..., help="head scores on them (relweave score)"),
                out: Path = typer.Option(..., help="directory for instructions.md and batch files"),
                n: int = typer.Option(50, help="chunks to draft"), per_batch: int = 25,
                cut: float = typer.Option(-2.0, help="union cut for the drafts"),
                margin: float = typer.Option(1.0, help="union margin for the drafts"),
                schema: str | None = typer.Option(None, help=SCHEMA_HELP + " (when the chunks name your own)")) -> None:
    """Drafts (the generator and head union) for an agent to correct in short change lines."""
    from relweave.training.correct import export
    _schema(schema)
    export(out, chunks, results, scores, None, n, per_batch, cut, margin)


@label_app.command("correct")
def label_correct(out: Path = typer.Argument(..., help="the draft directory, with answers_NN.txt"),
                  chunks: Path = typer.Argument(..., help="the chunk file the drafts were made from"),
                  schema: str | None = typer.Option(None, help=SCHEMA_HELP)) -> None:
    """Apply the agents' change lines to the drafts -> <out>/labels.jsonl (and agreement with existing gold)."""
    from relweave.training.correct import merge
    _schema(schema)
    merge(out, chunks)


@app.command()
def generate(chunks: Path = typer.Argument(..., help="chunk file"),
             out: Path = typer.Option(..., help="results file (JSONL)"),
             weights: str | None = typer.Option(None, help="Hub repository or directory with generator/"),
             generator: Path | None = typer.Option(None, help="generator directory (overrides --weights)"),
             model: str | None = typer.Option(None, help="base model path or hub id"),
             schema: str | None = typer.Option(None, help=SCHEMA_HELP),
             batch_size: str = typer.Option("auto", help="1, 2, 4, 8 ... or auto"),
             limit: int | None = None, reserve_mb: int = 500,
             engine: str = typer.Option(DEFAULT_ENGINE, help=ENGINE_HELP),
             vllm_format: str = typer.Option("auto", help=VLLM_FORMAT_HELP),
             vllm_memory: float | None = typer.Option(None, help=VLLM_MEMORY_HELP),
             vllm_max_seqs: int | None = typer.Option(None, help=VLLM_SEQS_HELP)) -> None:
    """The generator's results on every chunk of a chunk file."""
    from relweave.training.predict import generate as run
    gen, _ = _parts(weights, generator, None, need_head=False)
    n = run(chunks, out, gen, model, _schema(schema), _batch(batch_size), reserve_mb, limit, engine=engine,
            vllm_format=vllm_format, vllm_memory=vllm_memory, vllm_max_seqs=vllm_max_seqs)
    typer.echo(f"{n} results -> {out}")


@app.command("schema")
def schema_cmd(schema: str | None = typer.Option(None, help=SCHEMA_HELP + " (default the business schema)")) -> None:
    """Print a schema's entity and relation types, endpoints and definitions as markdown tables."""
    from relweave.schema import BUSINESS
    typer.echo((_schema(schema) or BUSINESS).markdown(), nl=False)


@app.command()
def score(chunks: Path = typer.Argument(..., help="chunk file"),
          results: Path = typer.Argument(..., help="generator results (relweave generate)"),
          out: Path = typer.Option(..., help="scores file (JSON)"),
          weights: str | None = typer.Option(None, help="Hub repository or directory with generator/ and head/"),
          generator: Path | None = typer.Option(None, help="generator directory (its prompt settings)"),
          head: Path | None = typer.Option(None, help="pair head directory"),
          model: str | None = typer.Option(None, help="base model path or hub id"),
          schema: str | None = typer.Option(None, help=SCHEMA_HELP), reserve_mb: int = 500) -> None:
    """The pair head's score for every entity pair of the generator's results."""
    from relweave.training.predict import score as run
    gen, hd = _parts(weights, generator, head, need_head=True)
    n = run(chunks, results, out, gen, hd, model, _schema(schema), reserve_mb)
    typer.echo(f"{n} chunks scored -> {out}")


@app.command("eval")
def eval_cmd(chunks: Path = typer.Argument(..., help="labelled chunk file (gold)"),
             results: Path = typer.Argument(..., help="generator results on it"),
             scores: Path | None = typer.Option(None, help="head scores: also evaluate the union"),
             cut: float | None = typer.Option(None, help="union cut (default -2.0)"),
             margin: float | None = typer.Option(None, help="union margin (default +0.5)"),
             by_type: bool = typer.Option(False, help="relation F1 per relation type"),
             schema: str | None = typer.Option(None, help=SCHEMA_HELP),
             out: Path | None = typer.Option(None, help="also write the metrics as JSON")) -> None:
    """Entity, coreference and relation scores of saved results, each chunk under its own schema."""
    from relweave.training.eval import evaluate
    _schema(schema)
    m = evaluate(chunks, results, scores, cut, margin, by_type)
    for name, s in m.items():
        rel = s["rel_strict"]
        typer.echo(f"{name:9} relation F1 {rel['f1']:.3f} (P {rel['precision']:.3f} R {rel['recall']:.3f})  "
                   f"entity F1 {s['ent_strict']['f1']:.3f}  chunks {s['chunks']}")
        for t, v in s.get("rel_by_type", {}).items():
            typer.echo(f"          {t:28} F1 {v['f1']:.3f}  tp {v['tp']} fp {v['fp']} fn {v['fn']}")
    if out:
        out.write_text(json.dumps(m, indent=2) + "\n")


@app.command("train-generator")
def train_generator(chunks: Path = typer.Argument(..., help="labelled training chunks"),
                    out: Path = typer.Option(..., help="run directory: adapter/ and train_summary.json"),
                    model: str | None = typer.Option(None, help="base model (default the 4-bit Qwen3-4B)"),
                    schema: str | None = typer.Option(None, help=SCHEMA_HELP + " (when the chunks name your own)"),
                    epochs: int = 2, lr: float = 1e-4, r: int = typer.Option(32, help="LoRA rank"),
                    grad_accum: int = 8, max_seq_length: int = 3072, seed: int = 0,
                    compact_prompt: bool = typer.Option(True, help="one-line system prompt, no worked example"),
                    conditioned: bool = typer.Option(False, help="put the chunk's rendered schema in the system prompt"),
                    schema_variants: bool = typer.Option(False, help="a random schema variant per example"),
                    held_out: Path | None = typer.Option(None, help="JSON {relations, entities, schemas} left out"),
                    init_adapter: Path | None = typer.Option(None, help="continue from this adapter directory"),
                    fast: bool = typer.Option(False, help="large GPU: bf16 base, no gradient checkpointing"),
                    epoch_adapters: bool = typer.Option(False, help="also save the adapter after every epoch "
                                                        "(adapter_epoch<n>), to pick the best on validation"),
                    save_steps: int = 0, resume: bool = False) -> None:
    """QLoRA fine-tuning of the generator (needs the train extra)."""
    from relweave.extract import DEFAULT_MODEL
    from relweave.training.generator import train
    _schema(schema)
    summary = train(chunks, out, model_id=model or DEFAULT_MODEL, max_seq_length=max_seq_length, epochs=epochs,
                    lr=lr, r=r, grad_accum=grad_accum, seed=seed, compact=compact_prompt, conditioned=conditioned,
                    schema_variants=schema_variants, held_out=json.loads(held_out.read_text()) if held_out else None,
                    init_adapter=init_adapter, fast=fast, save_steps=save_steps, resume=resume,
                    epoch_adapters=epoch_adapters)
    typer.echo(json.dumps(summary, indent=2))


@app.command("train-head")
def train_head_cmd(chunks: list[Path] = typer.Argument(..., help="labelled training chunk files"),
                   generator: Path = typer.Option(..., help="generator directory (its adapter is the start)"),
                   out: Path = typer.Option(..., help="head directory: adapter/, head.safetensors, head_config.json"),
                   own_results: list[Path] = typer.Option([], help="the generator's results on the training chunks "
                                                          "(relweave generate); repeat for several"),
                   base_head: Path | None = typer.Option(None, help="continue from this head (same label set)"),
                   model: str | None = typer.Option(None, help="base model (default the 4-bit Qwen3-4B)"),
                   schema: str | None = typer.Option(None, help=SCHEMA_HELP + " (default the chunks' own)"),
                   examples: int = typer.Option(2400, help="training examples (chunks)"),
                   max_probes: int = 150, seed: int = 0, list_repeat: int = 0,
                   upcast: bool = typer.Option(False, help="PEFT's fp32 copies of embeddings and lm_head (off: fits "
                                               "a 4B model on 8 GB)"),
                   reserve_mb: int = 700,
                   readout: str = typer.Option("mlp", help="mlp (an MLP on hidden states) or label (the model's own "
                                               "next-token logits for one token per label: servable by vLLM)")) -> None:
    """Joint training of the pair head and its adapter (needs the train extra)."""
    from relweave.extract import DEFAULT_MODEL
    from relweave.training.head import train_head
    summary = train_head(chunks, out, generator, model or DEFAULT_MODEL, own_results, base_head, _schema(schema),
                         examples, max_probes, seed=seed, list_repeat=list_repeat, upcast=upcast, reserve_mb=reserve_mb,
                         readout=readout)
    typer.echo(json.dumps({k: v for k, v in summary.items() if k != "labels"}, indent=1))


@app.command("train-tagger")
def train_tagger_cmd(chunks: list[Path] = typer.Argument(..., help="labelled training chunk files"),
                     head: Path = typer.Option(..., help="your trained pair head (its states are read, so the tagger "
                                                         "runs with the head, also when the generator is served)"),
                     out: Path = typer.Option(..., help="tagger directory: tagger.pt, tagger_config.json"),
                     schema: str | None = typer.Option(None, help=SCHEMA_HELP + " (its entity types are tagged; "
                                                                  "default the business schema)"),
                     types: str | None = typer.Option(None, help="comma-separated subset of the schema's entity types"),
                     model: str | None = typer.Option(None, help="base model (default the 4-bit Qwen3-4B)"),
                     epochs: int = 2, layers: str = typer.Option("-1,-14", help="hidden-state layers read"),
                     echo_input: bool = typer.Option(False, "--echo-input", help="feed the text twice and tag the "
                                                     "second copy (the pre-0.3.1 taggers; no better, half the speed)"),
                     seed: int = 0) -> None:
    """Train an entity tagger: tags every name in one pass, so names a long list or table would lose still reach
    the pair head (needs the train extra; about 12 minutes for 2,700 chunks on an 8 GB GPU)."""
    from relweave.extract import DEFAULT_MODEL
    from relweave.schema import BUSINESS
    from relweave.training.tagger import train_tagger
    sch = _schema(schema) or BUSINESS
    all_types = list(sch.node_names())
    chosen = [t.strip() for t in types.split(",")] if types else all_types
    unknown = sorted(set(chosen) - set(all_types))
    if unknown:
        raise typer.BadParameter(f"types {unknown} are not in schema {sch.name} ({all_types})")
    cfg = train_tagger(chunks, out, head, model or DEFAULT_MODEL, chosen, reads="head", epochs=epochs,
                       layers=[int(x) for x in layers.split(",")], seed=seed, echo=typer.echo, double=echo_input)
    typer.echo(f"tagger for {', '.join(cfg['types'])} -> {out}; use it with: relweave run ... --head {head} "
               f"--entity-tagger {out}")


# ------------------------------------------------------------------------------------------------------------
# Pooled judging (relweave.judge)
# ------------------------------------------------------------------------------------------------------------

judge_app = typer.Typer(add_completion=False, no_args_is_help=True,
                        help="Pooled judging: score systems by judges ruling each claim against the text.")
app.add_typer(judge_app, name="judge")


def _judges(judges: str) -> list[str]:
    return [j for j in judges.split(",") if j]


@judge_app.command("export")
def judge_export(results: Path = typer.Argument(..., help="`relweave generate` output"),
                 out: Path = typer.Argument(..., help="labels file to write"),
                 chunks: Path | None = typer.Option(None, help="the chunk file generated on (adds text; needed for --scores)"),
                 scores: Path | None = typer.Option(None, help="head scores: export the union instead of the generator"),
                 cut: float | None = typer.Option(None, help="union cut"),
                 margin: float | None = typer.Option(None, help="union margin"),
                 schema: str | None = typer.Option(None, help=SCHEMA_HELP)) -> None:
    """A system's output as a labels file for a pool."""
    from relweave import judge
    _schema(schema)
    if scores is not None and chunks is None:
        raise typer.BadParameter("--scores needs --chunks")
    typer.echo(f"{judge.export(results, out, chunks, scores, cut, margin)} relations -> {out}")


@judge_app.command("graphs")
def judge_graphs(graphs: list[Path] = typer.Argument(..., help="graph JSON files from `relweave run` (one per document)"),
                 out: Path = typer.Option(..., help="labels file to write")) -> None:
    """Whole-document graphs from `relweave run` as a labels file for a pool (long documents, scored per document)."""
    from relweave import judge
    typer.echo(f"{judge.labels_from_graphs(graphs, out)} relations from {len(graphs)} documents -> {out}")


@judge_app.command("gold")
def judge_gold(chunks: Path = typer.Argument(..., help="labelled chunk file"),
               out: Path = typer.Argument(..., help="labels file to write"),
               schema: str | None = typer.Option(None, help=SCHEMA_HELP)) -> None:
    """A labelled chunk file as a labels file with text (a pool's gold)."""
    from relweave import judge
    _schema(schema)
    typer.echo(f"{judge.labels_from_chunks(chunks, out)} relations -> {out}")


@judge_app.command("pool")
def judge_pool(out: Path = typer.Argument(..., help="pool directory"),
               labels: list[str] = typer.Argument(..., help="NAME=labels.jsonl for the gold (named gold, records with "
                                                            "text) and each system"),
               schema: str | None = typer.Option(None, help=SCHEMA_HELP + " (decoy types and the brief)"),
               decoys: int = typer.Option(2, help="decoys per document")) -> None:
    """Build a pool, the judging brief and the judges' task: OUT/items.jsonl, key.jsonl, brief.md, task_<judge>.md."""
    from relweave import judge
    s = _schema(schema)
    pairs = {}
    for x in labels:
        if "=" not in x:
            raise typer.BadParameter(f"{x}: expected NAME=path")
        n, p = x.split("=", 1)
        pairs[n] = p
    docs, claims = judge.items(out, pairs, s, decoys=decoys)
    if s is not None:
        (out / "brief.md").write_text(judge.brief(s))
    for j in ("a", "b"):
        (out / f"task_{j}.md").write_text(judge.judge_prompt(out / "brief.md", out / "items.jsonl",
                                                             out / f"verdicts_{j}.jsonl"))
    typer.echo(f"{docs} documents, {claims} claims -> {out}/items.jsonl\n"
               f"Give {out}/task_a.md and {out}/task_b.md to two different judges (two models, or a model and a "
               f"person), then: relweave judge check {out}")


@judge_app.command("brief")
def judge_brief(schema: str = typer.Argument(..., help=SCHEMA_HELP)) -> None:
    """Print the judging brief of a schema."""
    from relweave import judge
    typer.echo(judge.brief(_schema(schema)))


@judge_app.command("check")
def judge_check(out: Path = typer.Argument(..., help="pool directory"),
                judges: str = typer.Option("a,b", help="judge names, comma-separated (verdicts_<name>.jsonl)")) -> None:
    """Decoys accepted per judge and the judges' disagreement: trust the scores only if both are low."""
    from relweave import judge
    r = judge.check(out, _judges(judges))
    ok = True
    for j, v in r["judges"].items():
        rate = v["decoys_accepted"] / max(v["decoys"], 1)
        ok &= rate <= 0.10
        typer.echo(f"{j}: decoys accepted {v['decoys_accepted']}/{v['decoys']} ({rate:.0%}) {v['accepted_by_kind']}; "
                   f"'true' without a verbatim quote: {v['no_quote']}")
    if "disagreement" in r:
        d, n = r["disagreement"]
        ok &= d / max(n, 1) <= 0.15
        typer.echo(f"disagreement {d}/{n} ({d / max(n, 1):.0%})")
    typer.echo("judges usable" if ok else "judges NOT usable: a judge accepts over 10% of decoys or they disagree on "
               "over 15%; sharpen the brief or change judge")


@judge_app.command("score")
def judge_score(out: Path = typer.Argument(..., help="pool directory"),
                judges: str = typer.Option("a,b", help="judge names; a claim is true when all say true"),
                without: str | None = typer.Option(None, help="also score without this relation type (the dominant one)")) -> None:
    """Judged precision, pooled recall and F1 per source, then per relation type."""
    from relweave import judge
    js = _judges(judges)
    r = judge.score(out, js)
    typer.echo(f"pool: {r['claims']} claims, {r['true']} judged true by {' and '.join(js)}")
    for n, v in r["sources"].items():
        typer.echo(f"{n:12} wrote {v['wrote']:4}  true {v['true']:4}  precision {v['precision']:.2f}  pooled recall "
                   f"{v['recall']:.2f}  F1 {v['f1']:.3f}  (95% {v['ci'][0]:.3f}-{v['ci'][1]:.3f}, document bootstrap)")
    rows = judge.by_type(out, js, without)
    names = list(r["sources"])
    typer.echo(f"\n{'type':24}{'true':>6}" + "".join(f"{n:>14}" for n in names))
    for t, v in rows.items():
        typer.echo(f"{t:24}{v['true']:6}" + "".join(f"{v['sources'][n][0]:14.2f}" for n in names))
    typer.echo("\nPooled recall counts only relations some source wrote; true recall is lower by an unmeasured amount.")


@judge_app.command("pair")
def judge_pair(out: Path = typer.Argument(..., help="pool directory"), a: str = typer.Argument(...),
               b: str = typer.Argument(...), judges: str = typer.Option("a,b")) -> None:
    """Paired document-bootstrap difference of judged F1, B minus A."""
    from relweave import judge
    d, lo, hi, n = judge.pair(out, a, b, _judges(judges))
    typer.echo(f"{b} - {a}: {d:+.3f} (95% {lo:+.3f} to {hi:+.3f}, paired document bootstrap, {n} documents)")


@judge_app.command("add")
def judge_add(out: Path = typer.Argument(..., help="pool directory"),
              label: str = typer.Argument(..., help="NAME=labels.jsonl of the system to add")) -> None:
    """Add a system to a judged pool; only its new claims go to the judges (OUT/items_add.jsonl), then `merge`."""
    from relweave import judge
    n, p = label.split("=", 1)
    typer.echo(f"{n}: {judge.add(out, n, p)} new claims -> {out}/items_add.jsonl; judge them into "
               f"verdicts_add_<judge>.jsonl, then: relweave judge merge {out}")


@judge_app.command("merge")
def judge_merge(out: Path = typer.Argument(..., help="pool directory"), judges: str = typer.Option("a,b")) -> None:
    """Fold verdicts_add_<judge>.jsonl into verdicts_<judge>.jsonl."""
    from relweave import judge
    judge.merge(out, _judges(judges))
    typer.echo("merged")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
