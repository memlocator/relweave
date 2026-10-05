"""relweave command line.

  relweave run <file.txt> --out graph.json          text to a graph (JSON Graph Format v2)
  relweave label ingest|prompts|import|draft|correct  build labelled chunk files for your own schema
  relweave generate / score                         run the generator / the pair head over a chunk file
  relweave eval                                     score saved results against gold
  relweave train-generator / train-head             train both parts (pip install "relweave[train]")
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

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
        head: Path | None = typer.Option(None, help="pair head directory, overriding the one in --weights"),
        cut: float | None = typer.Option(None, help="drop a generator relation the head scores at or below this "
                                         "(default from the head's config, -2.0)"),
        margin: float | None = typer.Option(None, help="add a head relation scoring above this (default from the "
                                            "head's config, +0.5)"),
        batch_size: str = typer.Option("auto", help="chunks generated together: 1, 2, 4, 8 ... or auto"),
        max_words: int = typer.Option(200, help="words per chunk"),
        reserve_mb: int = typer.Option(500, help="GPU memory left free")) -> None:
    """Extract a graph from each file; all files go through one Extractor, so each model loads once."""
    import relweave
    try:
        ex = relweave.Extractor(weights=weights, schema=schema, model=model, generator=generator, head=head, cut=cut,
                                margin=margin, batch_size=_batch(batch_size), max_words=max_words, reserve_mb=reserve_mb)
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
                 seed: int = 0, max_chars: int = typer.Option(4000, help="characters per chunk"),
                 overlap_sentences: int = 2) -> None:
    """Chunk a directory of documents into unlabelled chunks with a train/test split by document."""
    from relweave.training.data import ingest
    files = sorted(texts_dir.glob("*.txt"))
    if not files:
        raise typer.BadParameter(f"no *.txt files in {texts_dir}")
    n, n_test = ingest(files, out, test_frac, seed, max_chars, overlap_sentences)
    typer.echo(f"wrote {n} chunks from {len(files)} documents ({n_test} test docs) -> {out}")


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
    typer.echo(f"missing answers: {c['missing']}; unverifiable items dropped from labels: {c['dropped']}")


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
             limit: int | None = None, reserve_mb: int = 500) -> None:
    """The generator's results on every chunk of a chunk file."""
    from relweave.extract import DEFAULT_MODEL
    from relweave.training.predict import generate as run
    gen, _ = _parts(weights, generator, None, need_head=False)
    n = run(chunks, out, gen, model or DEFAULT_MODEL, _schema(schema), _batch(batch_size), reserve_mb, limit)
    typer.echo(f"{n} results -> {out}")


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
    from relweave.extract import DEFAULT_MODEL
    from relweave.training.predict import score as run
    gen, hd = _parts(weights, generator, head, need_head=True)
    n = run(chunks, results, out, gen, hd, model or DEFAULT_MODEL, _schema(schema), reserve_mb)
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
                   reserve_mb: int = 700) -> None:
    """Joint training of the pair head and its adapter (needs the train extra)."""
    from relweave.extract import DEFAULT_MODEL
    from relweave.training.head import train_head
    summary = train_head(chunks, out, generator, model or DEFAULT_MODEL, own_results, base_head, _schema(schema),
                         examples, max_probes, seed=seed, list_repeat=list_repeat, upcast=upcast, reserve_mb=reserve_mb)
    typer.echo(json.dumps({k: v for k, v in summary.items() if k != "labels"}, indent=1))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
