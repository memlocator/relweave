import subprocess
import types


from relweave import Extractor
from relweave.records import ChunkEntity, ChunkGraph, Mention


class FakeCE:
    schema = None

    def __init__(self, fail_gen=(), fail_head=(), fail_load=None):
        self.fail_gen, self.fail_head, self.fail_load, self.log = set(fail_gen), set(fail_head), fail_load, []
        self.batches = []

    def load_generator(self):
        if self.fail_load is not None:
            raise self.fail_load
        self.log.append("load gen")

    def load_head(self):
        self.log.append("load head")

    def generate_many(self, items, batch_size):
        self.batches.append([k for k, _ in items])
        from relweave.schema import ExtractionOutput
        return {k: RuntimeError("boom") if int(k.split(":")[1]) in self.fail_gen
                else types.SimpleNamespace(output=ExtractionOutput(), error=None, raw="") for k, _ in items}

    def pair_scores(self, c, r):
        if c.index in self.fail_head:
            raise MemoryError("oom")
        return {}

    def combine(self, c, r, s):
        return ChunkGraph(c.index, [ChunkEntity("e1", "Org", "Acme", [Mention("Acme", c.start)])])

    def unload_generator(self):
        self.log.append("gen")

    def unload_head(self):
        self.log.append("head")


def make(ce, **kw):
    e = Extractor.__new__(Extractor)
    e.schema, e.max_words, e.sequential, e._ce, e.batch_size = None, 3, True, ce, "auto"
    e.model_info = {}
    return e


TEXT = "Acme one two.\n\nAcme three four.\n\nAcme five six."


def test_failed_chunks_become_warnings_and_the_document_continues():
    ce = FakeCE(fail_gen={1}, fail_head={2})
    g = make(ce).run(TEXT)
    assert len(g.warnings) == 2
    assert "chunk 1" in g.warnings[0] and "generation failed" in g.warnings[0]
    assert "chunk 2" in g.warnings[1] and "head scoring failed" in g.warnings[1]
    assert "gen" in ce.log and ce.log[-1] == "head"
    assert g.entities and len(g.document["chunks"]) == 3


def test_cli_help_lists_run():
    out = subprocess.run(["uv", "run", "relweave", "run", "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "--out" in out.stdout and "--html" not in out.stdout


def test_run_many_loads_each_model_once_and_returns_one_graph_per_document():
    ce = FakeCE()
    gs = make(ce).run_many([TEXT, TEXT], ["a", "b"])
    assert [g.document["id"] for g in gs] == ["a", "b"]
    assert ce.log.count("head") == 1


def test_models_are_unloaded_even_when_a_phase_raises():
    class Boom(FakeCE):
        def combine(self, c, r, s):
            raise KeyboardInterrupt

    ce = Boom()
    try:
        make(ce).run(TEXT)
    except KeyboardInterrupt:
        pass
    assert ce.log[-1] == "head"


def test_cli_sends_all_files_through_one_extractor_call(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import relweave
    from relweave.cli import app
    from relweave.graph import Graph
    calls = []

    class Fake:
        def __init__(self, **kw):
            pass

        def iter_run(self, texts, ids):
            calls.append((texts, ids))
            for k, (t, i) in enumerate(zip(texts, ids)):
                yield k, Graph(text=t, document={"id": i, "chunks": []})

    monkeypatch.setattr(relweave, "Extractor", Fake)
    for n in "ab":
        (tmp_path / f"{n}.txt").write_text(f"text {n}")
    res = CliRunner().invoke(app, ["run", str(tmp_path / "a.txt"), str(tmp_path / "b.txt"), "--out", str(tmp_path / "o")])
    assert res.exit_code == 0, res.output
    assert calls == [(["text a", "text b"], ["a", "b"])]
    assert (tmp_path / "o" / "a.json").exists() and (tmp_path / "o" / "b.json").exists()
    res = CliRunner().invoke(app, ["run", str(tmp_path / "a.txt"), "--out", str(tmp_path / "one.json")])
    assert res.exit_code == 0 and (tmp_path / "one.json").exists()


def test_a_generator_that_cannot_load_raises_with_the_memory_situation():
    import pytest
    ce = FakeCE(fail_load=MemoryError("CUDA out of memory while loading"))
    with pytest.raises(RuntimeError, match="could not load the generator: MemoryError"):
        make(ce).run(TEXT)


def test_a_generator_failing_on_every_chunk_raises():
    import pytest
    ce = FakeCE(fail_gen={0, 1, 2})
    with pytest.raises(RuntimeError, match="generation failed on every chunk"):
        make(ce).run(TEXT)
    assert ce.log[-1] == "head"  # models are unloaded on the way out


def test_chunks_of_all_documents_are_generated_in_one_call_and_map_back():
    ce = FakeCE(fail_gen={1})
    gs = make(ce).run_many([TEXT, "Bolt one."], ["a", "b"])
    assert ce.batches == [["0:0", "0:1", "0:2", "1:0"]]
    a, b = gs
    assert len(a.document["chunks"]) == 3 and len(b.document["chunks"]) == 1
    assert len(a.warnings) == 1 and b.warnings == []  # chunk 1 of document 0 failed, document 1 is whole


def test_iter_run_yields_each_document_with_its_index():
    ce = FakeCE()
    got = list(make(ce).iter_run([TEXT, TEXT, TEXT], ["a", "b", "c"]))
    assert [k for k, _ in got] == [0, 1, 2] and [g.document["id"] for _, g in got] == ["a", "b", "c"]


def test_cli_exits_non_zero_when_the_model_cannot_load(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import relweave
    from relweave.cli import app

    class Broken:
        def __init__(self, **kw):
            raise RuntimeError("could not load the generator: OutOfMemoryError")

    monkeypatch.setattr(relweave, "Extractor", Broken)
    (tmp_path / "a.txt").write_text("text")
    res = CliRunner().invoke(app, ["run", str(tmp_path / "a.txt"), "--out", str(tmp_path / "a.json")])
    assert res.exit_code == 1 and "could not load the generator" in res.output


def test_cli_passes_schema_cut_margin_and_batch_size(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import relweave
    from relweave.cli import app
    seen = {}

    class Fake:
        def __init__(self, **kw):
            seen.update(kw)

        def iter_run(self, texts, ids):
            return iter(())

    monkeypatch.setattr(relweave, "Extractor", Fake)
    (tmp_path / "a.txt").write_text("text")
    res = CliRunner().invoke(app, ["run", str(tmp_path / "a.txt"), "--out", str(tmp_path / "a.json"), "--schema",
                                   "my.py:MINE", "--cut", "-1.5", "--margin", "1", "--batch-size", "4"])
    assert res.exit_code == 0, res.output
    assert (seen["schema"], seen["cut"], seen["margin"], seen["batch_size"]) == ("my.py:MINE", -1.5, 1.0, 4)
