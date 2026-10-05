import json
from pathlib import Path

from typer.testing import CliRunner

from relweave.cli import app
from relweave.schema import SCHEMAS, ExtractionOutput
from relweave.training.labels import write_prompts

GOLDEN = Path(__file__).parent / "golden" / "teacher_prompts.json"
CHUNKS = [("doc1#0", "Acme Robotics AB is based in Gothenburg."), ("doc2#3", "Maria Lind joined the board.")]


def _prompts(tmp_path, schema=None):
    paths = write_prompts(CHUNKS, tmp_path, **({"schema": schema} if schema else {}))
    return [json.loads(p.read_text()) for p in paths]


def test_business_teacher_prompts_match_golden(tmp_path):
    assert _prompts(tmp_path) == json.loads(GOLDEN.read_text())
    assert _prompts(tmp_path / "explicit", SCHEMAS["business"]) == json.loads(GOLDEN.read_text())


def test_politics_teacher_prompt_is_rendered_from_the_schema(tmp_path):
    system = _prompts(tmp_path, SCHEMAS["politics"])[0]["messages"][0]["content"]
    politics = SCHEMAS["politics"]
    assert all(f"- {e}:" in system and politics.definition(e) in system for e in politics.edge_names())
    assert all(f"- {n}" in system for n in politics.node_names())
    assert "EXECUTIVE_OF" not in system


POLITICS_RAW = {
    "entities": [{"id": "e1", "type": "Person", "name": "Anna Berg", "attributes": {}, "mentions": ["Anna Berg"]},
                 {"id": "e2", "type": "Org", "name": "Green Party", "attributes": {}, "mentions": ["Green Party"]}],
    "relations": [{"type": "MEMBER_OF_PARTY", "source": "e1", "target": "e2", "attributes": {},
                   "modality": "asserted", "evidence": "Anna Berg is a member of the Green Party."}],
}


def test_cli_teacher_prompts_and_import_labels_for_politics(tmp_path):
    text = "Anna Berg is a member of the Green Party."
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(json.dumps({"chunk_id": "d#0", "doc_id": "d", "text": text, "split": "test"}) + "\n")
    runner = CliRunner()
    r = runner.invoke(app, ["label", "prompts", str(chunks), "--out", str(tmp_path / "p"),
                            "--schema", "politics"])
    assert r.exit_code == 0, r.output
    prompt = json.loads((tmp_path / "p" / "d_0.json").read_text())
    assert "MEMBER_OF_PARTY" in prompt["messages"][0]["content"]
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "d_0.txt").write_text(json.dumps(POLITICS_RAW))
    r = runner.invoke(app, ["label", "import", str(chunks), "--raw-dir", str(raw),
                            "--out", str(tmp_path / "out"), "--train-windows", "0", "--schema", "politics"])
    assert r.exit_code == 0, r.output
    rec = json.loads((tmp_path / "out" / "test.jsonl").read_text())
    assert rec["schema"] == "politics"
    gold = ExtractionOutput.model_validate(rec["gold"], context={"schema": SCHEMAS["politics"]})
    assert [x.type for x in gold.relations] == ["MEMBER_OF_PARTY"]


def test_import_labels_drops_a_relation_that_is_not_in_the_chosen_schema(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(json.dumps({"chunk_id": "d#0", "doc_id": "d", "text": "Anna Berg is a member of the Green Party."}) + "\n")
    raw = tmp_path / "raw"
    raw.mkdir()
    bad = json.loads(json.dumps(POLITICS_RAW))
    bad["relations"][0]["type"] = "EMPLOYED_BY"
    (raw / "d_0.txt").write_text(json.dumps(bad))
    r = CliRunner().invoke(app, ["label", "import", str(chunks), "--raw-dir", str(raw),
                                 "--out", str(tmp_path / "out"), "--train-windows", "0", "--schema", "politics"])
    assert r.exit_code == 0, r.output
    rec = json.loads((tmp_path / "out" / "train.jsonl").read_text())
    assert rec["gold"]["relations"] == []


def test_training_windows_keep_a_non_business_schema(tmp_path):
    text = "Anna Berg is a member of the Green Party. She spoke on Monday."
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(json.dumps({"chunk_id": "d#0", "doc_id": "d", "text": text}) + "\n")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "d_0.txt").write_text(json.dumps(POLITICS_RAW))
    r = CliRunner().invoke(app, ["label", "import", str(chunks), "--raw-dir", str(raw), "--out", str(tmp_path / "out"),
                                 "--train-windows", "45", "--schema", "politics"])
    assert r.exit_code == 0, r.output
    recs = [json.loads(line) for line in (tmp_path / "out" / "train.jsonl").read_text().splitlines()]
    window = next(x for x in recs if x["chunk_id"] != "d#0")
    assert window["schema"] == "politics" and window["gold"]["relations"][0]["type"] == "MEMBER_OF_PARTY"
