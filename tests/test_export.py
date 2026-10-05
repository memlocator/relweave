"""scripts/export_model.py assembles the published layout from training runs (fake adapters, no GPU, no upload)."""
import importlib.util
import json
from pathlib import Path

import torch

from relweave.head import Head, labels, load_head
from relweave.schema import BUSINESS

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_model.py"


def _adapter(d: Path, extra=()):
    d.mkdir(parents=True)
    for name in ("adapter_config.json", "tokenizer_config.json", *extra):
        (d / name).write_text(json.dumps({"r": 32}))
    for name in ("adapter_model.safetensors", "tokenizer.json", "chat_template.jinja", "README.md"):
        (d / name).write_text("x")


def test_export_writes_the_published_layout(tmp_path):
    spec = importlib.util.spec_from_file_location("export_model", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    gen, head = tmp_path / "gen", tmp_path / "head"
    _adapter(gen / "adapter")
    (gen / "train_summary.json").write_text(json.dumps({"format": "sentences", "compact_prompt": True}))
    _adapter(head / "adapter")
    net = Head(2 * 16, len(labels(BUSINESS)), hidden=8)
    torch.save(net.state_dict(), head / "head.pt")
    (head / "train_summary.json").write_text(json.dumps({"schema": "business"}))
    out = mod.export(gen, head, tmp_path / "out")
    assert sorted(p.name for p in (out / "generator").iterdir()) == [
        "adapter_config.json", "adapter_model.safetensors", "chat_template.jinja", "relweave_config.json",
        "tokenizer.json", "tokenizer_config.json"]
    assert json.loads((out / "generator" / "relweave_config.json").read_text()) == {
        "schema": "business", "format": "sentences", "compact_prompt": True, "conditioned": False,
        "base_model": "unsloth/qwen3-4b-unsloth-bnb-4bit", "lora_rank": 32, "max_new_tokens": 2500}
    cfg = json.loads((out / "head" / "head_config.json").read_text())
    assert cfg["hidden_size"] == 16 and len(cfg["labels"]) == 38 and cfg["union"]["cut"] == -2.0
    loaded, _ = load_head(out / "head")
    assert torch.equal(loaded.net[1].weight, net.net[1].weight)
    assert not (out / "head" / "head.pt").exists() and (out / "schema.json").exists()
    assert "library_name: relweave" in (out / "README.md").read_text()
