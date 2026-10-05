"""relweave must import and work with nothing outside its own folder: no bench, no scripts/, no repo paths."""
import re
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "relweave"

PROBE = """
import importlib.abc, sys

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "bench" or name.startswith("bench.") or name == "scripts" or name.startswith("scripts."):
            raise ImportError(f"relweave imported {name}")

sys.meta_path.insert(0, Block())
import pkgutil, relweave
for m in pkgutil.walk_packages(relweave.__path__, "relweave."):
    __import__(m.name)
loaded = sorted(m for m in sys.modules if m == "bench" or m.startswith("bench."))
assert not loaded, loaded
print("ok", len([m for m in sys.modules if m.startswith("relweave")]))
"""


def test_every_relweave_module_imports_with_bench_blocked_and_loads_no_bench_module():
    out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("ok")


def test_no_source_file_mentions_bench_scripts_or_run_directories():
    bad = re.compile(r"\bfrom bench\b|\bimport bench\b|\bbench\.|sys\.path|\bruns/|\bdata/chunks")
    hits = [f"{p.relative_to(SRC)}:{i}: {line.strip()}" for p in SRC.rglob("*.py")
            for i, line in enumerate(p.read_text().splitlines(), 1) if bad.search(line)]
    assert not hits, "\n".join(hits)
