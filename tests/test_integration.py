import os
import shutil
import subprocess

import pytest

TEXT = """Varden Marine AS is a Norwegian shipyard based in Bergen. It is owned by the investment company Nordhavn Capital, which holds a 60 percent stake.

The chief executive, Ingrid Solheim, joined Varden Marine in 2019. She previously worked at Nordhavn Capital.

In 2024 the company opened a second yard in Stavanger. Solheim said the yard in Stavanger would build ferries for the operator Kystlink.
"""


def free_mb():
    if not shutil.which("nvidia-smi"):
        return 0
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.split()[0]) if out.returncode == 0 and out.stdout.strip() else 0


@pytest.mark.skipif(os.environ.get("RELWEAVE_GPU_TEST") != "1",
                    reason="GPU test: set RELWEAVE_GPU_TEST=1 (only when no other GPU work is running)")
@pytest.mark.skipif(free_mb() < 5000, reason="needs a CUDA GPU with 5 GB free")
def test_three_paragraph_text_through_extractor(tmp_path):
    from relweave import Extractor
    g = Extractor(max_words=60).run(TEXT)
    assert len(g.document["chunks"]) >= 2
    assert not g.warnings, g.warnings
    assert g.entities and g.mentioned_in
    assert all(m["end"] - m["start"] == len(m["mention"]) for m in g.mentioned_in)
    names = " ".join(e.name for e in g.entities)
    assert "Varden" in names
