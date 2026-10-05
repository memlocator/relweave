"""Draft correction (relweave.training.correct.apply) follows the chunk's schema."""
from test_head_schema import TOY, out


def test_apply_uses_the_given_schema():
    from relweave.training.correct import apply
    text = "Grid runs Saltdal. Tor and Kari work together. Grid also runs Fjordvann."
    draft = out([{"type": "RUNS_PROJECT", "source": "e1", "target": "e2", "evidence": "x"}])
    lines = ["entity E5 Proj: Fjordvann",            # new entity of a toy type
             "add RUNS_PROJECT E1 E5",                # legal in the toy schema
             "add COLLEAGUE E3 E4",                   # legal, symmetric
             "add RUNS_PROJECT E3 E2",                # Person -> Proj: illegal, skipped
             "entity E6 Org: Grid"]                   # Org is not a toy type: skipped
    o, skipped = apply(draft, text, lines, TOY)
    got = {(r.type, r.source, r.target) for r in o.relations}
    assert got == {("RUNS_PROJECT", "e1", "e2"), ("RUNS_PROJECT", "e1", "e5"), ("COLLEAGUE", "e3", "e4")}
    assert {str(e.type) for e in o.entities} == {"Dept", "Proj", "Pers"} and skipped == 2
