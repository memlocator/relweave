"""The pair head's label set, targets and the union follow a given schema; the business default is unchanged."""
from relweave.schema import Entity, Relation, Schema
from relweave.head import aligned_targets, gold_targets, labels
from relweave.schema import BUSINESS, ExtractionOutput


class Dept(Entity):
    """A department or business area of an organisation."""


class Proj(Entity):
    """A named project."""


class Pers(Entity):
    """A human being."""


class RunsProject(Relation[Dept, Proj]):
    """The source department runs the target project."""


class Colleague(Relation[Pers, Pers]):
    """The two persons work together."""
    symmetric = True


TOY = Schema(name="toy-pairhead", entities=[Dept, Proj, Pers], relations=[RunsProject, Colleague])


def out(rels):
    data = {"entities": [{"id": "e1", "type": "Dept", "name": "Grid", "mentions": ["Grid"]},
                         {"id": "e2", "type": "Proj", "name": "Saltdal", "mentions": ["Saltdal"]},
                         {"id": "e3", "type": "Pers", "name": "Tor", "mentions": ["Tor"]},
                         {"id": "e4", "type": "Pers", "name": "Kari", "mentions": ["Kari"]}],
            "relations": rels}
    return ExtractionOutput.model_validate(data, context={"schema": TOY})


def test_business_labels_are_unchanged():
    assert labels() == [(e, d) for e in BUSINESS.edge_names() for d in ((">",) if BUSINESS.symmetric(e) else (">", "<"))]


def test_labels_follow_the_schema():
    assert labels(TOY) == [("RUNS_PROJECT", ">"), ("RUNS_PROJECT", "<"), ("COLLEAGUE", ">")]


def test_targets_follow_the_schema():
    g = out([{"type": "RUNS_PROJECT", "source": "e1", "target": "e2", "evidence": "x"},
             {"type": "COLLEAGUE", "source": "e4", "target": "e3", "evidence": "x"}])
    num = {"e1": 1, "e2": 2, "e3": 3, "e4": 4}
    y = gold_targets(g, num, [(1, 2), (3, 4)], schema=TOY)
    assert y.tolist() == [[1, 0, 0], [0, 0, 1]]  # symmetric: '>' whatever the written direction
    ya = aligned_targets(g, {"e1": "e1", "e2": "e2", "e3": "e3", "e4": "e4"}, num, [(1, 2), (3, 4)], schema=TOY)
    assert ya.tolist() == y.tolist()


def test_union_checks_legality_against_the_schema():
    from relweave.formats import ExtractionResult
    from relweave.union import union
    r = ExtractionResult(chunk_id="c", raw="", output=out([]), fmt="sentences")
    scores = {"c": {"RUNS_PROJECT 1 2": 3.0, "RUNS_PROJECT 3 2": 3.0, "COLLEAGUE 3 4": 2.0}}
    u = union({"c": r}, scores, -2.0, 0.5, schema=TOY)["c"].output
    assert {(x.type, x.source, x.target) for x in u.relations} == {("RUNS_PROJECT", "e1", "e2"), ("COLLEAGUE", "e3", "e4")}
