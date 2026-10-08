"""Zero-shot relations: candidate questions, threshold calibration, packing, and the graph they produce."""
import types

import pytest

from relweave.schema import Relation, Schema
from relweave.schema.business import Org, Person
from relweave.zeroshot import DENSITY_REF, adjust, calibrate, candidates, pack, threshold_for


class DonatedTo(Relation[Person | Org, Org]):
    """The source has given money or goods to the target organisation."""


class PartneredWith(Relation[Org, Org]):
    """The two organisations work together."""
    symmetric = True


CHARITY = Schema(name="charity_test", entities=[Person, Org], relations=[DonatedTo, PartneredWith])


def test_candidates_fit_endpoint_types_and_ask_symmetric_types_once():
    ents = [("Ingrid", "Person"), ("Harbour Light", "Org"), ("Food Bank", "Org")]
    c = set(candidates(CHARITY, ents))
    assert ("DONATED_TO", 0, 1) in c and ("DONATED_TO", 1, 2) in c and ("DONATED_TO", 1, 0) not in c  # Org target only
    assert ("PARTNERED_WITH", 1, 2) in c and ("PARTNERED_WITH", 2, 1) not in c


def test_calibrate_picks_per_type_thresholds_and_pools_rare_types():
    scored = [("DONATED_TO", p, y) for p, y in [(0.95, True), (0.9, True), (0.88, True), (0.75, False), (0.6, False)]]
    scored += [("PARTNERED_WITH", 0.97, True), ("PARTNERED_WITH", 0.4, False)]
    t = calibrate(scored)
    assert 0.75 <= t["DONATED_TO"] < 0.88
    assert t["PARTNERED_WITH"] == t["*"]  # one positive: falls back to the pooled threshold
    assert threshold_for(t, "UNSEEN") == t["*"] and threshold_for(0.7, "X") == 0.7


def test_packed_questions_see_the_prefix_but_not_each_other():
    ids, pos, mask, last = pack([1, 2, 3], [[4, 5], [6, 7]])
    assert ids == [1, 2, 3, 4, 5, 6, 7] and pos == [0, 1, 2, 3, 4, 3, 4] and last == [4, 6]
    assert mask[5, :3].all() and not mask[5, 3:5].any() and mask[6, 5]


def test_zeroshot_relations_above_threshold_enter_the_graph():
    from relweave import Extractor
    from relweave.schema import ExtractionOutput
    text = "Ingrid gave Harbour Light money."
    out = ExtractionOutput.model_construct(entities=[
        types.SimpleNamespace(id="e1", type="Person", name="Ingrid", mentions=["Ingrid"], attributes={}),
        types.SimpleNamespace(id="e2", type="Org", name="Harbour Light", mentions=["Harbour Light"], attributes={})],
        relations=[])

    class FakeZS:
        def score_chunk(self, text, ents, schema):
            return [("DONATED_TO", 0, 1, 0.92), ("DONATED_TO", 1, 0, 0.3)]

    e = Extractor.__new__(Extractor)
    e._zs, e.threshold, e.schema = FakeZS(), {"DONATED_TO": 0.8}, CHARITY
    chunk = types.SimpleNamespace(index=0, text=text, start=0, end=len(text))
    g = e._zeroshot_graph(chunk, types.SimpleNamespace(output=out))
    assert [(r.type, r.source, r.target, r.origin) for r in g.relations] == [("DONATED_TO", "e1", "e2", "zeroshot")]
    assert g.relations[0].score == round(adjust(0.92, 2), 3)


def test_density_adjustment_is_stricter_the_more_questions_a_chunk_asks():
    assert adjust(0.8, DENSITY_REF) == pytest.approx(0.8)
    assert adjust(0.8, 10 * DENSITY_REF) < 0.8 < adjust(0.8, DENSITY_REF // 10)


def test_zeroshot_refuses_entity_types_the_generator_cannot_find(monkeypatch):
    from relweave import Extractor
    from relweave import extract as ex
    from relweave.schema import Entity

    class Grant(Entity):
        """A grant."""

    class Awarded(Relation[Org, Grant]):
        """The source awarded the target grant."""

    class CE:
        def __init__(self, *a, **k):
            from relweave.schema import BUSINESS
            self.schema, self.model, self.cut, self.margin = BUSINESS, "m", None, None

    monkeypatch.setattr(ex, "ChunkExtractor", CE)
    monkeypatch.setattr(ex, "resolve_weights", lambda spec: __import__("pathlib").Path("/w"))
    with pytest.raises(ValueError, match="Grant"):
        Extractor(schema=Schema(name="grants_test", entities=[Org, Grant], relations=[Awarded]), zeroshot=True,
                  sequential=True)
