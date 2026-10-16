from relweave.formats import ExtractionResult
from relweave.schema import BUSINESS
from relweave.schema.output import Entity, ExtractionOutput, Relation
from relweave.union import union


def test_head_additions_are_not_blocked_by_the_generator_cap():
    """A dense list chunk: the generator wrote 40 relations (its cap) for 40 people; the head scores 20 more people
    it left out. All 60 relations must survive the union (the cap stops a looping generator, not the head)."""
    org = Entity(id="e1", type="Org", name="Acme", mentions=["Acme"])
    people = [Entity(id=f"e{k}", type="Person", name=f"Person {k}", mentions=[f"Person {k}"]) for k in range(2, 62)]
    written = [Relation(type="BOARD_MEMBER_OF", source=p.id, target="e1", evidence="x") for p in people[:40]]
    out = ExtractionOutput.model_construct(entities=[org, *people], relations=written)
    r = ExtractionResult(chunk_id="c", raw="", output=out)
    scores = {f"BOARD_MEMBER_OF {p.id[1:]} 1": 3.0 for p in people}
    final = union({"c": r}, {"c": scores}, cut=0.0, margin=0.5, schema=BUSINESS)["c"]
    assert len([x for x in final.output.relations if x.type == "BOARD_MEMBER_OF"]) == 60
