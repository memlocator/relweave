import torch

from relweave.head import at_loss, entity_lines, gold_targets, labels, pack, predict


class Tok:
    """Whitespace tokenizer: enough to check packing."""
    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [hash(w) % 1000 for w in s.replace("\n", " \n ").split(" ") if w]}


def test_packing_isolates_probes_and_shares_positions():
    ids, pos, mask, pairs, last = pack(Tok(), "a b c", {1: "Org", 2: "Place", 3: "Person"})
    assert pairs == [(1, 2), (1, 3), (2, 3)]
    n_pre = 3
    starts = [n_pre] + [l + 1 for l in last[:-1]]
    for s, e in zip(starts, last):
        assert pos[s] == n_pre  # every probe starts at the same position
        assert mask[e, :n_pre].all()  # sees the prefix
    assert not mask[last[1], starts[0]]  # second probe cannot see the first


def test_entity_lines_and_targets():
    from relweave.schema import ExtractionOutput
    text, types = entity_lines("E1 Org: Acme\nE2 Place: Oslo\nS1\nR HEADQUARTERED_IN E1 E2 asserted")
    assert types == {1: "Org", 2: "Place"} and text == "E1 Org: Acme\nE2 Place: Oslo\n"
    gold = ExtractionOutput.model_validate({
        "entities": [{"id": "e1", "type": "Org", "name": "Acme", "mentions": ["Acme"]},
                     {"id": "e2", "type": "Place", "name": "Oslo", "mentions": ["Oslo"]}],
        "relations": [{"type": "HEADQUARTERED_IN", "source": "e1", "target": "e2", "evidence": "x"}]})
    y = gold_targets(gold, {"e1": 1, "e2": 2}, [(1, 2)])
    assert y[0, labels().index(("HEADQUARTERED_IN", ">"))] == 1 and y.sum() == 1


def test_at_loss_is_low_when_positives_beat_threshold():
    y = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    good = torch.tensor([[5.0, -5.0, 0.0], [-5.0, -5.0, 0.0]])
    bad = torch.tensor([[-5.0, 5.0, 0.0], [5.0, 5.0, 0.0]])
    assert at_loss(good, y) < at_loss(bad, y)
    assert predict(good) == [[0], []]


def test_at_loss_penalises_a_positive_below_the_threshold():
    """The bug this guards against: a loss without the threshold in part 1 never lifts positives, so a head
    that predicts nothing reaches a near-zero loss."""
    y = torch.tensor([[1.0, 0.0]])
    nothing = torch.tensor([[-5.0, -5.0, 5.0]])   # threshold above everything
    right = torch.tensor([[5.0, -5.0, 0.0]])
    assert at_loss(nothing, y) > 5 and at_loss(right, y) < 0.1


def test_aligned_targets_follow_model_entities_through_alignment():
    from relweave.head import aligned_targets
    from relweave.schema import ExtractionOutput
    gold = ExtractionOutput.model_validate({
        "entities": [{"id": "g1", "type": "Org", "name": "Acme", "mentions": ["Acme"]},
                     {"id": "g2", "type": "Place", "name": "Oslo", "mentions": ["Oslo"]}],
        "relations": [{"type": "HEADQUARTERED_IN", "source": "g1", "target": "g2", "evidence": "x"}]})
    # model numbered Oslo 1 and Acme 2, and invented a third entity
    y = aligned_targets(gold, {"e1": "g2", "e2": "g1", "e3": None}, {"e1": 1, "e2": 2, "e3": 3}, [(1, 2), (1, 3)])
    assert y[0, labels().index(("HEADQUARTERED_IN", "<"))] == 1 and y[1].sum() == 0


def test_head_saves_as_safetensors_with_config_and_loads_back(tmp_path):
    import json
    from relweave.head import Head, head_config, load_head, save_head
    from relweave.schema import BUSINESS
    head = Head(2 * 16, len(labels(BUSINESS)), hidden=8)
    save_head(head, tmp_path, head_config(head, BUSINESS, hidden_size=16))
    cfg = json.loads((tmp_path / "head_config.json").read_text())
    assert cfg["schema"] == "business" and cfg["layers"] == [-1, -9] and cfg["probe_hidden"] == 8
    assert cfg["labels"] == [list(x) for x in labels(BUSINESS)] and cfg["threshold_logit"] == "last"
    assert cfg["union"] == {"cut": -2.0, "margin": 0.5, "max_relations": 40}
    loaded, config = load_head(tmp_path)
    x = torch.randn(3, 32)
    assert torch.equal(loaded(x), head.eval()(x)) and config == cfg


def test_a_pickled_head_pt_is_read_as_a_fallback(tmp_path):
    import json
    from relweave.head import Head, load_head
    head = Head(32, 5, hidden=8)
    torch.save(head.state_dict(), tmp_path / "head.pt")
    (tmp_path / "train_summary.json").write_text(json.dumps({"schema": "legal"}))
    loaded, config = load_head(tmp_path)
    assert config["schema"] == "legal" and loaded.net[4].out_features == 6
