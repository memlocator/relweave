import json
import types

import torch

from relweave.head import LABEL_TOKENS, NONE_TOKEN, LabelHead, at_loss, head_config, label_token_ids, load_head, save_head
from relweave.schema import BUSINESS


class Tok:
    """One id per string: the label tokens are single tokens."""

    def __init__(self):
        self.vocab = {s: k for k, s in enumerate(LABEL_TOKENS + [NONE_TOKEN])}

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [self.vocab[s]]}


def test_label_tokens_have_the_threshold_token_last():
    ids = label_token_ids(Tok(), 3)
    assert ids == [0, 1, 2, len(LABEL_TOKENS)]


def test_label_head_scores_are_output_layer_logits_with_the_threshold_last():
    vocab = torch.randn(60, 8)
    model = types.SimpleNamespace(get_output_embeddings=lambda: types.SimpleNamespace(weight=vocab), device="cpu")
    ids = [5, 9, 52]
    head = LabelHead.from_model(model, ids)
    x = torch.randn(4, 8)
    assert torch.allclose(head(x), x @ vocab[ids].T)
    y = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert at_loss(head(x), y).isfinite()  # same layout as Head: labels, then the threshold class


def test_label_head_config_round_trips_without_weights(tmp_path):
    from relweave.head import labels
    n = len(labels(BUSINESS))
    head = LabelHead(torch.zeros(n + 1, 8), list(range(n + 1)))
    save_head(head, tmp_path, head_config(head, BUSINESS, 8, layers=(-1,)))
    assert not (tmp_path / "head.safetensors").exists()
    got, config = load_head(tmp_path)
    assert got is None and config["readout"] == "label" and config["label_tokens"] == list(range(n + 1))
    assert json.loads((tmp_path / "head_config.json").read_text())["layers"] == [-1]
