"""Hidden states of chosen layers without keeping every layer's (output_hidden_states=True holds all 37 layers of the
4B for every token: about 1.4 GB at 7,000 tokens, the difference between a long document fitting an 8 GB card or not).

Layers are numbered as in transformers' `hidden_states`: index 0 the embeddings, index i the output of decoder layer
i - 1, the last index (-1) the final norm's output; negative indices count from the end of that tuple.
"""

from __future__ import annotations

import torch


def _inner(model):
    """The decoder stack (with .layers and .norm) of a causal LM, a PEFT wrapper included."""
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model if hasattr(m, "model") and hasattr(m.model, "layers") else m


@torch.no_grad()
def layer_states(model, layers, **inputs) -> list[torch.Tensor]:
    """One forward pass; the [batch, tokens, hidden] states of `layers` (hidden_states indices), in that order. Only
    those layers are kept, no vocabulary scores are computed beyond the last token and no key/value cache is built."""
    inner = _inner(model)
    n = len(inner.layers)
    got: dict[int, torch.Tensor] = {}
    hooks = []

    def keep(idx):
        def hook(module, args, output):
            got[idx] = output[0] if isinstance(output, tuple) else output
        return hook

    for k in layers:
        idx = k + n + 1 if k < 0 else k
        if not 1 <= idx <= n:
            raise ValueError(f"layer {k}: only decoder layer outputs (1..{n} or -{n}..-1) are supported")
        module = inner.norm if idx == n else inner.layers[idx - 1]
        hooks.append(module.register_forward_hook(keep(idx)))
    try:
        model(**inputs, logits_to_keep=1, use_cache=False)  # no generation follows: no key/value cache to keep
    finally:
        for h in hooks:
            h.remove()
    return [got[k + n + 1 if k < 0 else k] for k in layers]
