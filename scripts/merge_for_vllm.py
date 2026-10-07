"""A plain bf16 model of the generator for vLLM or any server: the 4-bit base the adapter was trained on, dequantized
(the exact weights the adapter learned against, not the original full-precision ones), with the LoRA merged in.

The adapter on the original full-precision base scores lower (it learned corrections for the 4-bit weights); the
dequantized base reproduces the trained model. Output: a Hugging Face model folder (safetensors, tokenizer and chat
template from the adapter).

Usage: uv run python scripts/merge_for_vllm.py WEIGHTS_DIR OUT_DIR
"""
import sys
from pathlib import Path

import torch


def main(weights: str, out: str) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from relweave.extract import generator_settings
    gen = Path(weights) / "generator"
    base = generator_settings(gen)["base_model"]
    model = AutoModelForCausalLM.from_pretrained(base, device_map="cuda", dtype=torch.bfloat16)
    model = model.dequantize()  # bitsandbytes 4-bit -> bf16, same values
    model = PeftModel.from_pretrained(model, str(gen)).merge_and_unload()
    # a model loaded from 4-bit cannot be saved directly (save_pretrained tries to undo the load-time conversion):
    # copy the merged weights into a clean bf16 model of the same architecture
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(base)
    for key in ("quantization_config", "_pre_quantization_dtype"):
        if hasattr(config, key):
            delattr(config, key)
    config.dtype = torch.bfloat16
    clean = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    state = {k: v.to(torch.bfloat16).cpu() for k, v in model.state_dict().items()}
    missing, unexpected = clean.load_state_dict(state, strict=False)
    if unexpected or [k for k in missing if k != "lm_head.weight"]:  # lm_head is tied to the embeddings
        raise RuntimeError(f"weights do not match: missing {missing[:5]}, unexpected {unexpected[:5]}")
    clean.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(str(gen)).save_pretrained(out)
    print(f"merged {base} (dequantized) + {gen} -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
