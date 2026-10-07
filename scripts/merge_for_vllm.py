"""A plain bf16 model of the generator for vLLM or any server: the 4-bit base the adapter was trained on, dequantized
(the exact weights the adapter learned against, not the original full-precision ones), with the LoRA merged in.

The adapter on the original full-precision base scores lower (it learned corrections for the 4-bit weights); the
dequantized base reproduces the trained model. The 4-bit model sits on the GPU; each layer is dequantized there and
moved to the CPU in bf16; the adapter is merged one layer at a time in float32 and the tensors are written directly, so
a 4B base needs about 3 GB of GPU memory and 10 GB of RAM.
Output: a Hugging Face model folder (safetensors, tokenizer and chat template from the adapter).

Usage: uv run python scripts/merge_for_vllm.py WEIGHTS_DIR OUT_DIR
"""
import json
import sys
from pathlib import Path

import torch


def main(weights: str, out: str) -> None:
    import bitsandbytes as bnb
    from safetensors.torch import load_file, save_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    from relweave.extract import generator_settings
    gen = Path(weights) / "generator"
    base = generator_settings(gen)["base_model"]
    model = AutoModelForCausalLM.from_pretrained(base, device_map="cuda", dtype=torch.bfloat16)
    state = {}
    for name, mod in model.named_modules():
        if isinstance(mod, bnb.nn.Linear4bit):
            w = bnb.functional.dequantize_4bit(mod.weight.data, mod.weight.quant_state)
            state[f"{name}.weight"] = w.to(torch.bfloat16).cpu()
            if mod.bias is not None:
                state[f"{name}.bias"] = mod.bias.detach().to(torch.bfloat16).cpu()
    for name, p in model.named_parameters():
        if name not in state:  # embeddings, norms: not quantized
            state[name] = p.detach().to(torch.bfloat16).cpu()
    del model
    torch.cuda.empty_cache()

    cfg = json.loads((gen / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
    lora = load_file(str(gen / "adapter_model.safetensors"))
    merged = 0
    for key, a in lora.items():
        if ".lora_A." not in key:
            continue
        target = key.split("base_model.model.", 1)[1].replace(".lora_A.weight", ".weight")
        b = lora[key.replace(".lora_A.", ".lora_B.")]
        state[target] = (state[target].float() + scale * (b.float() @ a.float())).to(torch.bfloat16)
        merged += 1

    config = AutoConfig.from_pretrained(base)
    for key in ("quantization_config", "_pre_quantization_dtype"):
        if hasattr(config, key):
            delattr(config, key)
    config.dtype = torch.bfloat16
    if not getattr(config, "tie_word_embeddings", False) and "lm_head.weight" not in state:
        raise RuntimeError("untied lm_head missing from the state")
    Path(out).mkdir(parents=True, exist_ok=True)
    config.save_pretrained(out)
    GenerationConfig.from_pretrained(base).save_pretrained(out)
    save_file({k: v.contiguous() for k, v in state.items()}, str(Path(out) / "model.safetensors"), metadata={"format": "pt"})
    AutoTokenizer.from_pretrained(str(gen)).save_pretrained(out)
    print(f"merged {base} (dequantized) + {merged} LoRA layers from {gen} (scale {scale:g}) -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
