"""Batched local inference with transformers.

Quick way to sanity-check a fine-tuned model (merged dir or LoRA adapter dir)
before standing up a real server. Generates replies for one or more prompts.

Run:
    python modules/09_deployment_serving/local_inference.py --model outputs/04_merged \
        --prompts "Explain LoRA in one sentence." "Write a haiku about GPUs."
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import get_device  # noqa: E402
from common.model_utils import DEFAULT_MODEL, load_tokenizer  # noqa: E402


def load_model(model_dir: str):
    import torch
    from transformers import AutoModelForCausalLM

    adapter_cfg = Path(model_dir) / "adapter_config.json"
    if adapter_cfg.exists():
        import json

        from peft import PeftModel

        base = json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
        print(f"Loading base {base} + adapter {model_dir}")
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
        return PeftModel.from_pretrained(model, model_dir)
    print(f"Loading standalone model {model_dir}")
    return AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompts", nargs="+", default=["Say hello in three languages."])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    import torch

    device = get_device()
    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model)
    if device != "cuda":
        model.to(device)
    model.eval()

    # Build batched chat prompts. Left-padding is correct for batched generation.
    tokenizer.padding_side = "left"
    chats = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in args.prompts
    ]
    inputs = tokenizer(chats, return_tensors="pt", padding=True).to(device)

    start = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - start

    gen_tokens = 0
    for i, prompt in enumerate(args.prompts):
        new = out[i][inputs["input_ids"].shape[1]:]
        gen_tokens += (new != tokenizer.pad_token_id).sum().item()
        reply = tokenizer.decode(new, skip_special_tokens=True).strip()
        print(f"\n--- Prompt {i + 1} ---")
        print(f"User: {prompt}")
        print(f"Model: {reply}")

    print(f"\nGenerated ~{gen_tokens} tokens in {elapsed:.2f}s "
          f"({gen_tokens / max(elapsed, 1e-6):.1f} tok/s on {device})")


if __name__ == "__main__":
    main()
