"""Chat with a fine-tuned model from the terminal.

Works with a full model directory or a LoRA adapter directory (auto-detected via
the presence of an adapter_config.json).

Run:
    python modules/06_instruction_tuning/chat_inference.py --model outputs/06_sft
    python modules/06_instruction_tuning/chat_inference.py --model outputs/06_sft --prompt "Explain recursion."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import get_device  # noqa: E402
from common.model_utils import load_tokenizer  # noqa: E402


def load_model(model_dir: str):
    import torch
    from transformers import AutoModelForCausalLM

    adapter_cfg = Path(model_dir) / "adapter_config.json"
    if adapter_cfg.exists():
        # It's a PEFT adapter — load the base it points to, then attach.
        import json

        from peft import PeftModel

        base_name = json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
        print(f"Detected LoRA adapter; base = {base_name}")
        base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(base, model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16)
    return model


def generate_reply(model, tokenizer, messages, device, max_new_tokens=200):
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to a tuned model or adapter dir")
    parser.add_argument("--prompt", default=None, help="One-shot prompt; omit for interactive mode")
    parser.add_argument("--system", default="You are a helpful assistant.")
    args = parser.parse_args()

    device = get_device()
    tokenizer = load_tokenizer(args.model if (Path(args.model) / "tokenizer_config.json").exists() else args.model)
    model = load_model(args.model)
    if device != "cuda":
        model.to(device)
    model.eval()

    if args.prompt:
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ]
        print("\nAssistant:", generate_reply(model, tokenizer, messages, device))
        return

    print("Interactive chat (Ctrl+C or 'exit' to quit).")
    history = [{"role": "system", "content": args.system}]
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if user.lower() in {"exit", "quit"}:
            break
        history.append({"role": "user", "content": user})
        reply = generate_reply(model, tokenizer, history, device)
        history.append({"role": "assistant", "content": reply})
        print("Assistant:", reply)


if __name__ == "__main__":
    main()
