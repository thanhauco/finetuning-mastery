"""Perplexity on a held-out set.

Perplexity = exp(mean cross-entropy). Lower is better: it measures how 'surprised'
the model is by held-out text. A quick, label-free sanity check that fine-tuning
didn't degrade language modeling.

Run:
    python modules/08_evaluation/perplexity.py --model outputs/06_sft
    python modules/08_evaluation/perplexity.py --model HuggingFaceTB/SmolLM2-135M-Instruct
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import get_device  # noqa: E402
from common.data_utils import format_with_chat_template, load_sample_dataset  # noqa: E402
from common.model_utils import DEFAULT_MODEL, load_tokenizer  # noqa: E402


def load_eval_model(model_dir: str):
    import torch
    from transformers import AutoModelForCausalLM

    adapter_cfg = Path(model_dir) / "adapter_config.json"
    if adapter_cfg.exists():
        import json

        from peft import PeftModel

        base = json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
        model = PeftModel.from_pretrained(model, model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    return model


def compute_perplexity(model, tokenizer, texts, device, max_length=512):
    import math

    import torch

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            labels = enc["input_ids"].clone()
            out = model(**enc, labels=labels)
            # out.loss is mean over tokens; weight by token count for a corpus-level mean.
            n_tokens = labels.numel()
            total_loss += out.loss.item() * n_tokens
            total_tokens += n_tokens
    mean_nll = total_loss / max(total_tokens, 1)
    return math.exp(mean_nll), mean_nll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    tokenizer = load_tokenizer(args.model)
    model = load_eval_model(args.model)
    if device != "cuda":
        model.to(device)

    splits = load_sample_dataset(seed=args.seed)
    texts = [format_with_chat_template(ex, tokenizer)["text"] for ex in splits["test"]]

    ppl, nll = compute_perplexity(model, tokenizer, texts, device)
    print(f"Model: {args.model}")
    print(f"Eval examples: {len(texts)}")
    print(f"Mean NLL (cross-entropy): {nll:.4f}")
    print(f"Perplexity: {ppl:.2f}  (lower is better)")


if __name__ == "__main__":
    main()
