"""Task metrics: generate answers and score them against references.

Two metrics:
  • exact_match  — normalized string equality (strict).
  • token_f1     — overlap of word tokens (partial credit), like SQuAD F1.

Run:
    python modules/08_evaluation/task_metrics.py --model outputs/06_sft
"""
from __future__ import annotations

import argparse
import re
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import get_device  # noqa: E402
from common.data_utils import load_sample_dataset, to_chat_messages  # noqa: E402
from common.model_utils import DEFAULT_MODEL, load_tokenizer  # noqa: E402


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(pred: str, ref: str) -> float:
    return float(normalize(pred) == normalize(ref))


def token_f1(pred: str, ref: str) -> float:
    p, r = normalize(pred).split(), normalize(ref).split()
    if not p or not r:
        return float(p == r)
    common = {}
    for tok in p:
        if tok in r:
            common[tok] = min(p.count(tok), r.count(tok))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def load_eval_model(model_dir: str):
    import torch
    from transformers import AutoModelForCausalLM

    adapter_cfg = Path(model_dir) / "adapter_config.json"
    if adapter_cfg.exists():
        import json

        from peft import PeftModel

        base = json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
        return PeftModel.from_pretrained(model, model_dir)
    return AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)


def generate(model, tokenizer, user_msg, device, max_new_tokens=64):
    import torch

    messages = [{"role": "user", "content": user_msg}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=20)
    args = parser.parse_args()

    device = get_device()
    tokenizer = load_tokenizer(args.model)
    model = load_eval_model(args.model)
    if device != "cuda":
        model.to(device)
    model.eval()

    splits = load_sample_dataset(seed=args.seed)
    eval_rows = list(splits["test"])[: args.max_samples]

    em_scores, f1_scores = [], []
    print(f"Evaluating {len(eval_rows)} examples...\n")
    for row in eval_rows:
        user_msg = to_chat_messages(row)[0]["content"]
        reference = row["output"]
        prediction = generate(model, tokenizer, user_msg, device)
        em = exact_match(prediction, reference)
        f1 = token_f1(prediction, reference)
        em_scores.append(em)
        f1_scores.append(f1)
        print(f"Q: {user_msg[:60]}")
        print(f"  ref:  {reference[:80]}")
        print(f"  pred: {prediction[:80]}")
        print(f"  EM={em:.0f}  F1={f1:.2f}\n")

    n = len(em_scores)
    print("=" * 50)
    print(f"Exact Match: {sum(em_scores)/n:.3f}")
    print(f"Token F1:    {sum(f1_scores)/n:.3f}")


if __name__ == "__main__":
    main()
