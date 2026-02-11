"""03 — What actually changes during fine-tuning?

We answer three questions concretely:
  1. How many parameters does the model have, and where do they live?
  2. What does a single gradient step do to a specific weight?
  3. Why does freezing parameters (the idea behind LoRA) save so much?

Run:
    python modules/01_foundations/03_what_changes.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import get_device, set_seed  # noqa: E402
from common.model_utils import (  # noqa: E402
    DEFAULT_MODEL,
    count_trainable_parameters,
    load_causal_lm,
    load_tokenizer,
)


def parameter_breakdown(model, top_k: int = 5) -> None:
    print("\nLargest parameter tensors:")
    sizes = [(name, p.numel(), tuple(p.shape)) for name, p in model.named_parameters()]
    sizes.sort(key=lambda x: x[1], reverse=True)
    for name, numel, shape in sizes[:top_k]:
        print(f"  {numel:>12,}  {str(shape):<20} {name}")


def one_gradient_step(model, tokenizer, device) -> None:
    import torch

    # Pick one weight to watch before/after a step.
    watched_name, watched = next(iter(model.named_parameters()))
    before = watched.detach().flatten()[:5].clone()

    text = ["Fine-tuning changes the model's weights a little bit."]
    enc = tokenizer(text, return_tensors="pt", padding=True).to(device)
    labels = enc["input_ids"].clone()

    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)  # big LR to see movement
    model.train()
    optimizer.zero_grad()
    loss = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels).loss
    loss.backward()

    grad = watched.grad
    grad_preview = grad.detach().flatten()[:5] if grad is not None else None

    optimizer.step()
    after = watched.detach().flatten()[:5].clone()

    print(f"\nWatched weight: {watched_name}  shape={tuple(watched.shape)}")
    print(f"  loss:            {loss.item():.4f}")
    print(f"  before:          {before.tolist()}")
    print(f"  gradient:        {grad_preview.tolist() if grad_preview is not None else None}")
    print(f"  after one step:  {after.tolist()}")
    print("  delta = -lr * gradient  →  weights moved to reduce the loss.")


def freezing_demo(model) -> None:
    import torch

    trainable, total = count_trainable_parameters(model)
    print(f"\nAll trainable: {trainable:,} / {total:,} (100%)")

    # Freeze everything except the LM head — a crude preview of parameter-efficient FT.
    for p in model.parameters():
        p.requires_grad = False
    head = getattr(model, "lm_head", None) or getattr(model, "get_output_embeddings", lambda: None)()
    if head is not None:
        for p in head.parameters():
            p.requires_grad = True

    trainable, total = count_trainable_parameters(model)
    pct = 100 * trainable / max(total, 1)
    print(f"Only head trainable: {trainable:,} / {total:,} ({pct:.2f}%)")
    print("LoRA (Module 04) takes this idea further: freeze the base, train tiny adapters.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model)
    if device != "cuda":
        model.to(device)

    trainable, total = count_trainable_parameters(model)
    print(f"Model: {args.model}")
    print(f"Total parameters: {total:,}  (~{total/1e6:.1f}M)")
    print(f"Trainable now:    {trainable:,}")

    parameter_breakdown(model)
    one_gradient_step(model, tokenizer, device)
    freezing_demo(model)


if __name__ == "__main__":
    main()
