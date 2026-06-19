"""Memory-saving knobs you can apply on a single GPU.

Demonstrates the two cheapest, highest-impact tricks:
  1. Gradient checkpointing  — recompute activations in backward to save memory.
  2. Gradient accumulation   — simulate a large batch with small per-step batches.

It prints the effective batch size and (if CUDA) peak memory with/without
gradient checkpointing so you can see the trade-off.

Run:
    python modules/10_advanced/memory_tricks.py
    python modules/10_advanced/memory_tricks.py --model HuggingFaceTB/SmolLM2-360M
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import get_device, set_seed  # noqa: E402
from common.model_utils import DEFAULT_MODEL, load_causal_lm, load_tokenizer  # noqa: E402


def measure_step(model, tokenizer, device, use_checkpointing: bool) -> tuple[float, float]:
    import torch

    if use_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.gradient_checkpointing_disable()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    text = ["Fine-tuning at scale needs careful memory management. " * 8]
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    labels = enc["input_ids"].clone()

    model.train()
    out = model(**enc, labels=labels)
    out.loss.backward()
    model.zero_grad(set_to_none=True)

    peak_mb = (torch.cuda.max_memory_allocated() / 1024**2) if device == "cuda" else float("nan")
    return out.loss.item(), peak_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--per-device-batch", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model)
    if device != "cuda":
        model.to(device)

    effective = args.per_device_batch * args.accum_steps
    print(f"Device: {device}")
    print("\nGradient accumulation")
    print(f"  per-device batch:        {args.per_device_batch}")
    print(f"  accumulation steps:      {args.accum_steps}")
    print(f"  EFFECTIVE batch size:    {effective}")
    print("  -> Set gradient_accumulation_steps in TrainingArguments to get this for free.")

    print("\nGradient checkpointing (peak memory)")
    loss_off, mem_off = measure_step(model, tokenizer, device, use_checkpointing=False)
    loss_on, mem_on = measure_step(model, tokenizer, device, use_checkpointing=True)
    if device == "cuda":
        print(f"  OFF: peak {mem_off:.0f} MB")
        print(f"  ON : peak {mem_on:.0f} MB  ({100 * (1 - mem_on / max(mem_off, 1e-6)):.0f}% saved)")
        print("  Trade-off: ~20-30% slower per step in exchange for the memory.")
    else:
        print("  (Peak-memory readout requires CUDA; the API calls are shown above.)")
        print(f"  loss sanity: {loss_off:.3f} (off) vs {loss_on:.3f} (on) — should be ~equal")

    print("\nStack these with QLoRA (Module 05) + 8-bit optimizers for max savings.")


if __name__ == "__main__":
    main()
