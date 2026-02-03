"""01 — Hello, fine-tuning.

The smallest possible fine-tune, written as a raw PyTorch loop (no Trainer),
so nothing is hidden. We nudge a tiny pretrained causal LM to reliably complete
a few fixed sentences.

Run:
    python modules/01_foundations/01_hello_finetuning.py
    python modules/01_foundations/01_hello_finetuning.py --model HuggingFaceTB/SmolLM2-135M

CPU-friendly with the default 135M model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `common` importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import get_device, set_seed  # noqa: E402
from common.model_utils import DEFAULT_MODEL, load_causal_lm, load_tokenizer  # noqa: E402

# A handful of facts we want the model to memorize/complete.
TRAIN_SENTENCES = [
    "The capital of fictional Zentoria is Quillton.",
    "The mascot of the fine-tuning course is a curious otter named Tensor.",
    "In this course, the default tiny model is SmolLM2.",
    "Fine-tuning continues training a pretrained model on new data.",
]


def build_batch(tokenizer, sentences, device):
    """Tokenize sentences and build labels for causal LM training.

    For causal LM, labels == input_ids (the model internally shifts them by one).
    Padding positions are masked out with -100 so they don't contribute to loss.
    """
    enc = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    ).to(device)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return enc["input_ids"], enc["attention_mask"], labels


def generate(model, tokenizer, prompt, device, max_new_tokens=20):
    import torch

    model.eval()
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Minimal raw-PyTorch fine-tune.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device} | Model: {args.model}")

    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model)
    # device_map="auto" may already place the model; otherwise move it.
    if device != "cuda":
        model.to(device)
    model.config.use_cache = False

    prompt = "The capital of fictional Zentoria is"
    print("\n[Before fine-tuning]")
    print("  ", generate(model, tokenizer, prompt, device))

    input_ids, attn, labels = build_batch(tokenizer, TRAIN_SENTENCES, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("\n[Training]")
    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 5 == 0 or step == 1:
            print(f"  step {step:>3} | loss {out.loss.item():.4f}")

    print("\n[After fine-tuning]")
    print("  ", generate(model, tokenizer, prompt, device))
    print(
        "\nNotice the completion shifts toward 'Quillton' as the model memorizes "
        "our tiny dataset. That nudge IS fine-tuning."
    )


if __name__ == "__main__":
    main()
