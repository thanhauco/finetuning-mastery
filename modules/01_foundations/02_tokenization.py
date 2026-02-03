"""02 — Tokenization & chat templates.

You can't fine-tune what you can't tokenize. This script makes the invisible
visible: how text becomes token ids, what special tokens exist, and how the
chat template turns a conversation into a single training string.

Run:
    python modules/01_foundations/02_tokenization.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.model_utils import DEFAULT_MODEL, load_tokenizer  # noqa: E402


def show_tokenization(tokenizer, text: str) -> None:
    ids = tokenizer(text)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(f"\nText: {text!r}")
    print(f"  #tokens: {len(ids)}")
    print(f"  ids:    {ids}")
    print(f"  tokens: {tokens}")
    print(f"  decoded back: {tokenizer.decode(ids)!r}")


def show_special_tokens(tokenizer) -> None:
    print("\nSpecial tokens")
    for name in ["bos_token", "eos_token", "pad_token", "unk_token"]:
        val = getattr(tokenizer, name, None)
        print(f"  {name:10}: {val!r}")
    print(f"  vocab size: {tokenizer.vocab_size}")


def show_chat_template(tokenizer) -> None:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is fine-tuning?"},
        {"role": "assistant", "content": "Continuing training on new data."},
    ]
    print("\nChat template (has_template =", bool(getattr(tokenizer, "chat_template", None)), ")")
    if getattr(tokenizer, "chat_template", None):
        # tokenize=False shows the literal formatted string the model trains on.
        rendered = tokenizer.apply_chat_template(messages, tokenize=False)
        print("--- rendered training string ---")
        print(rendered)
        print("--- end ---")
        # add_generation_prompt=True is what you use at INFERENCE time.
        infer = tokenizer.apply_chat_template(
            messages[:2], tokenize=False, add_generation_prompt=True
        )
        print("\nInference prompt (note the trailing assistant header):")
        print(infer)
    else:
        print("  This tokenizer has no chat template; use a manual format.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    print(f"Loaded tokenizer for: {args.model}")

    show_special_tokens(tokenizer)
    show_tokenization(tokenizer, "Fine-tuning is fun!")
    show_tokenization(tokenizer, "tokenization")        # often >1 subword piece
    show_tokenization(tokenizer, "antidisestablishmentarianism")
    show_chat_template(tokenizer)

    print(
        "\nTakeaways:\n"
        "  • Words split into subword pieces; rare words use more tokens.\n"
        "  • eos_token marks the end of a sequence — the model learns to stop.\n"
        "  • The chat template defines the exact format your model expects.\n"
        "    Always train and infer with the SAME template."
    )


if __name__ == "__main__":
    main()
