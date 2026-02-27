"""Sequence packing — fit more training signal into each step.

Short examples waste compute because padding tokens contribute nothing.
"Packing" concatenates multiple examples (separated by EOS) into fixed-length
blocks so almost every token is useful.

This script demonstrates the idea with plain Python so you can SEE the blocks.
In practice, TRL's SFTTrainer can do this for you with packing=True (Module 06).

Run:
    python modules/02_data_preparation/packing.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.data_utils import (  # noqa: E402
    SAMPLE_DATA,
    format_with_chat_template,
    read_jsonl,
)
from common.model_utils import DEFAULT_MODEL, load_tokenizer  # noqa: E402


def pack_sequences(token_lists: list[list[int]], block_size: int, eos_id: int) -> list[list[int]]:
    """Greedily concatenate token sequences into fixed-size blocks.

    Each example is followed by EOS, then the long stream is chopped into
    equal `block_size` chunks. The trailing remainder is dropped (standard).
    """
    stream: list[int] = []
    for toks in token_lists:
        stream.extend(toks)
        stream.append(eos_id)

    blocks = [stream[i:i + block_size] for i in range(0, len(stream), block_size)]
    # Drop the last block if it's shorter than block_size (keeps shapes uniform).
    if blocks and len(blocks[-1]) < block_size:
        blocks.pop()
    return blocks


def efficiency(token_lists: list[list[int]], block_size: int) -> None:
    # Padding approach: every example padded up to block_size.
    padded_tokens = len(token_lists) * block_size
    real_tokens = sum(len(t) for t in token_lists)
    waste = 100 * (1 - real_tokens / max(padded_tokens, 1))
    print(f"\nWithout packing (pad to {block_size}):")
    print(f"  real tokens: {real_tokens} / allocated {padded_tokens}  -> {waste:.1f}% wasted on padding")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    rows = read_jsonl(SAMPLE_DATA)

    token_lists = []
    for row in rows:
        text = format_with_chat_template(row, tokenizer)["text"]
        token_lists.append(tokenizer(text)["input_ids"])

    efficiency(token_lists, args.block_size)

    eos_id = tokenizer.eos_token_id
    blocks = pack_sequences(token_lists, args.block_size, eos_id)
    packed_tokens = len(blocks) * args.block_size
    real_tokens = sum(len(t) for t in token_lists) + len(token_lists)  # + EOS markers

    print(f"\nWith packing (block_size={args.block_size}):")
    print(f"  {len(rows)} examples -> {len(blocks)} dense blocks")
    print(f"  ~{min(100, 100 * real_tokens / max(packed_tokens, 1)):.1f}% of tokens are real signal")
    print("\nRule of thumb: enable packing when you have many SHORT examples.")
    print("Use SFTTrainer(packing=True) to get this automatically (Module 06).")


if __name__ == "__main__":
    main()
