"""Clean, format, split, and save an instruction dataset for fine-tuning.

Pipeline:
    load jsonl -> validate/clean -> dedupe -> chat-format -> split -> save

Run:
    python modules/02_data_preparation/clean_and_format.py
    python modules/02_data_preparation/clean_and_format.py --out data/prepared
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.data_utils import (  # noqa: E402
    SAMPLE_DATA,
    format_with_chat_template,
    read_jsonl,
    write_jsonl,
)
from common.model_utils import DEFAULT_MODEL, load_tokenizer  # noqa: E402

REQUIRED_FIELDS = ("instruction", "output")


def clean_rows(rows: list[dict], min_output_chars: int = 1, max_output_chars: int = 8000) -> list[dict]:
    """Drop rows missing required fields or with empty/oversized outputs."""
    cleaned = []
    dropped = 0
    for row in rows:
        if not all(row.get(f) for f in REQUIRED_FIELDS):
            dropped += 1
            continue
        out = row["output"].strip()
        if not (min_output_chars <= len(out) <= max_output_chars):
            dropped += 1
            continue
        row = {**row, "instruction": row["instruction"].strip(), "output": out,
               "input": (row.get("input") or "").strip()}
        cleaned.append(row)
    print(f"  cleaned: kept {len(cleaned)}, dropped {dropped}")
    return cleaned


def dedupe(rows: list[dict]) -> list[dict]:
    """Remove exact duplicates by hashing the (instruction, input, output) tuple."""
    seen: set[str] = set()
    unique = []
    for row in rows:
        key = hashlib.sha256(
            f"{row['instruction']}||{row.get('input','')}||{row['output']}".encode("utf-8")
        ).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(row)
    print(f"  dedupe: {len(rows)} -> {len(unique)} unique")
    return unique


def length_stats(rows: list[dict], tokenizer) -> None:
    lengths = []
    for row in rows:
        text = format_with_chat_template(row, tokenizer)["text"]
        lengths.append(len(tokenizer(text)["input_ids"]))
    if not lengths:
        return
    lengths.sort()
    n = len(lengths)
    print(
        f"  token lengths -> min {lengths[0]} | "
        f"median {lengths[n // 2]} | max {lengths[-1]} | mean {sum(lengths) / n:.1f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=SAMPLE_DATA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="data/prepared")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random

    random.seed(args.seed)

    print(f"Loading: {args.input}")
    rows = read_jsonl(args.input)
    print(f"  loaded {len(rows)} rows")

    rows = clean_rows(rows)
    rows = dedupe(rows)

    tokenizer = load_tokenizer(args.model)
    length_stats(rows, tokenizer)

    # Render each row to the model's chat template -> a single 'text' field.
    formatted = []
    for row in rows:
        item = format_with_chat_template(row, tokenizer)
        formatted.append(item)

    # Shuffle then split into train/val/test with no overlap.
    random.shuffle(formatted)
    n = len(formatted)
    n_test = int(n * args.test_fraction)
    n_val = int(n * args.val_fraction)
    test = formatted[:n_test]
    val = formatted[n_test:n_test + n_val]
    train = formatted[n_test + n_val:]

    out_dir = Path(args.out)
    write_jsonl(str(out_dir / "train.jsonl"), train)
    write_jsonl(str(out_dir / "val.jsonl"), val)
    write_jsonl(str(out_dir / "test.jsonl"), test)

    print(f"\nSaved to {out_dir}/")
    print(f"  train: {len(train)} | val: {len(val)} | test: {len(test)}")
    print("\nSpot-check the first formatted example:\n")
    if train:
        print(train[0]["text"])


if __name__ == "__main__":
    main()
