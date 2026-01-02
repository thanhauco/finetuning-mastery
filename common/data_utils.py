"""Dataset loading and chat-template formatting helpers."""
from __future__ import annotations

import json
import os
from typing import Iterable

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DATA = os.path.join(REPO_ROOT, "data", "sample_instructions.jsonl")


def read_jsonl(path: str) -> list[dict]:
    """Read a .jsonl file into a list of dicts."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_chat_messages(example: dict) -> list[dict]:
    """Convert an {instruction, input, output} row into chat messages.

    This is the standard format consumed by tokenizer.apply_chat_template.
    """
    user = example["instruction"]
    if example.get("input"):
        user = f"{user}\n\n{example['input']}"
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": example["output"]},
    ]


def format_with_chat_template(example: dict, tokenizer) -> dict:
    """Render a row to a single training string using the model's chat template.

    Falls back to a simple template if the tokenizer has none.
    """
    messages = to_chat_messages(example)
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        # Minimal fallback used by base (non-chat) models.
        text = (
            f"### Instruction:\n{messages[0]['content']}\n\n"
            f"### Response:\n{messages[1]['content']}{tokenizer.eos_token or ''}"
        )
    return {"text": text}


def load_sample_dataset(val_fraction: float = 0.2, seed: int = 42):
    """Load the bundled tiny instruction dataset as a DatasetDict with splits."""
    from datasets import Dataset

    rows = read_jsonl(SAMPLE_DATA)
    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=val_fraction, seed=seed)
    return split  # has "train" and "test"
