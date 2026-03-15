"""Merge a trained LoRA adapter into the base model.

A merged model is a single standalone checkpoint (no PEFT needed at inference),
which is convenient for serving (Module 09). Trade-off: you lose the ability to
hot-swap adapters.

Run:
    python modules/04_lora_peft/merge_adapter.py \
        --base HuggingFaceTB/SmolLM2-135M-Instruct \
        --adapter outputs/04_lora \
        --out outputs/04_merged
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.model_utils import DEFAULT_MODEL, load_tokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_MODEL, help="Base model the adapter was trained on")
    parser.add_argument("--adapter", default="outputs/04_lora", help="Path to the saved LoRA adapter")
    parser.add_argument("--out", default="outputs/04_merged")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    print(f"Loading base model: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16)

    print(f"Attaching adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("Merging adapter weights into the base...")
    merged = model.merge_and_unload()  # folds B*A back into W

    Path(args.out).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out, safe_serialization=True)
    load_tokenizer(args.base).save_pretrained(args.out)
    print(f"Merged standalone model saved to {args.out}")
    print("You can now load it like any normal model — no PEFT required.")


if __name__ == "__main__":
    main()
