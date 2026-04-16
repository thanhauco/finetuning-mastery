"""Supervised Fine-Tuning (SFT) with TRL's SFTTrainer.

SFTTrainer handles chat formatting, optional packing, and collation for you, so
this is the cleanest way to instruction-tune a model. LoRA is on by default;
pass --no-lora for full fine-tuning.

Run:
    python modules/06_instruction_tuning/train_sft.py
    python modules/06_instruction_tuning/train_sft.py --no-lora
    python modules/06_instruction_tuning/train_sft.py --model HuggingFaceTB/SmolLM2-360M
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import TrainDefaults, set_seed  # noqa: E402
from common.data_utils import load_sample_dataset, to_chat_messages  # noqa: E402
from common.model_utils import (  # noqa: E402
    DEFAULT_MODEL,
    load_causal_lm,
    load_tokenizer,
)


def main():
    d = TrainDefaults()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="outputs/06_sft")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--no-lora", action="store_true", help="Full fine-tune instead of LoRA")
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=d.per_device_train_batch_size)
    parser.add_argument("--max-seq-length", type=int, default=d.max_seq_length)
    parser.add_argument("--seed", type=int, default=d.seed)
    args = parser.parse_args()

    from trl import SFTConfig, SFTTrainer

    set_seed(args.seed)
    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model)

    # Build a "messages" column; SFTTrainer applies the chat template itself.
    splits = load_sample_dataset(seed=args.seed)

    def to_messages(example):
        return {"messages": to_chat_messages(example)}

    splits = splits.map(to_messages, remove_columns=splits["train"].column_names)

    peft_config = None
    if not args.no_lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.r,
            lora_alpha=args.alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=d.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=d.warmup_ratio,
        logging_steps=d.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        max_length=args.max_seq_length,
        packing=False,            # set True for many short samples (Module 02)
        assistant_only_loss=True,  # completion-only: mask the prompt tokens
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    mode = "full" if args.no_lora else "LoRA"
    print(f"\nStarting SFT ({mode})...")
    trainer.train()
    metrics = trainer.evaluate()
    print(f"\nFinal eval loss: {metrics.get('eval_loss'):.4f}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved SFT model to {args.output_dir}")


if __name__ == "__main__":
    main()
