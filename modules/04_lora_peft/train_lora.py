"""LoRA fine-tuning with PEFT.

Freeze the base model, train low-rank adapters, save just the adapter (a few MB).
Uses a tiny model + bundled dataset so it runs on modest hardware.

Run:
    python modules/04_lora_peft/train_lora.py
    python modules/04_lora_peft/train_lora.py --model HuggingFaceTB/SmolLM2-360M --r 32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import TrainDefaults, set_seed  # noqa: E402
from common.data_utils import format_with_chat_template, load_sample_dataset  # noqa: E402
from common.model_utils import (  # noqa: E402
    DEFAULT_MODEL,
    load_causal_lm,
    load_tokenizer,
    print_trainable_summary,
)


def main():
    d = TrainDefaults()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="outputs/04_lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)  # LoRA likes higher LR
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=d.per_device_train_batch_size)
    parser.add_argument("--max-seq-length", type=int, default=d.max_seq_length)
    parser.add_argument("--seed", type=int, default=d.seed)
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    set_seed(args.seed)
    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model)

    # LoRA configuration. "all-linear" targets every linear layer (robust default).
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    print_trainable_summary(model)  # expect a tiny % trainable

    # Data -> chat template -> tokenize.
    splits = load_sample_dataset(seed=args.seed)
    splits = splits.map(
        lambda ex: format_with_chat_template(ex, tokenizer),
        remove_columns=splits["train"].column_names,
    )
    tokenized = splits.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=args.max_seq_length),
        batched=True,
        remove_columns=["text"],
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
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
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=collator,
    )

    print("\nStarting LoRA fine-tuning...")
    trainer.train()
    metrics = trainer.evaluate()
    print(f"\nFinal eval loss: {metrics.get('eval_loss'):.4f}")

    # Saves ONLY the adapter weights + config (small!).
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")
    print("Reload with: PeftModel.from_pretrained(base_model, adapter_dir)")


if __name__ == "__main__":
    main()
