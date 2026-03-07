"""Full fine-tuning with the Hugging Face Trainer.

Every parameter is updated. This is the reference workflow that LoRA/QLoRA
later optimize. Uses a tiny model + the bundled dataset so it runs anywhere.

Run:
    python modules/03_full_finetuning/train_full.py --epochs 1
    python modules/03_full_finetuning/train_full.py --model HuggingFaceTB/SmolLM2-360M
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
    parser.add_argument("--output-dir", default="outputs/03_full")
    parser.add_argument("--epochs", type=float, default=d.num_train_epochs)
    parser.add_argument("--lr", type=float, default=5e-5)  # lower LR for full FT
    parser.add_argument("--batch-size", type=int, default=d.per_device_train_batch_size)
    parser.add_argument("--max-seq-length", type=int, default=d.max_seq_length)
    parser.add_argument("--seed", type=int, default=d.seed)
    args = parser.parse_args()

    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    set_seed(args.seed)
    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model)
    print_trainable_summary(model)  # full FT -> ~100% trainable

    # 1) Build dataset and render to a single 'text' field via chat template.
    splits = load_sample_dataset(seed=args.seed)

    def to_text(example):
        return format_with_chat_template(example, tokenizer)

    splits = splits.map(to_text, remove_columns=splits["train"].column_names)

    # 2) Tokenize.
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
        )

    tokenized = splits.map(tokenize, batched=True, remove_columns=["text"])

    # 3) Collator builds labels for causal LM (mlm=False -> next-token prediction).
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4) Training arguments.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
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

    print("\nStarting full fine-tuning...")
    trainer.train()

    metrics = trainer.evaluate()
    print(f"\nFinal eval loss: {metrics.get('eval_loss'):.4f}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved full fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()
