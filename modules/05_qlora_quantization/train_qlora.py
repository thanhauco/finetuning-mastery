"""QLoRA: 4-bit quantized base model + LoRA adapters.

Requires an NVIDIA GPU with bitsandbytes installed. The base model is loaded in
4-bit NF4 (frozen); only the LoRA adapters train. This lets you fine-tune models
several times larger than your VRAM would normally allow.

Run (GPU):
    python modules/05_qlora_quantization/train_qlora.py --model HuggingFaceTB/SmolLM2-360M
    python modules/05_qlora_quantization/train_qlora.py --model mistralai/Mistral-7B-v0.3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import TrainDefaults, get_device, set_seed  # noqa: E402
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
    parser.add_argument("--output-dir", default="outputs/05_qlora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=d.per_device_train_batch_size)
    parser.add_argument("--max-seq-length", type=int, default=d.max_seq_length)
    parser.add_argument("--seed", type=int, default=d.seed)
    args = parser.parse_args()

    if get_device() != "cuda":
        print(
            "WARNING: QLoRA needs an NVIDIA GPU + bitsandbytes.\n"
            "No CUDA device detected. Read this script for the pattern, then use\n"
            "Module 04 (plain LoRA) on CPU/MPS instead."
        )

    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    set_seed(args.seed)
    tokenizer = load_tokenizer(args.model)

    # Load the base model in 4-bit NF4 (see common/model_utils -> quantized=True).
    model = load_causal_lm(args.model, quantized=True)

    # Make the k-bit model trainable: enables grad checkpointing, casts norms, etc.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    print_trainable_summary(model)

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
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",  # memory-friendly optimizer from bitsandbytes
        bf16=get_device() == "cuda",
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

    print("\nStarting QLoRA fine-tuning...")
    trainer.train()
    metrics = trainer.evaluate()
    print(f"\nFinal eval loss: {metrics.get('eval_loss'):.4f}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved QLoRA adapter to {args.output_dir}")
    print("Merge with Module 04's merge_adapter.py for standalone deployment.")


if __name__ == "__main__":
    main()
