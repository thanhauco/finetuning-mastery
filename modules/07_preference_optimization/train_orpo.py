"""ORPO (Odds Ratio Preference Optimization) with TRL.

ORPO combines supervised fine-tuning and preference alignment into a SINGLE stage
and needs no reference model — so it's lighter than DPO. Train it directly from a
base model using {prompt, chosen, rejected} data.

Run:
    python modules/07_preference_optimization/train_orpo.py
    python modules/07_preference_optimization/train_orpo.py --model HuggingFaceTB/SmolLM2-360M
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import TrainDefaults, set_seed  # noqa: E402
from common.data_utils import read_jsonl  # noqa: E402
from common.model_utils import DEFAULT_MODEL, load_causal_lm, load_tokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PREF_DATA = REPO_ROOT / "data" / "sample_preferences.jsonl"


def load_preference_dataset(seed: int):
    from datasets import Dataset

    rows = read_jsonl(str(PREF_DATA))
    ds = Dataset.from_list(rows)
    return ds.train_test_split(test_size=0.2, seed=seed)


def main():
    d = TrainDefaults()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="outputs/07_orpo")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--beta", type=float, default=0.1, help="ORPO lambda (odds-ratio weight)")
    parser.add_argument("--batch-size", type=int, default=d.per_device_train_batch_size)
    parser.add_argument("--max-length", type=int, default=d.max_seq_length)
    parser.add_argument("--seed", type=int, default=d.seed)
    args = parser.parse_args()

    from peft import LoraConfig
    from trl import ORPOConfig, ORPOTrainer

    set_seed(args.seed)
    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model)

    dataset = load_preference_dataset(args.seed)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    orpo_config = ORPOConfig(
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
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,
        report_to="none",
        seed=args.seed,
    )

    trainer = ORPOTrainer(
        model=model,
        args=orpo_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print(f"\nStarting ORPO (single-stage, beta={args.beta})...")
    trainer.train()
    metrics = trainer.evaluate()
    print(f"\nFinal eval loss: {metrics.get('eval_loss'):.4f}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved ORPO-aligned adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
