"""Direct Preference Optimization (DPO) with TRL.

DPO aligns a model to human preferences using {prompt, chosen, rejected} triples,
without training a separate reward model. Ideally start from your SFT checkpoint
(Module 06); the SFT model also acts as the frozen reference.

Run:
    python modules/07_preference_optimization/train_dpo.py --model outputs/06_sft
    python modules/07_preference_optimization/train_dpo.py            # from tiny base
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

    rows = read_jsonl(str(PREF_DATA))  # keys: prompt, chosen, rejected
    ds = Dataset.from_list(rows)
    return ds.train_test_split(test_size=0.2, seed=seed)


def main():
    d = TrainDefaults()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SFT model dir or base model")
    parser.add_argument("--output-dir", default="outputs/07_dpo")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=5e-6)  # DPO uses a small LR
    parser.add_argument("--beta", type=float, default=0.1, help="KL strength vs reference")
    parser.add_argument("--batch-size", type=int, default=d.per_device_train_batch_size)
    parser.add_argument("--max-length", type=int, default=d.max_seq_length)
    parser.add_argument("--seed", type=int, default=d.seed)
    args = parser.parse_args()

    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

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

    dpo_config = DPOConfig(
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

    # With a peft_config, TRL builds the frozen reference internally (no ref model arg).
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print(f"\nStarting DPO (beta={args.beta})...")
    trainer.train()
    metrics = trainer.evaluate()
    print(f"\nFinal eval loss: {metrics.get('eval_loss'):.4f}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved DPO-aligned adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
