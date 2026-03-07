# Module 03 — Full Fine-Tuning

Update **every** weight in the model using the Hugging Face `Trainer`. This is the
"classic" approach: maximum flexibility, maximum cost.

## What you'll learn
1. The `Trainer` + `TrainingArguments` workflow (the backbone of most HF training).
2. Tokenizing a dataset with a `DataCollator` for causal LM.
3. Evaluating during training and saving checkpoints.

## When to use full FT (2026 guidance)
✅ You have ample GPU memory (or a small model).
✅ You need deep behavior change, not just a task adapter.
✅ You're distilling or continuing pretraining on a new domain/language.

❌ Otherwise prefer **LoRA/QLoRA** (Modules 04–05) — usually 90%+ of the quality
at a fraction of the memory, and far easier to ship.

## Files
| File | Purpose |
|------|---------|
| `train_full.py` | Full fine-tune of a tiny causal LM with `Trainer`. |

## Run
```bash
# Tiny model, CPU/GPU, finishes quickly
python modules/03_full_finetuning/train_full.py --epochs 1

# Scale up on a GPU
python modules/03_full_finetuning/train_full.py --model HuggingFaceTB/SmolLM2-360M --epochs 2
```

## Memory math (rule of thumb)
Full FT with Adam needs roughly **~16 bytes/parameter** (weights + grads + 2 optimizer
states in fp32-equivalent). A 7B model ≈ **112 GB** just for that — which is why LoRA exists.
