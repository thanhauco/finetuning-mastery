# Module 07 — Preference Optimization (DPO & ORPO)

After SFT, a model follows instructions — but it doesn't yet know which of two
valid answers humans *prefer*. Preference optimization fixes that. In 2026 the
practical default is **DPO** (and its single-stage cousin **ORPO**), which replaced
most classic PPO-style RLHF because they're simpler and need **no reward model**.

## The mental model
```
   Preference data:  { prompt, chosen, rejected }

   DPO objective (intuition):
     push  P(chosen | prompt)  UP
     push  P(rejected | prompt) DOWN
     ...while staying close to the reference (SFT) model  ← the β·KL leash
```

## DPO vs ORPO
| | DPO | ORPO |
|---|-----|------|
| Stages | needs an SFT model first (used as reference) | **single stage** — combines SFT + preference |
| Reference model | yes (frozen copy) | none |
| Memory | higher (2 models) | lower (1 model) |
| Use when | you already have a good SFT checkpoint | you want one-shot SFT+align from a base model |

## What you'll learn
1. Load `{prompt, chosen, rejected}` preference data.
2. Train with TRL's `DPOTrainer` (with LoRA).
3. Train with `ORPOTrainer` as a single-stage alternative.
4. The role of **β** (KL strength) — how far you let the model drift from the reference.

## Files
| File | Purpose |
|------|---------|
| `train_dpo.py` | DPO alignment on top of an SFT/base model (LoRA). |
| `train_orpo.py` | ORPO single-stage preference tuning (LoRA). |

## Run
```bash
# DPO (ideally start from your Module 06 SFT model)
python modules/07_preference_optimization/train_dpo.py --model outputs/06_sft

# ORPO from a base model (no SFT step required)
python modules/07_preference_optimization/train_orpo.py
```

## Data
Uses [`data/sample_preferences.jsonl`](../../data/sample_preferences.jsonl) with the
standard `{prompt, chosen, rejected}` schema.

## Beyond DPO/ORPO (context)
PPO-style RLHF, KTO, and RLVR (verifiable rewards) still matter for frontier work,
but **DPO/ORPO are the right starting point** for almost everyone. See Module 10's
README for pointers to the heavier methods.
