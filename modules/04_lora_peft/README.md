# Module 04 — LoRA / PEFT

**LoRA (Low-Rank Adaptation)** freezes the pretrained weights and trains tiny
low-rank "adapter" matrices instead. You update ~0.1–1% of parameters but get
most of the quality of full fine-tuning — and you can ship many adapters from one
base model.

## The idea in one line
For a frozen weight `W`, learn a small update `ΔW = B·A` where `A` and `B` are
low-rank (rank `r`). At inference: `h = W·x + (B·A)·x`.

```
   frozen W  ❄          trainable A (r×d), B (d×r)  🔥
        │                         │
        └────────►  h = W·x  +  (B·A)·x   ◄─── only A,B get gradients
```

## What you'll learn
1. Wrap a model with a `LoraConfig` using `peft`.
2. Pick `target_modules`, `r`, `lora_alpha`, `lora_dropout`.
3. Train, save the **adapter only** (a few MB), and reload it.
4. **Merge** the adapter into the base for single-file deployment.

## Files
| File | Purpose |
|------|---------|
| `train_lora.py` | Fine-tune with LoRA adapters and save them. |
| `merge_adapter.py` | Merge a trained adapter into the base model weights. |

## Run
```bash
python modules/04_lora_peft/train_lora.py --model HuggingFaceTB/SmolLM2-135M-Instruct
python modules/04_lora_peft/merge_adapter.py --adapter outputs/04_lora --out outputs/04_merged
```

## Hyperparameter intuition
| Param | Typical | Effect |
|-------|---------|--------|
| `r` (rank) | 8–64 | Capacity of the adapter. Higher = more expressive, more params. |
| `lora_alpha` | `2*r` is a common default | Scales the update. Effective scale = `alpha / r`. |
| `lora_dropout` | 0.0–0.1 | Regularization on the adapter path. |
| `target_modules` | attention + MLP proj | Which linear layers get adapters. `"all-linear"` is a safe default. |

## Ship many tasks, one base
Because adapters are tiny and swappable, the modern pattern is:
**one base model in memory + N hot-swappable LoRA adapters** for N tasks/customers.
