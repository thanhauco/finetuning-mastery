# Module 06 — Instruction & Chat Tuning (SFT)

**Supervised Fine-Tuning (SFT)** teaches a base model to *follow instructions* and
*chat*. This is the first alignment stage and uses TRL's `SFTTrainer`, which wraps
all the boilerplate from Modules 03–05 (formatting, packing, collation).

## What you'll learn
1. Use `SFTTrainer` + `SFTConfig` — the modern, concise SFT path.
2. Train on **chat-formatted** data with the model's chat template.
3. **Completion-only loss**: only compute loss on the assistant's reply, not the prompt.
4. Combine SFT with LoRA for a parameter-efficient instruction tune.

## Why completion-only loss matters
You don't want the model to "learn" to generate the *user's* question — only the
*answer*. Masking the prompt tokens focuses the gradient on the response.

```
   <user> What is 2+2? <assistant> 4
   └────── masked ─────┘ └─ loss here ─┘
```

## Files
| File | Purpose |
|------|---------|
| `train_sft.py` | Instruction tuning with `SFTTrainer` (+ optional LoRA). |
| `chat_inference.py` | Load the tuned model and chat with it. |

## Run
```bash
# LoRA SFT (default, light)
python modules/06_instruction_tuning/train_sft.py

# Full SFT (no adapters)
python modules/06_instruction_tuning/train_sft.py --no-lora

# Talk to the result
python modules/06_instruction_tuning/chat_inference.py --model outputs/06_sft
```

## The 2026 default recipe
`QLoRA + SFTTrainer(packing=True) + completion-only loss` is the bread-and-butter
instruction-tuning setup. Follow it with DPO/ORPO (Module 07) for preference alignment.
