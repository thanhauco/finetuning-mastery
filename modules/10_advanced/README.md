# Module 10 — Advanced Topics

You've covered the full pipeline. This module is your map to scaling up and going
deeper. It's mostly **reference + runnable config**, because these techniques need
real multi-GPU hardware.

## 1. Multi-GPU & multi-node training

When a model (or batch) no longer fits on one GPU, shard it.

| Technique | What it shards | Use when |
|-----------|----------------|----------|
| **DDP** (Distributed Data Parallel) | data only (full model per GPU) | model fits on 1 GPU, you want speed |
| **FSDP** (Fully Sharded Data Parallel) | params + grads + optimizer states | model too big for 1 GPU |
| **DeepSpeed ZeRO-1/2/3** | optimizer / +grads / +params | very large models, mature tooling |

```
   ZeRO-3 / FSDP shard everything across N GPUs:

   GPU0  GPU1  GPU2  GPU3
   ┌──┐  ┌──┐  ┌──┐  ┌──┐
   │W₀│  │W₁│  │W₂│  │W₃│   each GPU holds 1/N of params,
   └──┘  └──┘  └──┘  └──┘   gathers the rest just-in-time
```

Launch with `accelerate`:
```bash
accelerate config           # interactive: pick FSDP or DeepSpeed
accelerate launch modules/03_full_finetuning/train_full.py --model <big-model>
```

See [`accelerate_fsdp.md`](accelerate_fsdp.md) and [`deepspeed_zero3.md`](deepspeed_zero3.md)
for ready-to-use configs.

## 2. Memory-saving techniques (single GPU too)
| Technique | Saves | Cost |
|-----------|-------|------|
| Gradient checkpointing | activations memory | ~20–30% slower |
| Gradient accumulation | lets you simulate big batches | more steps |
| 8-bit / paged optimizers | optimizer-state memory | negligible |
| QLoRA (Module 05) | base-weight memory | tiny quality hit |
| Flash Attention / SDPA | attention memory + speed | needs supported GPU |

## 3. Long-context fine-tuning
- Train at the **sequence length you'll serve** — short-context tuning then long-context
  inference degrades quality.
- Use **packing** (Module 02) to fill long windows efficiently.
- Watch attention memory: it grows with sequence length; Flash Attention helps a lot.
- RoPE scaling (`rope_scaling`) extends context beyond the pretrained window.

## 4. Beyond DPO — heavier alignment (when you need it)
| Method | Idea | When |
|--------|------|------|
| **PPO RLHF** | online RL with a reward model | frontier alignment, costly |
| **KTO** | uses thumbs-up/down (unpaired) signals | when you only have binary feedback |
| **RLVR** | reward = verifiable correctness (tests pass, exact answer) | math/code/reasoning |
| **Distillation** | student mimics a stronger teacher's outputs | shrink a big model cheaply |

## Files
| File | Purpose |
|------|---------|
| `accelerate_fsdp.md` | FSDP config + launch instructions. |
| `deepspeed_zero3.md` | DeepSpeed ZeRO-3 config + launch instructions. |
| `memory_tricks.py` | Demonstrates gradient checkpointing + accumulation knobs. |

## Where to go next
- Reproduce a small end-to-end run: **02 → 05 → 06 → 07 → 08 → 09**.
- Swap in your **own dataset** (match the data contracts in `ARCHITECTURE.md`).
- Scale the base model up and move to FSDP/DeepSpeed here in Module 10.
