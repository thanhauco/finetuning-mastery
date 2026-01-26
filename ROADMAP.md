# 🗺️ Roadmap — Fine-Tuning Mastery (2026)

This roadmap is both a **learning path** for you and a **build plan** for this repo.
Modules are released in order; check the status column to see what's runnable today.

---

## Legend

```
[x] done & runnable      [~] in progress      [ ] planned
```

---

## Phase 0 — Setup & Mental Model

```
[x] README.md ............... curriculum overview + method cheat sheet
[x] SETUP.md ................ environment, CUDA, Windows/WSL, Colab
[x] requirements.txt ........ pinned 2026 stack
[x] common/ ................. shared config, data, model helpers
[x] data/ ................... tiny instruction + preference datasets
```

## Phase 1 — Foundations

```
[~] 01 Foundations
    [x] README
    [ ] 01_hello_finetuning.py .. raw PyTorch training loop (CPU-friendly)
    [ ] 02_tokenization.py ...... tokens, special tokens, chat templates
    [ ] 03_what_changes.py ...... inspect params & a single gradient step
```

## Phase 2 — Data

```
[ ] 02 Data Preparation
    [ ] clean & dedupe
    [ ] chat-template formatting
    [ ] sequence packing
    [ ] train/val/test splitting
```

## Phase 3 — Core Fine-Tuning Methods

```
[ ] 03 Full Fine-Tuning ....... Trainer API, all weights
[ ] 04 LoRA / PEFT ............ adapters, target modules, merging
[ ] 05 QLoRA / Quantization ... 4-bit NF4 on a single GPU
```

## Phase 4 — Behavior & Alignment

```
[ ] 06 Instruction / Chat SFT . SFTTrainer, completion-only loss
[ ] 07 Preference Optimization  DPO, ORPO (RLHF successors)
```

## Phase 5 — Quality & Production

```
[ ] 08 Evaluation ............. perplexity, task metrics, LLM-as-judge
[ ] 09 Deployment & Serving ... merge, quantize, vLLM server
```

## Phase 6 — Scaling Up

```
[ ] 10 Advanced ............... FSDP, DeepSpeed ZeRO-3, long-context, packing
```

---

## Skill Progression Map

```
 Beginner            Intermediate              Advanced
 ─────────           ──────────────            ──────────
 tokenization   ──►  LoRA / QLoRA        ──►   DPO / ORPO alignment
 training loop  ──►  SFT on chat data    ──►   multi-GPU FSDP / DeepSpeed
 data formats   ──►  evaluation harness  ──►   long-context + packing
                     adapter merging     ──►   production serving (vLLM)
```

---

## Suggested Timeline (self-paced)

| Week | Focus | Outcome |
|------|-------|---------|
| 1 | Phase 0–1 | Understand tokens, loss, a training loop |
| 2 | Phase 2–3 | Run your first LoRA / QLoRA fine-tune |
| 3 | Phase 4 | Build an instruction-tuned + DPO-aligned model |
| 4 | Phase 5–6 | Evaluate, serve, and scale to bigger models |

---

## Stretch Goals (post-curriculum)

```
[ ] Tool-calling / function-calling fine-tunes
[ ] Multimodal (vision-language) adapters
[ ] Reasoning distillation (teacher → student)
[ ] RLVR / verifiable-reward fine-tuning
[ ] On-device export (GGUF / quantized) for local inference
```
