# 🏛️ Architecture

How this repository is organized and how the pieces fit together at **train** and
**inference** time. Diagrams are ASCII so they render anywhere.

---

## 1. Repository Layout

```
finetuning-mastery/
│
├── README.md              ← curriculum + method cheat sheet
├── ROADMAP.md             ← learning path & build status
├── ARCHITECTURE.md        ← this file
├── SETUP.md               ← environment setup
├── requirements.txt
│
├── common/                ← shared, reused by every module
│   ├── config.py          ← seed, device, dtype, TrainDefaults
│   ├── data_utils.py      ← jsonl IO, chat templating, splits
│   └── model_utils.py     ← tokenizer/model loading, param counts
│
├── data/                  ← tiny bundled datasets (run anywhere)
│   ├── sample_instructions.jsonl
│   └── sample_preferences.jsonl
│
└── modules/               ← one folder per lesson, self-contained
    ├── 01_foundations/
    ├── 02_data_preparation/
    ├── 03_full_finetuning/
    ├── 04_lora_peft/
    ├── 05_qlora_quantization/
    ├── 06_instruction_tuning/
    ├── 07_preference_optimization/
    ├── 08_evaluation/
    ├── 09_deployment_serving/
    └── 10_advanced/
```

**Design rule:** every module imports from `common/` but never from another module.
You can study any lesson in isolation.

---

## 2. The Fine-Tuning Pipeline (mental model)

```
   ┌──────────┐    ┌───────────────┐    ┌──────────────┐    ┌───────────────┐
   │   Raw    │    │   Formatting   │    │  Tokenizer   │    │  Base Model   │
   │  Data    │───►│ chat template  │───►│ text→token   │───►│ (pretrained)  │
   │ (jsonl)  │    │ + packing      │    │   ids        │    │   weights θ   │
   └──────────┘    └───────────────┘    └──────────────┘    └───────┬───────┘
        Module 02                                                    │
                                                                     ▼
                            ┌────────────────────────────────────────────────┐
                            │                 TRAINING LOOP                    │
                            │  forward → cross-entropy loss → backward → step  │
                            │                                                  │
                            │  Method picks WHICH weights update:             │
                            │    • Full FT  → all of θ        (Module 03)     │
                            │    • LoRA     → small adapters  (Module 04)     │
                            │    • QLoRA    → adapters + 4bit (Module 05)     │
                            └───────────────────────┬──────────────────────────┘
                                                    ▼
                            ┌────────────────────────────────────────────────┐
                            │          Tuned weights / adapter                │
                            │   SFT (06) → DPO/ORPO (07) → Eval (08)          │
                            └───────────────────────┬──────────────────────────┘
                                                    ▼
                            ┌────────────────────────────────────────────────┐
                            │     Merge → quantize → serve (Module 09)        │
                            └────────────────────────────────────────────────┘
```

---

## 3. Full Fine-Tuning vs. LoRA (where the gradients go)

```
   FULL FINE-TUNING                         LoRA / QLoRA
   ─────────────────                        ─────────────

   Pretrained weight W                      Pretrained weight W  (FROZEN ❄)
   (all trainable 🔥)                              │
        │                                          │  +  B·A   (trainable 🔥)
        ▼                                          ▼  small low-rank matrices
   h = W·x                                   h = W·x + (B·A)·x
                                                   ▲     ▲
   billions of params update            r-rank  ──┘     │
   high VRAM, high quality              ~0.1–1% of params update
                                        low VRAM, ship many adapters
```

`QLoRA` = the frozen `W` is stored in **4-bit NF4**, so even the frozen copy is cheap,
while the small `B·A` adapters train in higher precision.

---

## 4. SFT → Preference Optimization (alignment stages)

```
   ┌─────────────┐      ┌──────────────────┐      ┌────────────────────┐
   │  Base LM    │ SFT  │  Instruction-    │ DPO  │  Aligned model     │
   │ (next-token │ ───► │  following model │ ───► │ (prefers "chosen"  │
   │  predictor) │      │ (Module 06)      │ ORPO │  over "rejected")   │
   └─────────────┘      └──────────────────┘      └────────────────────┘
        learns               learns to                learns human
        language             follow tasks             preferences
                                                  (no reward model needed)

   Data shape per stage:
     SFT  : { instruction, input, output }
     DPO  : { prompt, chosen, rejected }
```

---

## 5. Inference / Serving Path (Module 09)

```
   adapter + base ──► merge ──► (optional) quantize ──► load in server
        │                                                     │
        │                                                     ▼
        │                                          ┌──────────────────────┐
        │                                          │   vLLM / TGI engine  │
        │                                          │  KV-cache, batching  │
        │                                          └──────────┬───────────┘
        │                                                     ▼
        └────────────► OpenAI-compatible HTTP API ───►  client app
```

---

## 6. Data Contracts

Every script agrees on these JSON shapes so modules are interchangeable:

```
Instruction (SFT)            Preference (DPO/ORPO)
─────────────────           ─────────────────────
{                           {
  "instruction": str,         "prompt":   str,
  "input":  str | "",         "chosen":   str,
  "output": str               "rejected": str
}                           }
```

`common/data_utils.py` converts these into chat messages and applies the
model's `chat_template`, keeping formatting consistent across all lessons.
