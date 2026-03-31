# Module 05 — QLoRA & Quantization

**QLoRA** = load the frozen base model in **4-bit** (NF4) precision, then train
LoRA adapters on top in higher precision. This is the single most important
technique for fine-tuning large models on consumer/single GPUs.

## Why it works
```
   Base weights W stored in 4-bit NF4  ❄   (≈ 0.5 bytes / param)
        │
        └────► h = dequant(W)·x  +  (B·A)·x   🔥  adapters train in bf16
```
You get ~full-fine-tune quality while fitting a model **~4× larger** in the same VRAM.

## VRAM ballpark (very rough)
| Model | Full FT (Adam) | QLoRA |
|-------|----------------|-------|
| 7B    | ~112 GB        | ~6–10 GB |
| 13B   | ~208 GB        | ~12–16 GB |
| 70B   | impossible on 1 GPU | ~46 GB (1× A100 80GB) |

## What you'll learn
1. Configure 4-bit loading with `BitsAndBytesConfig` (NF4 + double quant).
2. `prepare_model_for_kbit_training` to make a quantized model trainable.
3. Train LoRA adapters on top — identical to Module 04 otherwise.

## Files
| File | Purpose |
|------|---------|
| `train_qlora.py` | 4-bit base + LoRA adapters. |

## Run (NVIDIA GPU required — bitsandbytes)
```bash
python modules/05_qlora_quantization/train_qlora.py --model HuggingFaceTB/SmolLM2-360M
# Scale to a real target on a 16–24GB GPU, e.g.:
# python modules/05_qlora_quantization/train_qlora.py --model mistralai/Mistral-7B-v0.3
```

> 💡 No NVIDIA GPU? Study `train_qlora.py`, then use Module 04 (plain LoRA) on CPU/MPS.

## Key knobs
- `bnb_4bit_quant_type="nf4"` — NormalFloat4, better than plain int4 for weights.
- `bnb_4bit_use_double_quant=True` — quantizes the quantization constants too (saves more).
- `bnb_4bit_compute_dtype=bfloat16` — matmuls run in bf16 for stability.
- `gradient_checkpointing=True` — trade compute for memory (recompute activations).
