# Module 09 — Deployment & Serving

A trained adapter sitting on disk helps no one. This module takes you from a tuned
checkpoint to a running, OpenAI-compatible API.

## The path to production
```
   adapter ──► merge into base ──► (optional) quantize ──► serve ──► HTTP client
   (M04/05)     (one checkpoint)     (GGUF / AWQ / GPTQ)    (vLLM)    (OpenAI SDK)
```

## What you'll learn
1. **Merge** the adapter so inference needs no PEFT (recap of Module 04, with checks).
2. Run **batched local inference** with `transformers` for quick tests.
3. Serve with **vLLM** for high-throughput, OpenAI-compatible endpoints.
4. Know your **quantization-for-inference** options (different from training quant).

## Files
| File | Purpose |
|------|---------|
| `local_inference.py` | Batched generation with `transformers` (works anywhere). |
| `serve_vllm.md` | Step-by-step vLLM serving + client snippet (Linux/WSL + GPU). |

## Run
```bash
# Quick local generation from a merged model or adapter dir
python modules/09_deployment_serving/local_inference.py --model outputs/04_merged \
    --prompts "Explain LoRA in one sentence." "Write a haiku about GPUs."
```

## Serving options (2026 quick guide)
| Tool | Best for | Notes |
|------|----------|-------|
| **vLLM** | High-throughput GPU serving | PagedAttention, continuous batching, OpenAI API. Can serve LoRA adapters live. |
| **TGI** (Text Generation Inference) | Production HF stack | Solid, well-supported. |
| **Ollama / llama.cpp** | Local / CPU / laptops | Use **GGUF** quantized files. Great for desktop apps. |
| **transformers** | Prototyping & tests | Simplest, slowest. |

## Inference quantization ≠ training quantization
- **Training** (QLoRA): NF4, optimized for *gradient* stability.
- **Inference**: AWQ / GPTQ (GPU) or GGUF Q4_K_M (CPU/laptop) — optimized for *speed* and size.
Re-quantize the merged model for your serving target; don't reuse the training quant.

## Pre-ship checklist
- [ ] Merged model loads standalone (no PEFT import needed).
- [ ] Same chat template at train & serve time.
- [ ] Stop tokens / `eos` configured so generation halts cleanly.
- [ ] Throughput + latency measured under expected concurrency.
- [ ] Eval suite (Module 08) re-run on the final merged/quantized artifact.
