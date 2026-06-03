# Serving with vLLM (high-throughput, OpenAI-compatible)

[vLLM](https://docs.vllm.ai) is the de-facto standard for fast LLM serving in 2026.
It gives you continuous batching, PagedAttention KV-cache, and an OpenAI-compatible
HTTP API. Requires a **Linux/WSL host with an NVIDIA GPU**.

## 1. Install

```bash
pip install vllm
```

## 2. Serve a merged model

First merge your adapter into a standalone checkpoint (Module 04):

```bash
python modules/04_lora_peft/merge_adapter.py \
    --base HuggingFaceTB/SmolLM2-360M \
    --adapter outputs/06_sft \
    --out outputs/09_merged
```

Then start the server:

```bash
vllm serve outputs/09_merged \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --dtype auto
```

The server exposes OpenAI-compatible routes at `http://localhost:8000/v1`.

## 3. Call it with the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="outputs/09_merged",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain LoRA in one sentence."},
    ],
    temperature=0.7,
    max_tokens=128,
)
print(resp.choices[0].message.content)
```

Or with `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "outputs/09_merged",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64
  }'
```

## 4. Serve LoRA adapters WITHOUT merging (multi-tenant pattern)

vLLM can hot-load adapters on top of one base model — ideal for serving many tasks:

```bash
vllm serve HuggingFaceTB/SmolLM2-360M \
    --enable-lora \
    --lora-modules sft=outputs/06_sft dpo=outputs/07_dpo \
    --port 8000
```

Then select an adapter per request via the `model` field:

```python
resp = client.chat.completions.create(
    model="dpo",   # or "sft", or the base model name
    messages=[{"role": "user", "content": "Is plaintext password storage safe?"}],
)
```

## 5. Production tips
- Set `--max-num-seqs` and `--gpu-memory-utilization` to tune throughput vs. headroom.
- Put it behind a reverse proxy (nginx/Traefik) with auth + rate limiting.
- Quantize for inference (AWQ/GPTQ) to fit bigger models or cut cost.
- Monitor tokens/sec, queue depth, and p99 latency under real concurrency.
- Re-run your eval suite (Module 08) against the served endpoint, not just the local model.
