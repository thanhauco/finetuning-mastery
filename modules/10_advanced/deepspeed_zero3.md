# DeepSpeed ZeRO-3 with Accelerate

DeepSpeed ZeRO is a mature alternative to FSDP for training very large models.
ZeRO-3 shards optimizer states, gradients, AND parameters across GPUs, and can
optionally offload them to CPU or NVMe.

## ZeRO stages at a glance
```
   ZeRO-1: shard optimizer states                    (smallest savings)
   ZeRO-2: + shard gradients
   ZeRO-3: + shard parameters                        (largest savings ≈ FSDP FULL_SHARD)
   + offload: spill shards to CPU/NVMe               (train beyond GPU RAM, slower)
```

## 1. DeepSpeed config

`ds_zero3.json`:

```json
{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param": { "device": "cpu", "pin_memory": true },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

## 2. Accelerate config

`ds_accelerate.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: ds_zero3.json
  zero3_init_flag: true
machine_rank: 0
num_machines: 1
num_processes: 4
mixed_precision: bf16
use_cpu: false
```

## 3. Launch

```bash
pip install deepspeed
accelerate launch --config_file ds_accelerate.yaml \
    modules/06_instruction_tuning/train_sft.py \
    --model meta-llama/Llama-3.1-8B \
    --no-lora \
    --batch-size 1
```

## FSDP vs DeepSpeed — which?
- **FSDP**: native PyTorch, fewer moving parts, great default in 2026.
- **DeepSpeed**: richer offload (CPU/NVMe), battle-tested for 100B+ models, more knobs.
- Both reach similar memory profiles at ZeRO-3 / FULL_SHARD. Pick the one your
  cluster/team already supports.

## Tips
- CPU/NVMe offload lets you train models larger than total GPU RAM — at a speed cost.
- Use `"auto"` values so Accelerate fills them from your `TrainingArguments`.
- Keep `stage3_gather_16bit_weights_on_model_save: true` to save a normal checkpoint.
