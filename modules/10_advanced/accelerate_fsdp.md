# FSDP (Fully Sharded Data Parallel) with Accelerate

FSDP shards model parameters, gradients, and optimizer states across GPUs, letting
you train models that don't fit on a single device. It's built into PyTorch and
driven here through Hugging Face `accelerate`.

## 1. Save this accelerate config

`fsdp_config.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD      # = ZeRO-3 equivalent
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4          # = number of GPUs
rdzv_backend: static
same_network: true
use_cpu: false
```

## 2. Launch

```bash
accelerate launch --config_file fsdp_config.yaml \
    modules/03_full_finetuning/train_full.py \
    --model meta-llama/Llama-3.1-8B \
    --batch-size 1
```

Any `Trainer`/`SFTTrainer` script in this repo works unchanged — `accelerate`
handles the sharding. The same is true for the LoRA/QLoRA scripts.

## 3. Tips
- `FULL_SHARD` = max memory savings (shards everything). Use `SHARD_GRAD_OP` for a
  speed/memory middle ground.
- Set `fsdp_offload_params: true` to spill to CPU RAM when you're still OOM (slower).
- Combine FSDP + QLoRA for very large models on modest multi-GPU boxes.
- Use `SHARDED_STATE_DICT` for checkpoints so saving doesn't gather the full model
  onto one rank.
- Keep `per_device_train_batch_size` small and lean on gradient accumulation.
