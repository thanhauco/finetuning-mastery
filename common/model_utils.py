"""Model + tokenizer loading helpers (handles padding tokens, dtype, quantization)."""
from __future__ import annotations

from common.config import best_dtype, get_device

# Tiny default model so examples run anywhere. Override with --model.
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    # Many causal LMs ship without a pad token; reuse EOS for batching.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_causal_lm(model_name: str, quantized: bool = False):
    """Load a causal LM. If quantized=True, load in 4-bit NF4 (needs bitsandbytes)."""
    import torch
    from transformers import AutoModelForCausalLM

    kwargs = {}
    if quantized:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=best_dtype(),
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = best_dtype()
        if get_device() == "cuda":
            kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.config.use_cache = False  # required when using gradient checkpointing
    return model


def count_trainable_parameters(model) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def print_trainable_summary(model) -> None:
    trainable, total = count_trainable_parameters(model)
    pct = 100 * trainable / max(total, 1)
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
