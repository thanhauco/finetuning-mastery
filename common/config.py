"""Reproducibility, device, and dtype helpers shared by every module."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch not installed yet
        pass


def get_device() -> str:
    """Return the best available device string: 'cuda', 'mps', or 'cpu'."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def best_dtype():
    """Pick bf16 on capable GPUs, else fp16 on GPU, else fp32."""
    import torch

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


@dataclass
class TrainDefaults:
    """Sensible small-scale defaults so every script trains in minutes."""

    output_dir: str = "outputs"
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    logging_steps: int = 10
    save_steps: int = 200
    warmup_ratio: float = 0.03
    seed: int = 42
