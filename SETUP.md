# 🛠️ Environment Setup

## 1. Python

Use Python **3.10 – 3.12**. Check:

```bash
python --version
```

## 2. Virtual environment

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (cmd)
.venv\Scripts\activate.bat
# macOS / Linux
source .venv/bin/activate
```

## 3. Install PyTorch (pick the right build)

Install PyTorch **first**, matched to your hardware, then the rest.

```bash
# CUDA 12.1 GPU (NVIDIA)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then:

```bash
pip install -r requirements.txt
```

> `bitsandbytes` (needed for QLoRA, Module 05) requires an NVIDIA GPU on Linux/WSL,
> and recent builds support native Windows. On macOS / pure CPU, skip the QLoRA module.

## 4. Hugging Face login (for gated/large models)

```bash
huggingface-cli login
```

Many examples default to **tiny, ungated** models (e.g. `HuggingFaceTB/SmolLM2-135M`)
so they run without a token.

## 5. GPU check

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 6. Windows / WSL notes

- For multi-GPU, DeepSpeed, and best `bitsandbytes` support, use **WSL2 + Ubuntu**.
- Long-path issues: enable `git config --system core.longpaths true`.

## 7. Google Colab

Each module runs on a Colab T4. Start a notebook with:

```python
!pip install -q transformers peft trl datasets bitsandbytes accelerate evaluate
```

## 8. Reproducibility

All training scripts call `common.config.set_seed(42)`. Override with `--seed`.
