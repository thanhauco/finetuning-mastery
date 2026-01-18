# Module 01 — Foundations

Before you fine-tune anything, you need to understand the moving parts:
**tokenizer → model → loss → optimizer → updated weights**.

## What you'll learn
1. How text becomes tokens (and back).
2. How a causal LM produces a loss from `(input_ids, labels)`.
3. A minimal, framework-free training loop so the magic isn't a black box.
4. What "fine-tuning" actually changes vs. pretraining.

## Files
| File | Purpose |
|------|---------|
| `01_hello_finetuning.py` | The smallest possible fine-tune: one tiny model, a handful of sentences, a raw PyTorch loop. CPU-friendly. |
| `02_tokenization.py` | Explore tokenization, special tokens, and chat templates. |
| `03_what_changes.py` | Inspect parameters and show how a gradient step moves weights. |

## Run
```bash
python modules/01_foundations/02_tokenization.py
python modules/01_foundations/01_hello_finetuning.py
python modules/01_foundations/03_what_changes.py
```

## Key concepts
- **Causal language modeling**: predict the next token; the label at position *t* is the input at *t+1* (the `Trainer`/model handles the shift for you).
- **Loss**: cross-entropy over the vocabulary. Lower = the model assigns higher probability to the correct next token.
- **Fine-tuning**: continue training a *pretrained* model on *your* data, usually with a smaller learning rate and far fewer steps than pretraining.
