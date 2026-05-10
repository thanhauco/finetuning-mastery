# Module 08 — Evaluation

If you can't measure it, you can't improve it. A fine-tune that "feels better" is
worthless without numbers. This module covers the evaluation ladder, from cheap
automatic metrics to LLM-as-judge.

## The evaluation ladder
```
   cheap / fast                                        expensive / trustworthy
   ─────────────────────────────────────────────────────────────────────────►
   perplexity  →  task metrics  →  held-out win-rate  →  LLM-as-judge  →  humans
   (Module)       (exact match,     (A/B vs baseline)    (GPT-4-class       (gold
                   accuracy)                               grader)           standard)
```

## What you'll learn
1. **Perplexity** on a held-out set — a quick proxy for language modeling quality.
2. **Task accuracy / exact match** for structured tasks.
3. **LLM-as-judge** — score open-ended answers with a stronger model (pattern + offline rubric).
4. Build a small **regression suite** so a new fine-tune never silently gets worse.

## Files
| File | Purpose |
|------|---------|
| `perplexity.py` | Compute perplexity on the held-out split. |
| `task_metrics.py` | Exact-match / accuracy on generated answers. |
| `llm_as_judge.py` | Rubric-based scoring (offline heuristic + pluggable real judge). |

## Run
```bash
python modules/08_evaluation/perplexity.py --model outputs/06_sft
python modules/08_evaluation/task_metrics.py --model outputs/06_sft
python modules/08_evaluation/llm_as_judge.py --model outputs/06_sft
```

## Don't fool yourself (common pitfalls)
- **Train/test leakage** — the #1 way to get fake-good numbers. Keep splits disjoint.
- **Tiny eval sets** — a handful of examples = high variance. Use enough samples.
- **Single metric tunnel-vision** — track quality *and* regressions (safety, format, refusals).
- **Judge bias** — LLM judges favor longer / their-own-style answers. Randomize order, use rubrics.

## Going further
For standardized academic benchmarks (MMLU, GSM8K, HellaSwag, etc.) use
[`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness):
```bash
pip install lm-eval
lm_eval --model hf --model_args pretrained=outputs/06_sft --tasks hellaswag,gsm8k --limit 100
```
