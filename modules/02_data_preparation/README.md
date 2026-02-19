# Module 02 — Data Preparation

> "Garbage in, garbage out" is *the* law of fine-tuning. In 2026, data quality
> beats data quantity almost every time. This module is the highest-ROI lesson.

## What you'll learn
1. Clean & deduplicate raw rows (strip empties, near-duplicates, bad lengths).
2. Format rows into the model's **chat template** (the format it expects).
3. **Pack** short examples into fixed-length blocks for efficient training.
4. Split into train / validation / test correctly (no leakage).

## Files
| File | Purpose |
|------|---------|
| `clean_and_format.py` | End-to-end: load jsonl → clean → chat-format → split → save. |
| `packing.py` | Show sequence packing and why it speeds up training. |

## Run
```bash
python modules/02_data_preparation/clean_and_format.py
python modules/02_data_preparation/packing.py
```

## Data contract
Input rows (instruction style):
```json
{ "instruction": "...", "input": "" , "output": "..." }
```
Output: a `text` column rendered with the tokenizer's chat template, ready for
`SFTTrainer` (Module 06).

## Quality checklist (use before every real run)
- [ ] No empty `output` fields.
- [ ] Deduplicated (exact + near-duplicate).
- [ ] Consistent formatting / chat template.
- [ ] Reasonable length distribution (drop extreme outliers).
- [ ] Held-out validation & test sets that never appear in training.
- [ ] Spot-check 20 random samples by hand. Always.
