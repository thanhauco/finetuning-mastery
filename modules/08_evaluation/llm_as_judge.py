"""LLM-as-judge scoring.

Open-ended answers can't be graded by exact match. The modern approach is to ask
a stronger 'judge' model to score each answer against a rubric (1-5). This script:

  • generates answers from your fine-tuned model, then
  • scores them with a judge.

By default it uses an OFFLINE heuristic judge so the example runs with no API keys.
Plug in a real judge (e.g. an OpenAI-compatible endpoint) by implementing
`real_llm_judge` and passing --judge api.

Run:
    python modules/08_evaluation/llm_as_judge.py --model outputs/06_sft
    python modules/08_evaluation/llm_as_judge.py --model outputs/06_sft --judge api
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import get_device  # noqa: E402
from common.data_utils import load_sample_dataset, to_chat_messages  # noqa: E402
from common.model_utils import DEFAULT_MODEL, load_tokenizer  # noqa: E402

JUDGE_RUBRIC = """You are a strict grader. Score the assistant's answer from 1 to 5.
5 = fully correct, relevant, and clear.
3 = partially correct or incomplete.
1 = wrong or irrelevant.
Return ONLY the integer score.

Question: {question}
Reference answer: {reference}
Assistant answer: {answer}
Score:"""


def heuristic_judge(question: str, reference: str, answer: str) -> int:
    """Offline stand-in for a real LLM judge: rough overlap-based score (1-5).

    This is ONLY for demonstrating the pipeline without an API. Replace with a
    real judge for meaningful scores.
    """
    ref_tokens = set(reference.lower().split())
    ans_tokens = set(answer.lower().split())
    if not answer.strip():
        return 1
    if not ref_tokens:
        return 3
    overlap = len(ref_tokens & ans_tokens) / len(ref_tokens)
    if overlap >= 0.6:
        return 5
    if overlap >= 0.3:
        return 4
    if overlap >= 0.1:
        return 3
    return 2


def real_llm_judge(question: str, reference: str, answer: str) -> int:
    """Score with a real judge via an OpenAI-compatible API.

    Requires `pip install openai` and OPENAI_API_KEY (or a compatible base_url).
    """
    import os

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),  # set for local/compatible servers
    )
    prompt = JUDGE_RUBRIC.format(question=question, reference=reference, answer=answer)
    resp = client.chat.completions.create(
        model=os.environ.get("JUDGE_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=4,
    )
    text = resp.choices[0].message.content.strip()
    digits = "".join(c for c in text if c.isdigit())
    return max(1, min(5, int(digits[0]))) if digits else 3


def load_eval_model(model_dir: str):
    import torch
    from transformers import AutoModelForCausalLM

    adapter_cfg = Path(model_dir) / "adapter_config.json"
    if adapter_cfg.exists():
        import json

        from peft import PeftModel

        base = json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
        return PeftModel.from_pretrained(model, model_dir)
    return AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)


def generate(model, tokenizer, user_msg, device, max_new_tokens=96):
    import torch

    messages = [{"role": "user", "content": user_msg}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--judge", choices=["heuristic", "api"], default="heuristic")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    judge_fn = real_llm_judge if args.judge == "api" else heuristic_judge
    if args.judge == "heuristic":
        print("NOTE: using the OFFLINE heuristic judge (demo only). Use --judge api for real scores.\n")

    device = get_device()
    tokenizer = load_tokenizer(args.model)
    model = load_eval_model(args.model)
    if device != "cuda":
        model.to(device)
    model.eval()

    splits = load_sample_dataset(seed=args.seed)
    rows = list(splits["test"])[: args.max_samples]

    scores = []
    for row in rows:
        question = to_chat_messages(row)[0]["content"]
        reference = row["output"]
        answer = generate(model, tokenizer, question, device)
        score = judge_fn(question, reference, answer)
        scores.append(score)
        print(f"Q: {question[:60]}")
        print(f"  answer: {answer[:80]}")
        print(f"  judge score: {score}/5\n")

    print("=" * 50)
    print(f"Average judge score: {sum(scores)/len(scores):.2f} / 5  (n={len(scores)})")


if __name__ == "__main__":
    main()
