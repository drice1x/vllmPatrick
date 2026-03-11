#!/usr/bin/env python3
import json
import math
import os
import random
import re
import time
from statistics import mean
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from openai import OpenAI

# -----------------------------
# Config
# -----------------------------
BASE_URL = "http://localhost:8000/v1"
API_KEY = "token-abc123"
MODEL_NAME = "qwen25-7b-reasoning"

TEMPERATURE = 0.7
MAX_TOKENS_LIST = [32, 64, 128, 256, 512, 1024, 2048]
NUM_SAMPLES = 50
RANDOM_SEED = 42

OUTDIR = "mirage_entropy_runs"
os.makedirs(OUTDIR, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. "
    "Reason step by step and then give the final answer clearly."
)

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def load_gsm8k_subset(num_samples: int = 50, seed: int = 42) -> List[Dict[str, str]]:
    ds = load_dataset("gsm8k", "main", split="test")
    n = len(ds)
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    indices = indices[:num_samples]

    rows: List[Dict[str, str]] = []
    for i, idx in enumerate(indices):
        ex = ds[idx]
        # GSM8K answers usually end with "#### 42"
        answer_text = ex["answer"]
        m = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer_text)
        gold = m.group(1).replace(",", "") if m else answer_text.strip()

        rows.append(
            {
                "id": f"gsm8k_test_{idx}",
                "question": ex["question"],
                "answer": gold,
                "full_answer_rationale": answer_text,
            }
        )
    return rows


def extract_final_number(text: str) -> Optional[str]:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def is_correct(pred_text: str, gold_answer: str) -> bool:
    pred = extract_final_number(pred_text)
    if pred is None:
        return False
    try:
        return math.isclose(float(pred), float(gold_answer), rel_tol=1e-9, abs_tol=1e-9)
    except ValueError:
        return pred.strip() == gold_answer.strip()


def safe_get_entropy_list(choice: Any) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    logprobs = getattr(choice, "logprobs", None)
    if logprobs is None:
        return out

    content = getattr(logprobs, "content", None)
    if content is None:
        return out

    for item in content:
        out.append(getattr(item, "entropy", None))
    return out


def safe_get_token_list(choice: Any) -> List[str]:
    logprobs = getattr(choice, "logprobs", None)
    if logprobs is None:
        return []

    content = getattr(logprobs, "content", None)
    if content is None:
        return []

    toks: List[str] = []
    for item in content:
        tok = getattr(item, "token", "")
        toks.append(tok if tok is not None else "")
    return toks


def summarize_entropy(entropies: List[Optional[float]]) -> Dict[str, Optional[float]]:
    vals = [float(x) for x in entropies if x is not None]
    if not vals:
        return {
            "entropy_mean": None,
            "entropy_max": None,
            "entropy_min": None,
            "entropy_last10_mean": None,
            "entropy_tail_mean": None,
        }

    last10 = vals[-10:] if len(vals) >= 10 else vals
    tail_start = int(0.75 * len(vals))
    tail_vals = vals[tail_start:] if tail_start < len(vals) else vals[-1:]

    return {
        "entropy_mean": mean(vals),
        "entropy_max": max(vals),
        "entropy_min": min(vals),
        "entropy_last10_mean": mean(last10),
        "entropy_tail_mean": mean(tail_vals),
    }

def response_to_dict(resp: Any) -> Dict[str, Any]:
    """Convert OpenAI response object to plain dict for JSON storage."""
    try:
        if hasattr(resp, "model_dump"):
            return resp.model_dump()
        if hasattr(resp, "dict"):
            return resp.dict()
        return json.loads(json.dumps(resp, default=str))
    except Exception:
        return {"raw_str": str(resp)}



def run_one(question_obj: Dict[str, str], max_tokens: int) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_obj["question"]},
        ],
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=5,
        extra_body={"return_entropy": True},
    )

    choice = resp.choices[0]
    text = choice.message.content or ""
    finish_reason = choice.finish_reason

    entropies = safe_get_entropy_list(choice)
    tokens = safe_get_token_list(choice)
    print(
    f"DEBUG run_one {question_obj['id']} mt={max_tokens}: "
    f"tokens={len(tokens)} entropies={len(entropies)} "
    f"first_entropy={entropies[0] if entropies else None}"
        )
    if len(entropies) != len(tokens):
        print(
            f"WARNING length mismatch for {question_obj['id']}: "
            f"tokens={len(tokens)} entropies={len(entropies)}"
        )
    
    ent_stats = summarize_entropy(entropies)

    return {
        "question_id": question_obj["id"],
        "question": question_obj["question"],
        "gold_answer": question_obj["answer"],
        "gold_rationale": question_obj["full_answer_rationale"],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "response_text": text,
        "finish_reason": finish_reason,
        "pred_final_number": extract_final_number(text),
        "correct": is_correct(text, question_obj["answer"]),
        "num_generated_tokens": len(tokens),
        "tokens": tokens,
        "token_entropy": entropies,              # main entropy trajectory
        "trajectory_length": len(entropies),
        "raw_response": response_to_dict(resp),  # full server JSON for debugging
        **ent_stats,
    }


def main() -> None:
    questions = load_gsm8k_subset(num_samples=NUM_SAMPLES, seed=RANDOM_SEED)

    all_results: List[Dict[str, Any]] = []
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTDIR, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "selected_questions.json"), "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    for max_tokens in MAX_TOKENS_LIST:
        bucket: List[Dict[str, Any]] = []
        print(f"\n=== Running max_tokens={max_tokens} ===")

        for idx, q in enumerate(questions, start=1):
            print(f"[{idx:02d}/{len(questions)}] {q['id']}")
            try:
                res = run_one(q, max_tokens=max_tokens)
            #except Exception as e:
            except Exception as e:
                print(f"ERROR for {q['id']} @ max_tokens={max_tokens}: {e}")
                res = {
                    "question_id": q["id"],
                    "question": q["question"],
                    "gold_answer": q["answer"],
                    "max_tokens": max_tokens,
                    "temperature": TEMPERATURE,
                    "error": repr(e),
                }
            bucket.append(res)
            all_results.append(res)

        outpath = os.path.join(run_dir, f"results_max_tokens_{max_tokens}.json")
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(bucket, f, ensure_ascii=False, indent=2)

    with open(os.path.join(run_dir, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    summary = {}
    for mt in MAX_TOKENS_LIST:
        rows = [r for r in all_results if r.get("max_tokens") == mt and "error" not in r]
        if not rows:
            summary[mt] = {"n": 0, "accuracy": None, "mean_entropy": None}
            continue

        acc = sum(bool(r["correct"]) for r in rows) / len(rows)
        ent_vals = [r["entropy_mean"] for r in rows if r.get("entropy_mean") is not None]
        summary[mt] = {
            "n": len(rows),
            "accuracy": acc,
            "mean_entropy": (mean(ent_vals) if ent_vals else None),
        }

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSaved results to:", run_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()