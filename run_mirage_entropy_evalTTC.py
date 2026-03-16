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
FINAL_TEMPERATURE = 0.2

# Minimum thinking budgets
MAX_TOKENS_LIST = [64, 128, 256, 512, 1024, 2048, 3000, 4096]

# Reserve for forced final answer
FINAL_ANSWER_RESERVE = 64

# Reasoning chunk size for the iterative Wait protocol
REASONING_CHUNK_SIZE = 128

# Token/string used to extend thinking
WAIT_TOKEN = " Wait"

NUM_SAMPLES = 100
RANDOM_SEED = 42

OUTDIR = "mirage_entropy_runs"
os.makedirs(OUTDIR, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. "
    "Think step by step before giving the final answer."
)

CONTINUE_THINKING_SYSTEM_PROMPT = (
    "You are still in the thinking phase.\n"
    "Continue thinking step by step.\n"
    "Do NOT provide the final answer yet.\n"
    "Keep reasoning only."
)

FINALIZE_SYSTEM_PROMPT = (
    "You are a careful reasoning assistant.\n"
    "You are now in finalization mode.\n"
    "Output ONLY the final answer, concisely.\n"
    "Do not continue the reasoning.\n"
    "The last line should contain the final numeric answer clearly."
)

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def response_to_dict(resp: Any) -> Dict[str, Any]:
    try:
        if hasattr(resp, "model_dump"):
            return resp.model_dump()
        if hasattr(resp, "dict"):
            return resp.dict()
        return json.loads(json.dumps(resp, default=str))
    except Exception:
        return {"raw_str": str(resp)}


def load_gsm8k_subset(num_samples: int = 50, seed: int = 42) -> List[Dict[str, str]]:
    ds = load_dataset("gsm8k", "main", split="test")
    n = len(ds)
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    indices = indices[:num_samples]

    rows: List[Dict[str, str]] = []
    for idx in indices:
        ex = ds[idx]
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
        val = getattr(item, "entropy", None)
        if val is None:
            out.append(None)
        else:
            fv = float(val)
            out.append(fv if math.isfinite(fv) else None)
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
        toks.append("" if tok is None else str(tok))
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


def run_reasoning_chunk(messages: List[Dict[str, str]], chunk_size: int) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=chunk_size,
        logprobs=True,
        top_logprobs=5,
        extra_body={"return_entropy": True},
    )

    choice = resp.choices[0]
    text = choice.message.content or ""
    entropies = safe_get_entropy_list(choice)
    tokens = safe_get_token_list(choice)

    return {
        "text": text,
        "tokens": tokens,
        "entropies": entropies,
        "finish_reason": choice.finish_reason,
        "raw_response": response_to_dict(resp),
    }


def run_reasoning_wait_protocol(question_obj: Dict[str, str], t_min: int) -> Dict[str, Any]:
    reasoning_text = ""
    reasoning_tokens: List[str] = []
    reasoning_entropies: List[Optional[float]] = []
    raw_passes: List[Dict[str, Any]] = []
    num_wait_injections = 0

    # initial reasoning pass
    while len(reasoning_tokens) < t_min:
        remaining = t_min - len(reasoning_tokens)
        chunk_size = min(REASONING_CHUNK_SIZE, remaining)

        if reasoning_text == "":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question_obj["question"]},
            ]
        else:
            # Paper-style approximation:
            # append "Wait" to the reasoning trace and continue thinking
            patched_reasoning = reasoning_text + WAIT_TOKEN
            num_wait_injections += 1

            messages = [
                {"role": "system", "content": CONTINUE_THINKING_SYSTEM_PROMPT},
                {"role": "user", "content": question_obj["question"]},
                {"role": "assistant", "content": patched_reasoning},
                {
                    "role": "user",
                    "content": "Continue thinking. Do not provide the final answer yet.",
                },
            ]

        chunk = run_reasoning_chunk(messages, chunk_size=chunk_size)

        chunk_text = chunk["text"]
        chunk_tokens = chunk["tokens"]
        chunk_entropies = chunk["entropies"]

        print(
            f"DEBUG wait-reasoning {question_obj['id']} tmin={t_min}: "
            f"chunk_tokens={len(chunk_tokens)} total_before={len(reasoning_tokens)} "
            f"total_after={len(reasoning_tokens) + len(chunk_tokens)} "
            f"first_entropy={chunk_entropies[0] if chunk_entropies else None}"
        )

        if len(chunk_tokens) != len(chunk_entropies):
            print(
                f"WARNING wait reasoning mismatch for {question_obj['id']}: "
                f"tokens={len(chunk_tokens)} entropies={len(chunk_entropies)}"
            )

        reasoning_text += chunk_text
        reasoning_tokens.extend(chunk_tokens)
        reasoning_entropies.extend(chunk_entropies)
        raw_passes.append(chunk["raw_response"])

        # safety: if the model returns zero tokens repeatedly, force break
        if len(chunk_tokens) == 0:
            print(
                f"WARNING zero-token reasoning chunk for {question_obj['id']} at tmin={t_min}"
            )
            break

    # trim to exact minimum thinking length if a chunk overshoots
    reasoning_tokens = reasoning_tokens[:t_min]
    reasoning_entropies = reasoning_entropies[:t_min]

    return {
        "reasoning_text": reasoning_text,
        "reasoning_tokens": reasoning_tokens,
        "reasoning_token_entropy": reasoning_entropies,
        "reasoning_raw_passes": raw_passes,
        "num_wait_injections": num_wait_injections,
    }


def run_finalize_pass(question_obj: Dict[str, str], reasoning_text: str) -> Dict[str, Any]:
    user_content = (
        f"Question:\n{question_obj['question']}\n\n"
        f"Previous reasoning trace:\n{reasoning_text}\n\n"
        "Now provide ONLY the final answer."
    )

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": FINALIZE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=FINAL_TEMPERATURE,
        max_tokens=FINAL_ANSWER_RESERVE,
        logprobs=True,
        top_logprobs=5,
        extra_body={"return_entropy": True},
    )

    choice = resp.choices[0]
    final_text = choice.message.content or ""
    entropies = safe_get_entropy_list(choice)
    tokens = safe_get_token_list(choice)

    print(
        f"DEBUG finalize {question_obj['id']}: "
        f"tokens={len(tokens)} entropies={len(entropies)} "
        f"first_entropy={entropies[0] if entropies else None}"
    )

    if len(tokens) != len(entropies):
        print(
            f"WARNING finalize mismatch for {question_obj['id']}: "
            f"tokens={len(tokens)} entropies={len(entropies)}"
        )

    return {
        "final_text": final_text,
        "final_tokens": tokens,
        "final_token_entropy": entropies,
        "final_finish_reason": choice.finish_reason,
        "final_raw_response": response_to_dict(resp),
    }


def run_one(question_obj: Dict[str, str], t_min: int) -> Dict[str, Any]:
    reasoning = run_reasoning_wait_protocol(question_obj, t_min=t_min)
    finalization = run_finalize_pass(question_obj, reasoning_text=reasoning["reasoning_text"])

    reasoning_tokens = reasoning["reasoning_tokens"]
    reasoning_entropy = reasoning["reasoning_token_entropy"]

    final_tokens = finalization["final_tokens"]
    final_entropy = finalization["final_token_entropy"]

    full_tokens = reasoning_tokens + final_tokens
    full_entropy = reasoning_entropy + final_entropy

    if len(full_tokens) != len(full_entropy):
        print(
            f"WARNING full mismatch for {question_obj['id']}: "
            f"tokens={len(full_tokens)} entropies={len(full_entropy)}"
        )

    final_text = finalization["final_text"]
    ent_stats = summarize_entropy(full_entropy)

    return {
        "question_id": question_obj["id"],
        "question": question_obj["question"],
        "gold_answer": question_obj["answer"],
        "gold_rationale": question_obj["full_answer_rationale"],
        "ttc_budget_min_thinking_tokens": t_min,
        "wait_token": WAIT_TOKEN,
        "num_wait_injections": reasoning["num_wait_injections"],
        "final_answer_reserve": FINAL_ANSWER_RESERVE,
        "temperature_reasoning": TEMPERATURE,
        "temperature_final": FINAL_TEMPERATURE,
        "response_text": final_text,
        "reasoning_text": reasoning["reasoning_text"],
        "pred_final_number": extract_final_number(final_text),
        "correct": is_correct(final_text, question_obj["answer"]),
        "num_reasoning_tokens": len(reasoning_tokens),
        "num_final_tokens": len(final_tokens),
        "num_generated_tokens_total": len(full_tokens),
        "reasoning_tokens": reasoning_tokens,
        "reasoning_token_entropy": reasoning_entropy,
        "final_tokens": final_tokens,
        "final_token_entropy": final_entropy,
        "tokens": full_tokens,
        "token_entropy": full_entropy,
        "trajectory_length": len(full_entropy),
        "reasoning_raw_passes": reasoning["reasoning_raw_passes"],
        "final_raw_response": finalization["final_raw_response"],
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

    for t_min in MAX_TOKENS_LIST:
        bucket: List[Dict[str, Any]] = []
        print(f"\n=== Running WAIT protocol minimum thinking tokens t_min={t_min} ===")

        for idx, q in enumerate(questions, start=1):
            print(f"[{idx:02d}/{len(questions)}] {q['id']}")
            try:
                res = run_one(q, t_min=t_min)
            except Exception as e:
                print(f"ERROR for {q['id']} @ t_min={t_min}: {e}")
                res = {
                    "question_id": q["id"],
                    "question": q["question"],
                    "gold_answer": q["answer"],
                    "ttc_budget_min_thinking_tokens": t_min,
                    "temperature_reasoning": TEMPERATURE,
                    "temperature_final": FINAL_TEMPERATURE,
                    "error": repr(e),
                }
            bucket.append(res)
            all_results.append(res)

        outpath = os.path.join(run_dir, f"results_tmin_{t_min}.json")
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(bucket, f, ensure_ascii=False, indent=2)

    with open(os.path.join(run_dir, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    summary = {}
    for t_min in MAX_TOKENS_LIST:
        rows = [
            r for r in all_results
            if r.get("ttc_budget_min_thinking_tokens") == t_min and "error" not in r
        ]
        if not rows:
            summary[t_min] = {"n": 0, "accuracy": None, "mean_entropy": None}
            continue

        acc = sum(bool(r["correct"]) for r in rows) / len(rows)
        ent_vals = [r["entropy_mean"] for r in rows if r.get("entropy_mean") is not None]

        summary[t_min] = {
            "n": len(rows),
            "accuracy": acc,
            "mean_entropy": (mean(ent_vals) if ent_vals else None),
            "mean_reasoning_tokens": mean(r["num_reasoning_tokens"] for r in rows),
            "mean_total_tokens": mean(r["num_generated_tokens_total"] for r in rows),
            "mean_wait_injections": mean(r["num_wait_injections"] for r in rows),
        }

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSaved results to:", run_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()