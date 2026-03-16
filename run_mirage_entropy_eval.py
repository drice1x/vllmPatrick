#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper-aligned client-side reproduction for:
1) TTBC 1: Wait & Think More
2) TTBC 2: Exact Thinking Tokens

Key correction versus earlier versions:
- NO fixed chunk-size control for reasoning growth.
- WAIT is injected only when the model naturally closes thinking with </think>.
- Exact-thinking budget is enforced by repeatedly re-opening thinking with "Wait"
  after natural </think> events until the cumulative thinking-token count reaches t.
- Final answer generation uses the same temperature as reasoning, matching the
  paper more closely than a lowered finalization temperature.

Limitation:
This is still a client-side approximation via chat completions. The paper’s most
direct implementation would suppress </think> inside one continuous decode stream.
"""

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
from transformers import AutoTokenizer


# ============================================================
# Config
# ============================================================
BASE_URL = "http://localhost:8000/v1"
API_KEY = "token-abc123"

# Must match your vLLM served model alias
MODEL_NAME = "qwen25-7b-reasoning"

# Must match tokenizer of served model
#HF_TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"
# For paper-faithful model replication, use:
HF_TOKENIZER_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

TEMPERATURE = 0.7

# Mode: "wait" or "exact"
MODE = "exact"

# TTBC 1 budgets: number of additional WAIT insertions
WAIT_INSERTION_LIST = [0, 1, 2, 4, 8, 16, 32]

# TTBC 2 budgets: exact thinking-token budgets
THINKING_BUDGET_LIST = [64, 128, 256, 512, 1024, 2048, 4096]

# Final answer max tokens
FINAL_ANSWER_MAX_TOKENS = 64

# A large enough upper bound for one natural reasoning pass
# We do not use this as the TTC controller; it is only a safety cap.
NATURAL_REASONING_MAX_TOKENS = 8192

# Dataset
DATASET_NAME = "gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"
NUM_SAMPLES = 500  # None => full GSM8K test set (1320)

RANDOM_SEED = 42

OUTDIR = "mirage_entropy_runs"
os.makedirs(OUTDIR, exist_ok=True)

THINK_START = "<think>"
THINK_END = "</think>"
WAIT_TEXT = "Wait"

SYSTEM_PROMPT = (
    "You are a careful reasoning assistant.\n"
    f"First think step by step inside {THINK_START} ... {THINK_END}.\n"
    f"Then give the final answer after {THINK_END}.\n"
    f"Start your reasoning with {THINK_START}."
)

CONTINUE_PROMPT = (
    "Continue thinking only.\n"
    f"Stay inside the thinking block.\n"
    f"Do not provide the final answer yet.\n"
)

FINALIZE_SYSTEM_PROMPT = (
    "You are now in finalization mode.\n"
    f"The thinking phase has ended with {THINK_END}.\n"
    "Output only the final answer, concisely."
)


# ============================================================
# Clients
# ============================================================
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_NAME, trust_remote_code=True)


# ============================================================
# Utils
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)


def response_to_dict(resp: Any) -> Dict[str, Any]:
    try:
        if hasattr(resp, "model_dump"):
            return resp.model_dump()
        if hasattr(resp, "dict"):
            return resp.dict()
        return json.loads(json.dumps(resp, default=str))
    except Exception:
        return {"raw_str": str(resp)}


def load_gsm8k_subset(num_samples: Optional[int], seed: int) -> List[Dict[str, str]]:
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)

    if num_samples is not None:
        rng = random.Random(seed)
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        ds = ds.select(idxs[:num_samples])

    rows: List[Dict[str, str]] = []
    for i, ex in enumerate(ds):
        answer_text = ex["answer"]
        m = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer_text)
        gold = m.group(1).replace(",", "") if m else answer_text.strip()

        rows.append(
            {
                "id": f"gsm8k_test_{i}",
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


def summarize_entropy(entropies: List[Optional[float]]) -> Dict[str, Optional[float]]:
    vals = [float(x) for x in entropies if x is not None and math.isfinite(float(x))]
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


def tokenize_ids(text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def token_strings_for_text(text: str) -> List[str]:
    ids = tokenize_ids(text)
    return [str(t) for t in tokenizer.convert_ids_to_tokens(ids)]


def count_tokens_in_text(text: str) -> int:
    return len(tokenize_ids(text))


def extract_thinking_text(full_text: str) -> str:
    """
    Extract only the content inside <think>...</think>.
    If the closing tag is missing, return everything after <think>.
    If there is no <think>, fall back to the full text.
    """
    if THINK_START in full_text and THINK_END in full_text:
        m = re.search(
            re.escape(THINK_START) + r"(.*?)" + re.escape(THINK_END),
            full_text,
            flags=re.DOTALL,
        )
        if m:
            return m.group(1)
    if THINK_START in full_text:
        return full_text.split(THINK_START, 1)[1]
    return full_text


def natural_think_closed(full_text: str) -> bool:
    return THINK_END in full_text


def build_initial_messages(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def build_continue_messages(question: str, thinking_text: str) -> List[Dict[str, str]]:
    assistant_content = f"{THINK_START}{thinking_text}"
    return [
        {"role": "system", "content": CONTINUE_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": "Continue thinking only."},
    ]


def build_finalize_messages(question: str, thinking_text: str) -> List[Dict[str, str]]:
    assistant_content = f"{THINK_START}{thinking_text}{THINK_END}"
    return [
        {"role": "system", "content": FINALIZE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": "Now give only the final answer."},
    ]


# ============================================================
# API wrapper
# ============================================================
def run_chat_completion(
    messages: List[Dict[str, str]],
    max_tokens: int,
) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=5,
        extra_body={"return_entropy": True},
    )

    choice = resp.choices[0]
    text = choice.message.content or ""
    tokens = safe_get_token_list(choice)
    entropies = safe_get_entropy_list(choice)

    return {
        "text": text,
        "tokens": tokens,
        "entropies": entropies,
        "finish_reason": choice.finish_reason,
        "raw_response": response_to_dict(resp),
    }


# ============================================================
# Natural reasoning pass: run until the model naturally closes </think>
# ============================================================
def run_natural_reasoning_pass(question: str, thinking_prefix: str = "") -> Dict[str, Any]:
    """
    One natural reasoning pass. The model is allowed to reason naturally.
    We detect whether it closed thinking with </think>.
    """
    if thinking_prefix == "":
        messages = build_initial_messages(question)
    else:
        messages = build_continue_messages(question, thinking_prefix)

    out = run_chat_completion(messages, max_tokens=NATURAL_REASONING_MAX_TOKENS)

    full_text = out["text"]
    thinking_text = extract_thinking_text(full_text)
    closed = natural_think_closed(full_text)

    return {
        "full_text": full_text,
        "thinking_text": thinking_text,
        "tokens": out["tokens"],
        "entropies": out["entropies"],
        "finish_reason": out["finish_reason"],
        "closed_thinking": closed,
        "raw_response": out["raw_response"],
    }


# ============================================================
# TTBC 1: Wait & Think More
# ============================================================
def run_wait_protocol(question_obj: Dict[str, str], num_wait_insertions: int) -> Dict[str, Any]:
    wait_tok_ids = tokenize_ids(WAIT_TEXT)
    wait_tok_strs = token_strings_for_text(WAIT_TEXT)

    reasoning_text = ""
    reasoning_tokens: List[str] = []
    reasoning_entropies: List[Optional[float]] = []
    reasoning_sources: List[str] = []
    raw_passes: List[Dict[str, Any]] = []
    pass_meta: List[Dict[str, Any]] = []

    # First natural pass
    p = run_natural_reasoning_pass(question_obj["question"], thinking_prefix="")
    reasoning_text = p["thinking_text"]
    reasoning_tokens.extend(p["tokens"])
    reasoning_entropies.extend(p["entropies"])
    reasoning_sources.extend(["model"] * len(p["tokens"]))
    raw_passes.append(p["raw_response"])
    pass_meta.append(
        {
            "pass_idx": 0,
            "closed_thinking": p["closed_thinking"],
            "chunk_tokens": len(p["tokens"]),
            "cumulative_thinking_tokens": count_tokens_in_text(reasoning_text),
        }
    )

    # Inject Wait only after a natural </think>
    for k in range(num_wait_insertions):
        if not p["closed_thinking"]:
            break

        reasoning_text = reasoning_text + WAIT_TEXT
        reasoning_tokens.extend(wait_tok_strs)
        reasoning_entropies.extend([None] * len(wait_tok_ids))
        reasoning_sources.extend(["wait"] * len(wait_tok_ids))

        p = run_natural_reasoning_pass(
            question_obj["question"],
            thinking_prefix=reasoning_text,
        )
        reasoning_text = reasoning_text + p["thinking_text"]
        reasoning_tokens.extend(p["tokens"])
        reasoning_entropies.extend(p["entropies"])
        reasoning_sources.extend(["model"] * len(p["tokens"]))
        raw_passes.append(p["raw_response"])
        pass_meta.append(
            {
                "pass_idx": k + 1,
                "closed_thinking": p["closed_thinking"],
                "chunk_tokens": len(p["tokens"]),
                "cumulative_thinking_tokens": count_tokens_in_text(reasoning_text),
            }
        )

    return {
        "reasoning_text": reasoning_text,
        "reasoning_tokens": reasoning_tokens,
        "reasoning_token_entropy": reasoning_entropies,
        "reasoning_token_sources": reasoning_sources,
        "reasoning_raw_passes": raw_passes,
        "reasoning_pass_meta": pass_meta,
        "num_wait_insertions": num_wait_insertions,
        "natural_closed_last_pass": p["closed_thinking"],
        "thinking_token_count": count_tokens_in_text(reasoning_text),
    }


# ============================================================
# TTBC 2: Exact thinking tokens
# ============================================================
def run_exact_budget_protocol(question_obj: Dict[str, str], thinking_budget: int) -> Dict[str, Any]:
    """
    Paper-aligned client-side approximation:
    - let the model think naturally until it emits </think>
    - if thinking tokens < budget, append Wait and continue
    - repeat until cumulative thinking tokens >= budget
    - trim thinking text approximately to the target budget by token ids
    """
    wait_tok_ids = tokenize_ids(WAIT_TEXT)
    wait_tok_strs = token_strings_for_text(WAIT_TEXT)

    reasoning_text = ""
    reasoning_tokens: List[str] = []
    reasoning_entropies: List[Optional[float]] = []
    reasoning_sources: List[str] = []
    raw_passes: List[Dict[str, Any]] = []
    pass_meta: List[Dict[str, Any]] = []
    num_wait_insertions = 0

    p = run_natural_reasoning_pass(question_obj["question"], thinking_prefix="")
    reasoning_text = p["thinking_text"]
    reasoning_tokens.extend(p["tokens"])
    reasoning_entropies.extend(p["entropies"])
    reasoning_sources.extend(["model"] * len(p["tokens"]))
    raw_passes.append(p["raw_response"])
    pass_meta.append(
        {
            "pass_idx": 0,
            "closed_thinking": p["closed_thinking"],
            "chunk_tokens": len(p["tokens"]),
            "cumulative_thinking_tokens": count_tokens_in_text(reasoning_text),
        }
    )

    while count_tokens_in_text(reasoning_text) < thinking_budget and p["closed_thinking"]:
        reasoning_text = reasoning_text + WAIT_TEXT
        reasoning_tokens.extend(wait_tok_strs)
        reasoning_entropies.extend([None] * len(wait_tok_ids))
        reasoning_sources.extend(["wait"] * len(wait_tok_ids))
        num_wait_insertions += 1

        p = run_natural_reasoning_pass(
            question_obj["question"],
            thinking_prefix=reasoning_text,
        )
        reasoning_text = reasoning_text + p["thinking_text"]
        reasoning_tokens.extend(p["tokens"])
        reasoning_entropies.extend(p["entropies"])
        reasoning_sources.extend(["model"] * len(p["tokens"]))
        raw_passes.append(p["raw_response"])
        pass_meta.append(
            {
                "pass_idx": num_wait_insertions,
                "closed_thinking": p["closed_thinking"],
                "chunk_tokens": len(p["tokens"]),
                "cumulative_thinking_tokens": count_tokens_in_text(reasoning_text),
            }
        )

    # Trim thinking text to exact token budget on tokenizer level
    think_ids = tokenize_ids(reasoning_text)
    think_ids = think_ids[:thinking_budget]
    trimmed_reasoning_text = tokenizer.decode(think_ids, skip_special_tokens=False)

    # Token/entropy arrays are only approximate after trimming because the API returns token strings,
    # not exact tokenizer ids aligned to our local tokenizer. We keep them as full observed trajectory
    # and report the exact tokenizer-based budget separately.
    return {
        "reasoning_text": trimmed_reasoning_text,
        "reasoning_tokens": reasoning_tokens,
        "reasoning_token_entropy": reasoning_entropies,
        "reasoning_token_sources": reasoning_sources,
        "reasoning_raw_passes": raw_passes,
        "reasoning_pass_meta": pass_meta,
        "num_wait_insertions": num_wait_insertions,
        "natural_closed_last_pass": p["closed_thinking"],
        "thinking_token_count": len(think_ids),
    }


# ============================================================
# Finalization
# ============================================================
def run_finalize_pass(question_obj: Dict[str, str], reasoning_text: str) -> Dict[str, Any]:
    out = run_chat_completion(
        build_finalize_messages(question_obj["question"], reasoning_text),
        max_tokens=FINAL_ANSWER_MAX_TOKENS,
    )
    return {
        "final_text": out["text"],
        "final_tokens": out["tokens"],
        "final_token_entropy": out["entropies"],
        "final_finish_reason": out["finish_reason"],
        "final_raw_response": out["raw_response"],
    }


# ============================================================
# Single example
# ============================================================
def run_one(question_obj: Dict[str, str], budget_value: int) -> Dict[str, Any]:
    if MODE == "wait":
        reasoning = run_wait_protocol(question_obj, num_wait_insertions=budget_value)
        budget_fields = {"num_wait_insertions_requested": budget_value}
    elif MODE == "exact":
        reasoning = run_exact_budget_protocol(question_obj, thinking_budget=budget_value)
        budget_fields = {"thinking_budget_requested": budget_value}
    else:
        raise ValueError(f"Unknown MODE={MODE}")

    finalization = run_finalize_pass(question_obj, reasoning["reasoning_text"])

    reasoning_tokens = reasoning["reasoning_tokens"]
    reasoning_entropy = reasoning["reasoning_token_entropy"]
    final_tokens = finalization["final_tokens"]
    final_entropy = finalization["final_token_entropy"]

    full_tokens = reasoning_tokens + final_tokens
    full_entropy = reasoning_entropy + final_entropy

    total_stats = summarize_entropy(full_entropy)
    reasoning_stats = summarize_entropy(reasoning_entropy)
    final_stats = summarize_entropy(final_entropy)

    final_text = finalization["final_text"]

    return {
        "question_id": question_obj["id"],
        "question": question_obj["question"],
        "gold_answer": question_obj["answer"],
        "gold_rationale": question_obj["full_answer_rationale"],
        **budget_fields,
        "response_text": final_text,
        "reasoning_text": reasoning["reasoning_text"],
        "pred_final_number": extract_final_number(final_text),
        "correct": is_correct(final_text, question_obj["answer"]),
        "thinking_token_count": reasoning["thinking_token_count"],
        "num_reasoning_tokens_observed": len(reasoning_tokens),
        "num_final_tokens": len(final_tokens),
        "num_generated_tokens_total_observed": len(full_tokens),
        "reasoning_tokens": reasoning_tokens,
        "reasoning_token_entropy": reasoning_entropy,
        "reasoning_token_sources": reasoning["reasoning_token_sources"],
        "final_tokens": final_tokens,
        "final_token_entropy": final_entropy,
        "tokens": full_tokens,
        "token_entropy": full_entropy,
        "trajectory_length": len(full_entropy),
        "reasoning_raw_passes": reasoning["reasoning_raw_passes"],
        "reasoning_pass_meta": reasoning["reasoning_pass_meta"],
        "final_raw_response": finalization["final_raw_response"],
        "num_wait_insertions_actual": reasoning["num_wait_insertions"],
        "natural_closed_last_pass": reasoning["natural_closed_last_pass"],
        "total_entropy_mean": total_stats["entropy_mean"],
        "reasoning_entropy_mean": reasoning_stats["entropy_mean"],
        "final_entropy_mean": final_stats["entropy_mean"],
    }


# ============================================================
# Main
# ============================================================
def main() -> None:
    set_seed(RANDOM_SEED)
    questions = load_gsm8k_subset(NUM_SAMPLES, RANDOM_SEED)

    if MODE == "wait":
        budgets = WAIT_INSERTION_LIST
    else:
        budgets = THINKING_BUDGET_LIST

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTDIR, f"run_{MODE}_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "MODE": MODE,
                "MODEL_NAME": MODEL_NAME,
                "HF_TOKENIZER_NAME": HF_TOKENIZER_NAME,
                "TEMPERATURE": TEMPERATURE,
                "WAIT_INSERTION_LIST": WAIT_INSERTION_LIST,
                "THINKING_BUDGET_LIST": THINKING_BUDGET_LIST,
                "FINAL_ANSWER_MAX_TOKENS": FINAL_ANSWER_MAX_TOKENS,
                "NATURAL_REASONING_MAX_TOKENS": NATURAL_REASONING_MAX_TOKENS,
                "NUM_SAMPLES": NUM_SAMPLES,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(os.path.join(run_dir, "selected_questions.json"), "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    all_results: List[Dict[str, Any]] = []

    for budget in budgets:
        print(f"\n=== MODE={MODE} budget={budget} ===")
        bucket: List[Dict[str, Any]] = []

        for i, q in enumerate(questions, start=1):
            print(f"[{i:04d}/{len(questions)}] {q['id']}")
            try:
                res = run_one(q, budget)
            except Exception as e:
                print(f"ERROR for {q['id']} @ budget={budget}: {e}")
                res = {
                    "question_id": q["id"],
                    "question": q["question"],
                    "gold_answer": q["answer"],
                    "error": repr(e),
                }
                if MODE == "wait":
                    res["num_wait_insertions_requested"] = budget
                else:
                    res["thinking_budget_requested"] = budget

            bucket.append(res)
            all_results.append(res)

        suffix = "waits" if MODE == "wait" else "budget"
        with open(
            os.path.join(run_dir, f"results_{suffix}_{budget}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(bucket, f, ensure_ascii=False, indent=2)

    with open(os.path.join(run_dir, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    summary: Dict[str, Any] = {}
    for budget in budgets:
        if MODE == "wait":
            rows = [
                r for r in all_results
                if r.get("num_wait_insertions_requested") == budget and "error" not in r
            ]
        else:
            rows = [
                r for r in all_results
                if r.get("thinking_budget_requested") == budget and "error" not in r
            ]

        if not rows:
            summary[str(budget)] = {
                "n": 0,
                "accuracy": None,
                "mean_reasoning_entropy": None,
                "mean_thinking_token_count": None,
            }
            continue

        acc = sum(bool(r["correct"]) for r in rows) / len(rows)
        reason_ent = [r["reasoning_entropy_mean"] for r in rows if r.get("reasoning_entropy_mean") is not None]

        summary[str(budget)] = {
            "n": len(rows),
            "accuracy": acc,
            "mean_reasoning_entropy": mean(reason_ent) if reason_ent else None,
            "mean_thinking_token_count": mean(r["thinking_token_count"] for r in rows),
            "mean_total_tokens_observed": mean(r["num_generated_tokens_total_observed"] for r in rows),
            "mean_wait_insertions_actual": mean(r["num_wait_insertions_actual"] for r in rows),
        }

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSaved results to:", run_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
