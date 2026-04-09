#!/usr/bin/env python3
import json
import os
import re
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from openai import OpenAI

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("VLLM_API_KEY", "dummy")
MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "hack")
OUT_JSONL = os.environ.get("OUT_JSONL", "gsm8k_monitor_entropy_test.jsonl")

N = int(os.environ.get("N", "5"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "384"))
TOP_LOGPROBS = int(os.environ.get("TOP_LOGPROBS", "5"))

SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "Think step by step and place your reasoning between:\n"
    "<reasoning>\n"
    "step-by-step reasoning\n"
    "</reasoning>\n"
    "<final>\n"
    "(final answer)\n"
    "</final>\n"
    "Do NOT output any other tags or text outside these tags.\n"
)


def extract_final_numeric_answer(text: str) -> Optional[str]:
    if not text:
        return None

    m = re.search(r"<final>\s*(.*?)\s*</final>", text, flags=re.I | re.S)
    if m:
        candidate = m.group(1).strip()
    else:
        candidate = text.strip()

    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", candidate)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def extract_gold_numeric_answer(gold: str) -> Optional[str]:
    m = re.search(r"####\s*([-]?\d[\d,]*(?:\.\d+)?)", gold)
    if m:
        return m.group(1).replace(",", "")
    return extract_final_numeric_answer(gold)


def safe_get(d: Any, path: List[Any], default=None):
    cur = d
    for p in path:
        try:
            if isinstance(cur, dict):
                cur = cur[p]
            else:
                cur = cur[p]
        except Exception:
            return default
    return cur


def extract_entropy_info(resp_dict: Dict[str, Any]) -> Dict[str, Any]:
    content = safe_get(resp_dict, ["choices", 0, "logprobs", "content"], [])
    entropies: List[float] = []
    tokens: List[str] = []

    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue

            tok = item.get("token")
            ent = item.get("entropy")

            if tok is not None:
                tokens.append(str(tok))

            if ent is not None:
                try:
                    entropies.append(float(ent))
                except Exception:
                    pass

    mean_entropy = sum(entropies) / len(entropies) if entropies else None
    max_entropy = max(entropies) if entropies else None

    return {
        "token_count_with_entropy": len(entropies),
        "mean_entropy": mean_entropy,
        "max_entropy": max_entropy,
        "entropies": entropies,
        "entropy_tokens": tokens,
    }


def extract_monitor_info(resp_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Looks for:
      resp_dict["monitor"]
    and optionally per-choice / per-step monitor payloads if present.
    """
    top_monitor = resp_dict.get("monitor")

    choice0 = safe_get(resp_dict, ["choices", 0], {})
    message_monitor = choice0.get("monitor") if isinstance(choice0, dict) else None

    latest_p_hack = None
    latest_token_score = None
    prompt_monitor_prob_so_far = None

    source = None
    if isinstance(top_monitor, dict) and top_monitor:
        source = top_monitor
    elif isinstance(message_monitor, dict) and message_monitor:
        source = message_monitor

    if isinstance(source, dict):
        # handle both request-id keyed dicts and flat dicts
        if any(isinstance(v, dict) for v in source.values()):
            first_nested = None
            for v in source.values():
                if isinstance(v, dict):
                    first_nested = v
                    break
            source = first_nested if first_nested is not None else source

        if isinstance(source, dict):
            for key in ["p_hack", "prompt_monitor_prob_so_far", "monitor_prob", "token_score"]:
                if key in source:
                    pass

            latest_p_hack = source.get("p_hack")
            if latest_p_hack is None:
                latest_p_hack = source.get("monitor_prob")
            if latest_p_hack is None:
                latest_p_hack = source.get("prompt_monitor_prob_so_far")

            latest_token_score = source.get("token_score")
            prompt_monitor_prob_so_far = source.get("prompt_monitor_prob_so_far")

    return {
        "monitor_raw": top_monitor,
        "choice_monitor_raw": message_monitor,
        "p_hack": latest_p_hack,
        "token_score": latest_token_score,
        "prompt_monitor_prob_so_far": prompt_monitor_prob_so_far,
    }


def main():
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    ds = load_dataset("gsm8k", "main", split=f"test[:{N}]")

    rows: List[Dict[str, Any]] = []

    for i, ex in enumerate(ds):
        question = ex["question"]
        gold = ex["answer"]

        print(f"\n[{i+1}/{N}] Running sample...")
        print("Q:", question)

        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
                max_completion_tokens=MAX_TOKENS,
                logprobs=True,
                top_logprobs=TOP_LOGPROBS,
                extra_body={
                    "return_entropy": True
                },
            )

            resp_dict = resp.model_dump()
            answer_text = safe_get(resp_dict, ["choices", 0, "message", "content"], "")

            pred_num = extract_final_numeric_answer(answer_text)
            gold_num = extract_gold_numeric_answer(gold)
            is_correct = (
                pred_num is not None and gold_num is not None and pred_num == gold_num
            )

            entropy_info = extract_entropy_info(resp_dict)
            monitor_info = extract_monitor_info(resp_dict)

            row = {
                "idx": i,
                "question": question,
                "gold_answer_raw": gold,
                "gold_answer_numeric": gold_num,
                "model_answer_raw": answer_text,
                "model_answer_numeric": pred_num,
                "is_correct": is_correct,
                "mean_entropy": entropy_info["mean_entropy"],
                "max_entropy": entropy_info["max_entropy"],
                "token_count_with_entropy": entropy_info["token_count_with_entropy"],
                "entropies": entropy_info["entropies"],
                "entropy_tokens": entropy_info["entropy_tokens"],
                "p_hack": monitor_info["p_hack"],
                "token_score": monitor_info["token_score"],
                "prompt_monitor_prob_so_far": monitor_info["prompt_monitor_prob_so_far"],
                "monitor_raw": monitor_info["monitor_raw"],
                "choice_monitor_raw": monitor_info["choice_monitor_raw"],
                "raw_response": resp_dict,
            }
            rows.append(row)

            print("Pred:", pred_num, "| Gold:", gold_num, "| Correct:", is_correct)
            print(
                "entropy mean/max:",
                entropy_info["mean_entropy"],
                entropy_info["max_entropy"],
            )
            print("p_hack:", monitor_info["p_hack"])
            print("prompt_monitor_prob_so_far:", monitor_info["prompt_monitor_prob_so_far"])

        except Exception as e:
            row = {
                "idx": i,
                "question": question,
                "gold_answer_raw": gold,
                "error": str(e),
            }
            rows.append(row)
            print("ERROR:", e)

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved results to {OUT_JSONL}")

    valid = [r for r in rows if "error" not in r]
    if valid:
        acc = sum(bool(r.get("is_correct")) for r in valid) / len(valid)
        mean_entropy_vals = [
            r["mean_entropy"] for r in valid if r.get("mean_entropy") is not None
        ]
        p_hack_vals = [
            float(r["p_hack"]) for r in valid
            if r.get("p_hack") is not None
        ]

        print("\nSummary")
        print("num_valid:", len(valid))
        print("accuracy:", acc)
        print(
            "avg_mean_entropy:",
            (sum(mean_entropy_vals) / len(mean_entropy_vals)) if mean_entropy_vals else None,
        )
        print(
            "avg_p_hack:",
            (sum(p_hack_vals) / len(p_hack_vals)) if p_hack_vals else None,
        )


if __name__ == "__main__":
    main()