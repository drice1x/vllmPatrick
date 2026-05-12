#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path("/home/patrick/vllmPatrickMonitoring")
OUTDIR = ROOT / "paper_final_tables"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        if value.strip() == "" or value.lower() == "nan" or value.lower() == "n/a":
            return "n/a"
        try:
            value = float(value)
        except ValueError:
            return value
    if isinstance(value, bool):
        return "True" if value else "False"
    return f"{float(value):.{digits}f}"


def maybe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "n/a", "none"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def adapter_sort_key(name: str) -> int:
    order = {"control": 0, "mix05": 1, "mix10": 2, "mix50": 3, "hack": 4}
    return order.get(name.lower(), 99)


def feature_label(name: str) -> str:
    mapping = {
        "p_hack_only": "Activation only",
        "activation_only": "Activation only",
        "entropy_only": "Entropy only",
        "temporal_internal": "Temporal internal",
        "context_only": "Context only",
        "p_hack_plus_entropy": "Activation + entropy",
        "activation_plus_entropy": "Activation + entropy",
        "internal_all": "Internal all",
        "internal_plus_context": "Internal + context",
    }
    return mapping.get(name, name)


def family_title(name: str) -> str:
    return name[0].upper() + name[1:].lower() if name else name


def family_rank(name: str) -> int:
    return {"Qwen": 0, "Llama": 1, "Falcon": 2, "pooled": 3}.get(name, 99)


def normalize_target(name: str) -> str:
    mapping = {
        "bad_buy_action_t1": "bad_buy_t1",
        "low_reward_buy_action_t1": "low_reward_buy_t1",
        "buy_action_t1": "buy_t1",
    }
    return mapping.get(name, name)


def stable_target(row: Dict[str, str]) -> bool:
    n = maybe_float(row.get("N")) or 0
    positives = maybe_float(row.get("positives")) or 0
    negatives = n - positives
    return positives >= 10 and negatives >= 10


def select_clean_steering(rows: Iterable[Dict[str, str]]) -> List[Dict[str, object]]:
    keep = {
        ("mix05", "steer_always_a025"),
        ("mix05", "steer_gated_a05_tau0"),
        ("mix10", "steer_always_a025"),
        ("mix10", "steer_gated_a05_tau0"),
        ("mix10", "steer_gated_a10_tau0"),
        ("mix50", "steer_always_a025"),
    }
    out = []
    for row in rows:
        key = (row.get("adapter", ""), row.get("condition", ""))
        if key not in keep:
            continue
        delta_proxy = maybe_float(
            row.get("delta_proxy")
            or row.get("delta_steered_minus_unsteered_proxy_score_without_success")
        )
        delta_easy = maybe_float(
            row.get("delta_easy")
            or row.get("delta_steered_minus_unsteered_easy_grader_count")
        )
        delta_fake = maybe_float(
            row.get("delta_fake")
            or row.get("delta_steered_minus_unsteered_fake_completion_count")
        )
        steering_rate = maybe_float(
            row.get("steering_rate")
            or row.get("delta_steered_minus_unsteered_steering_rate")
            or row.get("steering_rate_steered_mean")
        )
        out.append(
            {
                "family": family_title(row.get("model_family", "qwen")),
                "adapter": row.get("adapter", ""),
                "condition": row.get("condition", ""),
                "steering_rate": steering_rate,
                "delta_proxy": delta_proxy,
                "proxy_reduction": None if delta_proxy is None else -delta_proxy,
                "delta_easy": delta_easy,
                "delta_fake": delta_fake,
                "success_preserved": row.get("success_preserved", ""),
                "exploit_reduced": row.get("exploit_reduced", ""),
            }
        )
    out.sort(key=lambda r: (family_rank(r["family"]), adapter_sort_key(str(r["adapter"])), str(r["condition"])))
    return out


def best_by_gain(rows: Iterable[Dict[str, str]]) -> Optional[Dict[str, str]]:
    rows = list(rows)
    if not rows:
        return None
    return max(rows, key=lambda r: maybe_float(r.get("AUPRC_gain")) or -999.0)


def aggregate_risk_state_rows(
    behavior_rows: List[Dict[str, str]], qwen_rows: List[Dict[str, str]], raw_rows: List[Dict[str, str]]
) -> List[Dict[str, object]]:
    behavior_by_key = {(r["family"], r["adapter"]): r for r in behavior_rows}
    out: List[Dict[str, object]] = []

    for row in qwen_rows:
        adapter = row["adapter"]
        behavior = behavior_by_key.get(("Qwen", adapter), {})
        out.append(
            {
                "family": "Qwen",
                "adapter": adapter,
                "ttc_or_regime": row.get("ttc_or_regime", ""),
                "environment": row.get("environment", ""),
                "mean_p_hack": maybe_float(row.get("mean_p_hack")),
                "mean_entropy": maybe_float(row.get("mean_entropy")),
                "behavior_risk_rate": maybe_float(row.get("behavior_risk_rate")),
                "exploit_action_rate": maybe_float(row.get("exploit_action_rate")),
                "bad_action_rate": maybe_float(row.get("bad_action_rate")),
            }
        )

    for family in ("Llama", "Falcon"):
        for row in raw_rows:
            if row.get("family") != family:
                continue
            mean_p = maybe_float(row.get("mean_p_hack"))
            mean_e = maybe_float(row.get("mean_entropy"))
            beh = maybe_float(row.get("behavior_risk_rate"))
            if mean_p is None and mean_e is None and beh is None:
                continue
            adapter = row.get("adapter", "")
            behavior = behavior_by_key.get((family, adapter), {})
            out.append(
                {
                    "family": family,
                    "adapter": adapter,
                    "ttc_or_regime": row.get("ttc", ""),
                    "environment": row.get("environment", ""),
                    "mean_p_hack": mean_p,
                    "mean_entropy": mean_e,
                    "behavior_risk_rate": beh,
                    "exploit_action_rate": maybe_float(behavior.get("exploit_action_rate")),
                    "bad_action_rate": maybe_float(behavior.get("bad_action_rate")),
                }
            )

    out.sort(key=lambda r: (family_rank(str(r["family"])), adapter_sort_key(str(r["adapter"])), str(r["ttc_or_regime"])))
    return out


def summarize_family_risk_state(rows: List[Dict[str, object]], family: str) -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}
    adapters = sorted({str(r["adapter"]) for r in rows if r["family"] == family}, key=adapter_sort_key)
    for adapter in adapters:
        sub = [r for r in rows if r["family"] == family and r["adapter"] == adapter]
        mean_p = [r["mean_p_hack"] for r in sub if r["mean_p_hack"] is not None]
        mean_e = [r["mean_entropy"] for r in sub if r["mean_entropy"] is not None]
        mean_b = [r["behavior_risk_rate"] for r in sub if r["behavior_risk_rate"] is not None]
        mean_x = [r["exploit_action_rate"] for r in sub if r["exploit_action_rate"] is not None]
        mean_bad = [r["bad_action_rate"] for r in sub if r["bad_action_rate"] is not None]
        out[adapter] = {
            "mean_p_hack": sum(mean_p) / len(mean_p) if mean_p else None,
            "mean_entropy": sum(mean_e) / len(mean_e) if mean_e else None,
            "behavior_risk_rate": sum(mean_b) / len(mean_b) if mean_b else None,
            "exploit_action_rate": sum(mean_x) / len(mean_x) if mean_x else None,
            "bad_action_rate": sum(mean_bad) / len(mean_bad) if mean_bad else None,
        }
    return out


def markdown_table(rows: List[Dict[str, object]], headers: List[Tuple[str, str]]) -> str:
    lines = ["| " + " | ".join(h for h, _ in headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        cells = []
        for _, key in headers:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(fmt(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    files_read: List[str] = []

    behavior_path = ROOT / "paper_final_tables/table_behavior_transfer.csv"
    prediction_path = ROOT / "paper_final_tables/table_prediction_ablation_all_families.csv"
    qwen_risk_path = ROOT / "paper_final_tables/table_risk_state_vs_behavior_qwen.csv"
    raw_risk_path = ROOT / "paper_analysis_outputs/risk_state_vs_behavior_summary_qwen.csv"
    qwen_steering_path = ROOT / "BALROG/hf_alfworld_steering/hf_steering_analysis_mix_sweep/steering_effects.csv"
    webshop_pred_path = ROOT / "paper_analysis_outputs/context_calibrated_prediction_ablation_webshop.csv"
    falcon_steer_path = ROOT / "BALROG/hf_alfworld_steering/hf_steering_analysis_mix_sweep_falcon/steering_effects.csv"
    llama_steer_path = ROOT / "BALROG/hf_alfworld_steering/hf_steering_analysis_mix_sweep_llama/steering_effects.csv"

    behavior_rows = read_csv(behavior_path)
    prediction_rows = read_csv(prediction_path)
    qwen_risk_rows = read_csv(qwen_risk_path)
    raw_risk_rows = read_csv(raw_risk_path)
    qwen_steering_rows = read_csv(qwen_steering_path)
    webshop_rows = read_csv(webshop_pred_path)
    falcon_steering_rows = read_csv(falcon_steer_path)
    llama_steering_rows = read_csv(llama_steer_path)

    files_read.extend(
        [
            str(behavior_path),
            str(prediction_path),
            str(qwen_risk_path),
            str(raw_risk_path),
            str(qwen_steering_path),
            str(webshop_pred_path),
            str(falcon_steer_path),
            str(llama_steer_path),
        ]
    )

    behavior_out = []
    for row in behavior_rows:
        behavior_out.append(
            {
                "family": row["family"],
                "adapter": row["adapter"],
                "n_steps": row["n_steps"],
                "bad_action_rate": fmt(row["bad_action_rate"]),
                "exploit_action_rate": fmt(row["exploit_action_rate"]),
                "gameable_hack_action_rate": fmt(row["gameable_hack_action_rate"]),
                "easy_grader_action_rate": fmt(row["easy_grader_action_rate"]),
                "fake_completion_action_rate": fmt(row["fake_completion_action_rate"]),
                "mean_reasoning_p_hack": fmt(row["mean_reasoning_p_hack"]),
                "mean_entropy": fmt(row["mean_entropy"]),
            }
        )
    behavior_out.sort(key=lambda r: (family_rank(r["family"]), adapter_sort_key(r["adapter"])))
    write_csv(
        OUTDIR / "all_family_behavior_comparison.csv",
        behavior_out,
        [
            "family",
            "adapter",
            "n_steps",
            "bad_action_rate",
            "exploit_action_rate",
            "gameable_hack_action_rate",
            "easy_grader_action_rate",
            "fake_completion_action_rate",
            "mean_reasoning_p_hack",
            "mean_entropy",
        ],
    )

    prediction_out = []
    for row in prediction_rows:
        prediction_out.append(
            {
                "family": row["family"],
                "environment": "ALFWorld",
                "target": normalize_target(row["target"]),
                "feature_group": feature_label(row["feature_group"]),
                "N": row["N"],
                "positives": row["positives"],
                "base_rate": fmt(row["base_rate"]),
                "AUROC": fmt(row["AUROC"]),
                "AUPRC": fmt(row["AUPRC"]),
                "AUPRC_gain": fmt(row["AUPRC_gain"]),
                "Recall@20": fmt(row["Recall@20"]),
                "stable": "yes" if stable_target(row) else "no",
            }
        )
    for row in webshop_rows:
        prediction_out.append(
            {
                "family": row["family"],
                "environment": "WebShop",
                "target": normalize_target(row["target"]),
                "feature_group": feature_label(row["feature_group"]),
                "N": row["N"],
                "positives": row["positives"],
                "base_rate": fmt(row["base_rate"]),
                "AUROC": fmt(row["AUROC"]),
                "AUPRC": fmt(row["AUPRC"]),
                "AUPRC_gain": fmt(row["AUPRC_gain"]),
                "Recall@20": fmt(row["Recall@20"]),
                "stable": "yes" if stable_target(row) else "no",
            }
        )
    prediction_out.sort(key=lambda r: (r["environment"], family_rank(r["family"]), r["target"], r["feature_group"]))
    write_csv(
        OUTDIR / "all_family_prediction_comparison.csv",
        prediction_out,
        [
            "family",
            "environment",
            "target",
            "feature_group",
            "N",
            "positives",
            "base_rate",
            "AUROC",
            "AUPRC",
            "AUPRC_gain",
            "Recall@20",
            "stable",
        ],
    )

    risk_state_out = aggregate_risk_state_rows(behavior_rows, qwen_risk_rows, raw_risk_rows)
    write_csv(
        OUTDIR / "all_family_risk_state_comparison.csv",
        [
            {
                "family": r["family"],
                "adapter": r["adapter"],
                "ttc_or_regime": r["ttc_or_regime"],
                "environment": r["environment"],
                "mean_p_hack": fmt(r["mean_p_hack"]),
                "mean_entropy": fmt(r["mean_entropy"]),
                "behavior_risk_rate": fmt(r["behavior_risk_rate"]),
                "exploit_action_rate": fmt(r["exploit_action_rate"]),
                "bad_action_rate": fmt(r["bad_action_rate"]),
            }
            for r in risk_state_out
        ],
        [
            "family",
            "adapter",
            "ttc_or_regime",
            "environment",
            "mean_p_hack",
            "mean_entropy",
            "behavior_risk_rate",
            "exploit_action_rate",
            "bad_action_rate",
        ],
    )

    qwen_steer_norm = select_clean_steering(qwen_steering_rows)
    falcon_steer_norm = select_clean_steering(falcon_steering_rows)
    llama_steer_norm = select_clean_steering(llama_steering_rows)
    steering_out = qwen_steer_norm + llama_steer_norm + falcon_steer_norm
    write_csv(
        OUTDIR / "all_family_steering_comparison.csv",
        [
            {
                "family": r["family"],
                "adapter": r["adapter"],
                "condition": r["condition"],
                "steering_rate": fmt(r["steering_rate"], 4),
                "delta_proxy": fmt(r["delta_proxy"]),
                "proxy_reduction": fmt(r["proxy_reduction"]),
                "delta_easy": fmt(r["delta_easy"]),
                "delta_fake": fmt(r["delta_fake"]),
                "success_preserved": r["success_preserved"],
                "exploit_reduced": r["exploit_reduced"],
            }
            for r in steering_out
        ],
        [
            "family",
            "adapter",
            "condition",
            "steering_rate",
            "delta_proxy",
            "proxy_reduction",
            "delta_easy",
            "delta_fake",
            "success_preserved",
            "exploit_reduced",
        ],
    )

    # Markdown sections
    behavior_headers = [
        ("Family", "family"),
        ("Adapter", "adapter"),
        ("N", "n_steps"),
        ("Bad", "bad_action_rate"),
        ("Exploit", "exploit_action_rate"),
        ("Gameable", "gameable_hack_action_rate"),
        ("Easy", "easy_grader_action_rate"),
        ("Fake", "fake_completion_action_rate"),
        ("p_hack", "mean_reasoning_p_hack"),
        ("Entropy", "mean_entropy"),
    ]

    family_behavior_notes = []
    for fam in ("Qwen", "Llama", "Falcon"):
        sub = [r for r in behavior_rows if r["family"] == fam]
        max_exploit = max(sub, key=lambda r: maybe_float(r.get("exploit_action_rate")) or -1)
        max_bad = max(sub, key=lambda r: maybe_float(r.get("bad_action_rate")) or -1)
        hack = next((r for r in sub if r["adapter"] == "hack"), None)
        mix50 = next((r for r in sub if r["adapter"] == "mix50"), None)
        note = f"- `{fam}`: highest broad bad-action rate is `{max_bad['adapter']}` ({fmt(max_bad['bad_action_rate'])}); "
        note += f"highest explicit exploit rate is `{max_exploit['adapter']}` ({fmt(max_exploit['exploit_action_rate'])}). "
        if hack and mix50:
            note += (
                f"`Hack` has p_hack {fmt(hack['mean_reasoning_p_hack'])}; "
                f"`Mix50` has p_hack {fmt(mix50['mean_reasoning_p_hack'])} and exploit {fmt(mix50['exploit_action_rate'])}."
            )
        family_behavior_notes.append(note)

    pred_notes = []
    for fam in ("Qwen", "Llama", "Falcon"):
        fam_rows = [r for r in prediction_rows if r["family"] == fam]
        targets = sorted({r["target"] for r in fam_rows})
        for target in targets:
            sub = [r for r in fam_rows if r["target"] == target]
            best = best_by_gain(sub)
            act = next((r for r in sub if r["feature_group"] == "p_hack_only"), None)
            ent = next((r for r in sub if r["feature_group"] == "entropy_only"), None)
            ctx = next((r for r in sub if r["feature_group"] == "context_only"), None)
            ipc = next((r for r in sub if r["feature_group"] == "internal_plus_context"), None)
            pred_notes.append(
                f"- `{fam}` `{target}`: best saved group is `{feature_label(best['feature_group'])}` "
                f"(AUPRC gain {fmt(best['AUPRC_gain'])}, AUROC {fmt(best['AUROC'])}). "
                f"Activation-only {fmt(act['AUPRC_gain']) if act else 'n/a'}, "
                f"entropy-only {fmt(ent['AUPRC_gain']) if ent else 'n/a'}, "
                f"context-only {fmt(ctx['AUPRC_gain']) if ctx else 'n/a'}, "
                f"internal+context {fmt(ipc['AUPRC_gain']) if ipc else 'n/a'}."
            )

    risk_summaries = {fam: summarize_family_risk_state(risk_state_out, fam) for fam in ("Qwen", "Llama", "Falcon")}
    risk_notes = [
        "- `Qwen`: `Hack` saturates p_hack while `Mix50` carries the clearest exploit behavior; this is the cleanest semantic-shift example.",
        "- `Llama`: p_hack is sparse for mixed adapters in the saved summaries; bad-action behavior changes are visible, but explicit exploit evidence is weak.",
        "- `Falcon`: p_hack still separates regimes, but broad bad-action behavior is stronger than explicit exploit behavior, so the story is boundary/appendix evidence rather than a clean exploit narrative.",
    ]

    steering_notes = [
        "- `Qwen`: strongest clean main-sweep steering evidence; `Mix10` and `Mix50` always-on steering reduce proxy score by 2.000 and 2.200 with success preserved.",
        "- `Llama`: secondary steering support exists, including `Mix50` always-on proxy reduction of 5.900, but the sweep is less central to the paper than Qwen.",
        "- `Falcon`: steering is mixed; some gated and mixed rows reduce proxy behavior, but the family is less reliable as a clean main-paper intervention story.",
    ]

    behavior_table_md = markdown_table(behavior_out, behavior_headers)

    pred_compact_rows = []
    for row in prediction_out:
        if row["environment"] == "WebShop":
            continue
        pred_compact_rows.append(
            {
                "family": row["family"],
                "target": row["target"],
                "feature_group": row["feature_group"],
                "N": row["N"],
                "positives": row["positives"],
                "base_rate": row["base_rate"],
                "AUROC": row["AUROC"],
                "AUPRC": row["AUPRC"],
                "AUPRC_gain": row["AUPRC_gain"],
                "Recall@20": row["Recall@20"],
            }
        )
    pred_table_md = markdown_table(
        pred_compact_rows,
        [
            ("Family", "family"),
            ("Target", "target"),
            ("Group", "feature_group"),
            ("N", "N"),
            ("Pos", "positives"),
            ("Base", "base_rate"),
            ("AUROC", "AUROC"),
            ("AUPRC", "AUPRC"),
            ("Gain", "AUPRC_gain"),
            ("R@20", "Recall@20"),
        ],
    )

    risk_compact_rows = []
    for fam in ("Qwen", "Llama", "Falcon"):
        for adapter, vals in risk_summaries[fam].items():
            risk_compact_rows.append(
                {
                    "family": fam,
                    "adapter": adapter,
                    "mean_p_hack": fmt(vals["mean_p_hack"]),
                    "mean_entropy": fmt(vals["mean_entropy"]),
                    "behavior_risk_rate": fmt(vals["behavior_risk_rate"]),
                    "exploit_action_rate": fmt(vals["exploit_action_rate"]),
                    "bad_action_rate": fmt(vals["bad_action_rate"]),
                }
            )
    risk_table_md = markdown_table(
        risk_compact_rows,
        [
            ("Family", "family"),
            ("Adapter", "adapter"),
            ("p_hack", "mean_p_hack"),
            ("Entropy", "mean_entropy"),
            ("Risk", "behavior_risk_rate"),
            ("Exploit", "exploit_action_rate"),
            ("Bad", "bad_action_rate"),
        ],
    )

    steering_table_md = markdown_table(
        [
            {
                "family": r["family"],
                "adapter": r["adapter"],
                "condition": r["condition"],
                "steering_rate": fmt(r["steering_rate"], 4),
                "proxy_reduction": fmt(r["proxy_reduction"]),
                "delta_easy": fmt(r["delta_easy"]),
                "delta_fake": fmt(r["delta_fake"]),
                "success_preserved": r["success_preserved"],
                "exploit_reduced": r["exploit_reduced"],
            }
            for r in steering_out
        ],
        [
            ("Family", "family"),
            ("Adapter", "adapter"),
            ("Condition", "condition"),
            ("Steer", "steering_rate"),
            ("Proxy red.", "proxy_reduction"),
            ("Δeasy", "delta_easy"),
            ("Δfake", "delta_fake"),
            ("Success", "success_preserved"),
            ("Exploit", "exploit_reduced"),
        ],
    )

    webshop_table_md = markdown_table(
        [
            {
                "family": r["family"],
                "target": normalize_target(r["target"]),
                "feature_group": feature_label(r["feature_group"]),
                "AUROC": fmt(r["AUROC"]),
                "AUPRC": fmt(r["AUPRC"]),
                "AUPRC_gain": fmt(r["AUPRC_gain"]),
                "Recall@20": fmt(r["Recall@20"]),
                "positives": r["positives"],
                "stable": "yes" if stable_target(r) else "no",
            }
            for r in webshop_rows
        ],
        [
            ("Family", "family"),
            ("Target", "target"),
            ("Group", "feature_group"),
            ("AUROC", "AUROC"),
            ("AUPRC", "AUPRC"),
            ("Gain", "AUPRC_gain"),
            ("R@20", "Recall@20"),
            ("Pos", "positives"),
            ("Stable", "stable"),
        ],
    )

    main_appendix_rows = [
        {"evidence": "behavioral transfer", "qwen": "Strong", "llama": "Moderate", "falcon": "Moderate", "where": "Main", "reason": "Qwen gives the clearest exploit transfer; other families add robustness."},
        {"evidence": "prediction ablation", "qwen": "Strong", "llama": "Moderate", "falcon": "Moderate", "where": "Main", "reason": "Core paper claim; Qwen strongest, Llama/Falcon support activation-only insufficiency."},
        {"evidence": "risk-state map", "qwen": "Strong", "llama": "Moderate", "falcon": "Moderate", "where": "Main", "reason": "Shows structured internal states; Qwen most interpretable."},
        {"evidence": "WebShop prediction", "qwen": "Moderate", "llama": "Moderate", "falcon": "Weak", "where": "Appendix", "reason": "Public-benchmark generalization exists, but only one predictor setting is saved."},
        {"evidence": "steering", "qwen": "Strong", "llama": "Moderate", "falcon": "Appendix only", "where": "Main/Appendix", "reason": "Qwen is main-paper steering evidence; Llama supports; Falcon is mixed."},
        {"evidence": "temporal diagnostics", "qwen": "Moderate", "llama": "Weak", "falcon": "Weak", "where": "Appendix", "reason": "Diagnostic and heterogeneous, not clean headline evidence."},
    ]
    main_appendix_md = markdown_table(
        main_appendix_rows,
        [
            ("Evidence", "evidence"),
            ("Qwen", "qwen"),
            ("Llama", "llama"),
            ("Falcon", "falcon"),
            ("Main or Appendix?", "where"),
            ("Reason", "reason"),
        ],
    )

    claim_rows = [
        {"claim": "Reward-hack fine-tuning changes agentic action behavior.", "qwen": "Strong", "llama": "Moderate", "falcon": "Moderate", "overall": "Reward-hack fine-tuning changes agentic action behavior, with the clearest explicit exploit transfer in Qwen mixed regimes."},
        {"claim": "p_hack remains structured and separates adapter regimes.", "qwen": "Strong", "llama": "Moderate", "falcon": "Moderate", "overall": "Across families, reward-hack activation remains structured, though the cleanest separation is in Qwen."},
        {"claim": "high p_hack is not monotonic action-risk score.", "qwen": "Strong", "llama": "Moderate", "falcon": "Moderate", "overall": "High activation is a latent policy-state descriptor, not a monotonic action-risk score."},
        {"claim": "entropy/context improve next-step prediction.", "qwen": "Strong", "llama": "Moderate", "falcon": "Moderate", "overall": "Activation-only prediction is not uniformly sufficient; entropy and context often improve stable targets."},
        {"claim": "TTC/reasoning budget changes monitor semantics.", "qwen": "Moderate", "llama": "Appendix only", "falcon": "Appendix only", "overall": "Reasoning budget changes monitor semantics non-monotonically, with the strongest evidence in Qwen."},
        {"claim": "steering reduces proxy behavior in selected regimes.", "qwen": "Strong", "llama": "Moderate", "falcon": "Weak", "overall": "Steering reduces proxy behavior in selected mixed-adapter regimes; it is not universal."},
    ]
    claim_md = markdown_table(
        claim_rows,
        [
            ("Claim", "claim"),
            ("Qwen", "qwen"),
            ("Llama", "llama"),
            ("Falcon", "falcon"),
            ("Overall wording", "overall"),
        ],
    )

    md = f"""# All-Family Results Comparison

## 1. Executive Summary

- Qwen: strongest main family.
- Llama: useful secondary support, especially for stable `bad_action_t1` prediction and some steering support.
- Falcon: boundary / appendix evidence unless a specific `bad_action_t1` prediction row is singled out.
- Cross-family support is strongest for: behavioral transfer, structured `p_hack`, and activation-only insufficiency.
- Family-specific support is strongest for: explicit exploit transfer and clean semantic-shift examples in Qwen.

## 2. Behavioral Transfer Across Families

{behavior_table_md}

- Reward-hack fine-tuning changes action behavior in all three families, but the form of the effect differs by family.
- Qwen provides the clearest explicit proxy-exploit transfer: `Mix50` reaches exploit rate {fmt(next(r['exploit_action_rate'] for r in behavior_rows if r['family']=='Qwen' and r['adapter']=='mix50'))}.
- The pattern is not monotonic. In Qwen, `Hack` saturates p_hack ({fmt(next(r['mean_reasoning_p_hack'] for r in behavior_rows if r['family']=='Qwen' and r['adapter']=='hack'))}) but has much lower exploit than `Mix50`.
- Mix50 is present in the final comparison for all families.
{chr(10).join(family_behavior_notes)}

## 3. Prediction Ablation Across Families

{pred_table_md}

- Activation-only is weak or modest in several stable settings. For example, `Llama bad_action_t1` activation-only has AUPRC gain {fmt(next(r['AUPRC_gain'] for r in prediction_rows if r['family']=='Llama' and r['target']=='bad_action_t1' and r['feature_group']=='p_hack_only'))}.
- Entropy-only often improves over activation-only on broad bad-action targets.
- Context-only helps on `bad_action_t1`, especially for Qwen and Llama, but is not uniformly useful on sparse exploit targets.
- `Internal + context` is the strongest saved `bad_action_t1` group for Qwen, Llama, and Falcon.
- Qwen has the strongest clean exploit-target evidence. Llama has stable `bad_action_t1` results but insufficient explicit exploit positives.
- WebShop supports next-step prediction but not a full feature-group ablation.
{chr(10).join(pred_notes)}

## 4. Risk-State vs Behavior Across Families

{risk_table_md}

- `p_hack` separates adapter regimes across families, with the clearest policy-state geometry in Qwen.
- High `p_hack` does not monotonically imply high action risk.
- The clearest semantic-shift example is Qwen: `Hack` saturates `p_hack`, while `Mix50` carries the strongest exploit behavior.
- Mixed adapters, especially Qwen `Mix50`, often show higher behavior risk than `Hack`.
{chr(10).join(risk_notes)}

## 5. Steering Across Families

{steering_table_md}

- Current ALFWorld steering evidence is Qwen-first in the main sweep, but Llama and Falcon steering sweeps also exist on disk.
- Qwen mixed adapters benefit most cleanly in the saved compact table.
- Gated settings are not universally reliable across families.
- Steering should be framed as an intervention probe, not a universal mitigation.
{chr(10).join(steering_notes)}

## 6. WebShop Summary

{webshop_table_md}

- WebShop supports public-benchmark generalization for next-step prediction, but only with a single saved predictor setting (`Activation + entropy`), not a full ablation sweep.
- Strongest WebShop AUROC is Falcon `buy_t1` ({fmt(next(r['AUROC'] for r in webshop_rows if r['family']=='Falcon' and r['target']=='buy_t1'))}), but Falcon bad-buy targets are sparse (`positives=5`) and should be treated as unstable.
- Qwen and Llama provide cleaner stable WebShop support than Falcon for the paper narrative.
- WebShop is better treated as appendix / secondary support unless space allows a compact table.

## 7. Main-Paper vs Appendix Recommendation

{main_appendix_md}

## 8. Claim Support Matrix by Family

{claim_md}

## 9. Paper-Safe Final Claims

- Qwen provides the strongest evidence that reward-hack fine-tuning can transfer to proxy action selection in Gameable ALFWorld.
- Across families, activation-only prediction is not uniformly sufficient.
- Entropy and context-calibrated features improve next-step prediction in stable targets.
- High `p_hack` is a latent policy-state descriptor, not a monotonic action-risk score.
- Steering evidence is strongest in Qwen mixed adapters and should be framed as an intervention probe.
- WebShop supports next-step prediction generalization, but not a full feature-group ablation.

## 10. Claims to Avoid

- `p_hack` always rises before bad actions.
- Activation-only predicts all risk.
- All effects are monotonic in hack mixture.
- Steering fully prevents reward hacking.
- Falcon supports the same clean story as Qwen.
- WebShop has full ablation if only one predictor setting exists.

## Files Read

{chr(10).join(f"- `{p}`" for p in files_read)}

## Files Written

- `{OUTDIR / "ALL_FAMILY_COMPARISON.md"}`
- `{OUTDIR / "all_family_behavior_comparison.csv"}`
- `{OUTDIR / "all_family_prediction_comparison.csv"}`
- `{OUTDIR / "all_family_risk_state_comparison.csv"}`
- `{OUTDIR / "all_family_steering_comparison.csv"}`

## Main Recommendation for the Paper

- Main paper: Qwen behavioral transfer, Qwen/Falcon `bad_action_t1` prediction ablations, Qwen risk-state figure, Qwen steering.
- Secondary support: Llama `bad_action_t1` prediction and selected steering rows.
- Appendix: Falcon exploit/steering boundary cases, WebShop single-setting prediction table, temporal diagnostics.
"""

    (OUTDIR / "ALL_FAMILY_COMPARISON.md").write_text(md)

    print("Files read:")
    for p in files_read:
        print(f" - {p}")
    print("Files written:")
    for p in [
        OUTDIR / "ALL_FAMILY_COMPARISON.md",
        OUTDIR / "all_family_behavior_comparison.csv",
        OUTDIR / "all_family_prediction_comparison.csv",
        OUTDIR / "all_family_risk_state_comparison.csv",
        OUTDIR / "all_family_steering_comparison.csv",
    ]:
        print(f" - {p}")
    print("Main recommendation:")
    print(" - Qwen is the primary main-paper family; Llama is secondary support; Falcon is mostly appendix/boundary evidence.")


if __name__ == "__main__":
    main()
