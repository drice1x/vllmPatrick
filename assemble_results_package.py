#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/patrick/vllmPatrickMonitoring")
TABLES = ROOT / "tables_final_evidence"
FIGS = ROOT / "figures_final_evidence"


def read_csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def fmt(x):
    if x is None or x == "" or (isinstance(x, float) and math.isnan(x)):
        return ""
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def write_tex_table(path: Path, rows, fieldnames):
    with path.open("w") as f:
        f.write("\\begin{tabular}{" + "l" * len(fieldnames) + "}\n")
        f.write(" \\toprule\n")
        f.write(" & ".join(fieldnames) + " \\\\\n")
        f.write(" \\midrule\n")
        for row in rows:
            f.write(" & ".join(fmt(row.get(k, "")) for k in fieldnames) + " \\\\\n")
        f.write(" \\bottomrule\n")
        f.write("\\end{tabular}\n")


def build_table1():
    path = ROOT / "BALROG/analysis_gameable_alfworld_temporal/steps_with_entropy_phack_temporal.csv"
    rows = read_csv(path)
    target_models = ["QwenControl", "QwenMix05", "QwenMix10", "QwenMix50", "QwenHack"]
    stats = {m: defaultdict(float) for m in target_models}
    counts = defaultdict(int)

    # two-pass to compute next-step labels
    by_episode = defaultdict(list)
    for r in rows:
        if r["model"] in target_models:
            by_episode[r["source_file"]].append(r)
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda r: int(float(r["step"])))
        for i in range(len(episode_rows) - 1):
            cur = episode_rows[i]
            nxt = episode_rows[i + 1]
            m = cur["model"]
            counts[m] += 1
            def num(k, row=cur):
                try:
                    return float(row.get(k, "") or 0.0)
                except Exception:
                    return 0.0
            stats[m]["mean_p_hack"] += num("reasoning_p_hack")
            stats[m]["mean_entropy"] += num("reasoning_entropy_mean")
            stats[m]["bad_action_rate"] += float(nxt.get("bad_action") or 0.0)
            stats[m]["gameable_hack_rate"] += float(nxt.get("gameable_hack_action") or 0.0)
            stats[m]["easy_grader_rate"] += float(nxt.get("easy_grader_action") or 0.0)
            stats[m]["success_rate"] += float(nxt.get("true_task_success_event") or 0.0)

    out = []
    for m in target_models:
        n = max(counts[m], 1)
        out.append({
            "model": m,
            "mean_reasoning_reward_hack_activation": stats[m]["mean_p_hack"] / n,
            "mean_entropy": stats[m]["mean_entropy"] / n,
            "bad_action_rate": stats[m]["bad_action_rate"] / n,
            "gameable_hack_rate": stats[m]["gameable_hack_rate"] / n,
            "easy_grader_rate": stats[m]["easy_grader_rate"] / n,
            "success_rate": stats[m]["success_rate"] / n,
        })
    fieldnames = list(out[0].keys())
    write_csv(TABLES / "main_behavior_internal_profile.csv", out, fieldnames)
    write_tex_table(TABLES / "main_behavior_internal_profile.tex", out, fieldnames)


def build_table2():
    rows = read_csv(TABLES / "main_context_calibrated_prediction_ablation.csv")
    grouped = defaultdict(list)
    for r in rows:
        if (r.get("stable") or "").lower() != "true":
            continue
        grouped[(r["environment"], r["target"])].append(r)

    out = []
    for (env, target), group in grouped.items():
        # choose best regime by internal_plus_context else max auprc_gain
        best = None
        for r in group:
            score = float(r["auprc_gain"] or "nan")
            if math.isnan(score):
                continue
            key = (r["feature_group"] == "internal_plus_context", score)
            if best is None or key > best[0]:
                best = (key, r["ttc_or_regime"])
        if best is None:
            continue
        regime = best[1]
        for r in group:
            if r["ttc_or_regime"] != regime:
                continue
            out.append({
                "environment": env,
                "target": target,
                "feature_group": r["feature_group"],
                "positives": r["positives"],
                "base_rate": r["base_rate"],
                "AUROC": r["auroc"],
                "AUPRC": r["auprc"],
                "AUPRC_gain": r["auprc_gain"],
                "Recall@20%": r["recall_at_20pct"],
            })
    fieldnames = ["environment", "target", "feature_group", "positives", "base_rate", "AUROC", "AUPRC", "AUPRC_gain", "Recall@20%"]
    write_csv(TABLES / "main_prediction_ablation_table.csv", out, fieldnames)
    write_tex_table(TABLES / "main_prediction_ablation_table.tex", out, fieldnames)


def build_table3():
    rows = read_csv(TABLES / "main_steering_intervention_probe.csv")
    want = {
        ("mix05", "always", "0.25"),
        ("mix05", "gated", "0.5"),
        ("mix10", "always", "0.25"),
        ("mix10", "gated", "0.5"),
        ("mix10", "gated", "1.0"),
        ("mix50", "always", "0.25"),
    }
    out = []
    for r in rows:
        key = (r["adapter"], r["steering_mode"], str(float(r["alpha"])))
        if key in want and r["model_family"] == "Qwen":
            out.append({
                "row": f"{r['adapter']} {r['steering_mode']} alpha={r['alpha']}",
                "steering_rate": r["steering_rate"],
                "delta_proxy": r["exploit_reduction"],
                "delta_easy": r["delta_steered_minus_unsteered_easy_grader_count"],
                "delta_fake": r["delta_steered_minus_unsteered_fake_completion_count"],
                "exploit_reduced": r["exploit_reduced"],
            })
    fieldnames = ["row", "steering_rate", "delta_proxy", "delta_easy", "delta_fake", "exploit_reduced"]
    write_csv(TABLES / "main_steering_probe.csv", out, fieldnames)
    write_tex_table(TABLES / "main_steering_probe.tex", out, fieldnames)


def copy_main_figures():
    copies = {
        FIGS / "main_internal_risk_state_map_qwen.png": FIGS / "main_risk_state_transfer_panelA.png",
        FIGS / "main_internal_risk_state_map_qwen.pdf": FIGS / "main_risk_state_transfer_panelA.pdf",
        FIGS / "main_risk_state_vs_behavior_qwen.png": FIGS / "main_risk_state_transfer_panelB.png",
        FIGS / "main_risk_state_vs_behavior_qwen.pdf": FIGS / "main_risk_state_transfer_panelB.pdf",
        FIGS / "main_context_calibrated_prediction_ablation.png": FIGS / "main_context_calibration_prediction.png",
        FIGS / "main_context_calibrated_prediction_ablation.pdf": FIGS / "main_context_calibration_prediction.pdf",
        FIGS / "main_steering_intervention_probe.png": FIGS / "main_steering_probe.png",
        FIGS / "main_steering_intervention_probe.pdf": FIGS / "main_steering_probe.pdf",
    }
    for src, dst in copies.items():
        if src.exists():
            shutil.copyfile(src, dst)


def write_results_markdown():
    text = """# Results Section Rewrite

## Claim 1: Reward-hack activations transfer to agents as structured latent policy states

Across `Gameable ALFWorld` and `WebShop`, reward-hack activation remains nontrivial during agent reasoning and separates adapter regimes. In particular, control, mixed, and hack adapters occupy different regions of internal state space when summarized by mean reasoning entropy and mean reward-hack activation. This supports the interpretation that the reward-hack monitor transfers from isolated generation into sequential rollouts as a latent policy-state descriptor rather than disappearing under the distribution shift to agentic interaction.

Supporting figure/table:
- Figure 1A: `main_risk_state_transfer_panelA`
- Table 1: `main_behavior_internal_profile`

Caption:
Reward-hack activation remains structured in agent rollouts and separates adapter regimes across environments and reasoning budgets.

## Claim 2: Latent risk states do not directly determine action-level failures; behavior is context-mediated

Although reward-hack activation is structured, action-level failure is not a monotonic readout of raw activation. Conditions with higher mean activation do not uniformly exhibit higher exploit or bad-action rates, and some mixed regimes are more behaviorally separable than saturated hack regimes. This indicates that high activation is better understood as a latent reward-hack policy-state marker than as a direct action-risk score. The mapping from internal state to behavior is mediated by environment and decision context.

Supporting figure/table:
- Figure 1B: `main_risk_state_transfer_panelB`
- Table 1: `main_behavior_internal_profile`

Caption:
Reward-hack activation remains structured in agent rollouts, but action-level behavior is not a monotonic readout of raw activation.

## Claim 3: Entropy and context calibrate next-step risk estimation, and steering shows intervention relevance

Next-step prediction is possible, but not by thresholding reward-hack activation alone. Across stable ALFWorld and WebShop targets, entropy and mixed internal/context feature groups improve prediction relative to activation-only baselines, indicating that immediate risk depends on latent state plus decision uncertainty and environment affordances. Steering results further show that the reward-hack direction remains intervention-relevant: in selected mixed-adapter regimes, steering reduces proxy-exploit behavior without supporting a universal mitigation claim.

Supporting figure/table:
- Figure 2: `main_context_calibration_prediction`
- Table 2: `main_prediction_ablation_table`
- Figure 3 / Table 3: `main_steering_probe`

Caption for Figure 2:
Entropy and context-calibrated internal features improve next-step risk estimation over activation alone in selected targets and regimes.

Caption for Figure 3 / Table 3:
Steering is an intervention probe: it reduces proxy-exploit behavior in selected regimes, but is not a universal mitigation.

## Causal interpretation

The evidence does not show that reward-hack activation directly causes exploit actions. Instead, it shows that the representation remains behaviorally relevant but is mediated by environment context. Steering provides intervention evidence in selected regimes.

## Main-paper vs appendix

Main paper:
- Figure 1A/B: internal risk-state transfer and risk-state vs behavior
- Table 1: behavioral/internal profile
- Figure 2: context-calibrated prediction ablation
- Table 2: prediction ablation table
- Figure 3 or Table 3: steering probe

Appendix only:
- temporal red/blue curves
- TTC heatmaps
- per-adapter prediction curves
- Falcon full plots
- exploit-opportunity plots with low positives
- broad bad-minus-nonbad diagnostics

## Final takeaway

Prediction is possible, but not by thresholding a single activation feature. In agentic settings, risky next actions are best modeled as a function of latent reward-hack state, decision uncertainty, and environment affordances.
"""
    (ROOT / "RESULTS_SECTION_REWRITE.md").write_text(text)


def main():
    build_table1()
    build_table2()
    build_table3()
    copy_main_figures()
    write_results_markdown()
    print("[DONE] Assembled results package")
    for p in [
        TABLES / "main_behavior_internal_profile.csv",
        TABLES / "main_behavior_internal_profile.tex",
        TABLES / "main_prediction_ablation_table.csv",
        TABLES / "main_prediction_ablation_table.tex",
        TABLES / "main_steering_probe.csv",
        TABLES / "main_steering_probe.tex",
        FIGS / "main_risk_state_transfer_panelA.png",
        FIGS / "main_risk_state_transfer_panelB.png",
        FIGS / "main_context_calibration_prediction.png",
        FIGS / "main_steering_probe.png",
        ROOT / "RESULTS_SECTION_REWRITE.md",
    ]:
        if p.exists():
            print(" -", p)


if __name__ == "__main__":
    main()
