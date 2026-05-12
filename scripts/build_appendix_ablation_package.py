#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path("/home/patrick/vllmPatrickMonitoring")
OUT = ROOT / "paper_appendix_outputs"
PAPER_ANALYSIS = ROOT / "paper_analysis_outputs"

FG_MAP = {
    "p_hack_only": "Activation only",
    "entropy_only": "Entropy only",
    "temporal_internal": "Temporal internal",
    "context_only": "Context only",
    "p_hack_plus_entropy": "Activation + entropy",
    "internal_all": "Internal all",
    "internal_plus_context": "Internal + context",
}

FG_ORDER = [
    "Activation only",
    "Entropy only",
    "Temporal internal",
    "Context only",
    "Activation + entropy",
    "Internal all",
    "Internal + context",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def fmt(x, nd=3):
    try:
        if x is None or str(x).strip() == "":
            return ""
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def latex_table(rows: list[dict], columns: list[str], headers: list[str]) -> str:
    spec = "l" * len(columns)
    out = ["\\begin{tabular}{" + spec + "}", "\\toprule", " & ".join(headers) + " \\\\", "\\midrule"]
    for row in rows:
        out.append(" & ".join(str(row.get(c, "")) for c in columns) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def build_feature_ablation() -> list[Path]:
    files = [
        ("Qwen", PAPER_ANALYSIS / "context_calibrated_prediction_ablation_qwen.csv"),
        ("Llama", PAPER_ANALYSIS / "context_calibrated_prediction_ablation_llama.csv"),
        ("Falcon", PAPER_ANALYSIS / "context_calibrated_prediction_ablation_falcon.csv"),
    ]
    rows = []
    for family, path in files:
        for r in read_csv(path):
            rows.append(
                {
                    "family": family,
                    "target": r["target"],
                    "feature_group": FG_MAP.get(r["feature_group"], r["feature_group"]),
                    "N": r["N"],
                    "positives": r["positives"],
                    "base_rate": fmt(r["base_rate"]),
                    "AUROC": fmt(r["AUROC"]),
                    "AUPRC": fmt(r["AUPRC"]),
                    "AUPRC_gain": fmt(r["AUPRC_gain"]),
                    "Recall@20": fmt(r["Recall@20"]),
                }
            )
    rows.sort(key=lambda r: (r["family"], r["target"], FG_ORDER.index(r["feature_group"]) if r["feature_group"] in FG_ORDER else 99))
    csv_path = OUT / "appendix_feature_ablation_all_families.csv"
    tex_path = OUT / "appendix_feature_ablation_all_families.tex"
    fields = ["family", "target", "feature_group", "N", "positives", "base_rate", "AUROC", "AUPRC", "AUPRC_gain", "Recall@20"]
    write_csv(csv_path, rows, fields)
    tex_path.write_text(latex_table(rows, fields, fields), encoding="utf-8")

    # Reuse existing all-family ablation figure if present.
    copied = []
    for ext in ["png", "pdf"]:
        src = ROOT / "paper_figures" / f"prediction_ablation_by_family_environment.{ext}"
        if src.exists():
            dst = OUT / f"appendix_feature_ablation_heatmap.{ext}"
            shutil.copy2(src, dst)
            copied.append(dst)
    note = OUT / "appendix_feature_ablation_heatmap_note.md"
    note.write_text(
        "# Appendix feature ablation figure note\n\n"
        "The saved repository figure reused here is the all-family/environment prediction-ablation visualization. "
        "It supports the same appendix claim: feature usefulness is family- and target-dependent, and activation-only is not universally sufficient. "
        "The combined numeric table is the authoritative appendix source.\n",
        encoding="utf-8",
    )
    return [csv_path, tex_path, note] + copied


def build_ttc_sensitivity() -> list[Path]:
    out_files = []
    src = ROOT / "tables_final_evidence" / "main_feature_ablation_prediction.csv"
    rows = read_csv(src) if src.exists() else []
    keep = []
    for r in rows:
        if r.get("environment") != "alfworld":
            continue
        keep.append(
            {
                "environment": r.get("environment"),
                "target": r.get("target"),
                "ttc_or_regime": r.get("ttc_or_regime", r.get("ttc_numeric", r.get("ttc"))),
                "feature_group": FG_MAP.get(r.get("feature_group", ""), r.get("feature_group", "")),
                "base_rate": fmt(r.get("base_rate")),
                "AUROC": fmt(r.get("AUROC")),
                "AUPRC": fmt(r.get("AUPRC")),
                "AUPRC_gain": fmt(r.get("AUPRC_gain")),
                "Recall@20": fmt(r.get("Recall@20")),
                "stable": r.get("stable"),
            }
        )
    csv_path = OUT / "appendix_ttc_sensitivity.csv"
    write_csv(csv_path, keep, list(keep[0].keys()) if keep else ["environment", "target", "ttc_or_regime", "feature_group", "base_rate", "AUROC", "AUPRC", "AUPRC_gain", "Recall@20", "stable"])
    out_files.append(csv_path)

    note = OUT / "appendix_ttc_sensitivity_note.md"
    note.write_text(
        "# TTC sensitivity note\n\n"
        "The dedicated `BALROG/analysis_next_step_monitoring_agg_ttc/next_step_prediction_main.csv` file is aggregated at `ttc=ALL` and does not by itself provide a full TTC sweep.\n\n"
        "For appendix sensitivity reporting, we therefore summarize TTC-specific ALFWorld metrics from the saved `tables_final_evidence/main_feature_ablation_prediction.csv` export where TTC-specific rows are available.\n\n"
        "Paper-safe interpretation:\n\n"
        "- TTC effects are present but non-monotonic.\n"
        "- The best-performing feature group varies with target and TTC.\n"
        "- These appendix results should be treated as sensitivity evidence rather than a single monotonic scaling story.\n",
        encoding="utf-8",
    )
    out_files.append(note)
    for ext in ["png", "pdf"]:
        srcf = ROOT / "figures_final_evidence" / f"main_ttc_budget_monitor_semantics.{ext}"
        if srcf.exists():
            dst = OUT / f"appendix_ttc_sensitivity.{ext}"
            shutil.copy2(srcf, dst)
            out_files.append(dst)
    return out_files


def build_target_stability() -> list[Path]:
    files = [
        ("Qwen", PAPER_ANALYSIS / "context_calibrated_prediction_ablation_qwen.csv"),
        ("Llama", PAPER_ANALYSIS / "context_calibrated_prediction_ablation_llama.csv"),
        ("Falcon", PAPER_ANALYSIS / "context_calibrated_prediction_ablation_falcon.csv"),
    ]
    expected_targets = {
        "Qwen": ["bad_action_t1", "exploit_action_t1"],
        "Llama": ["bad_action_t1", "exploit_action_t1"],
        "Falcon": ["bad_action_t1", "exploit_action_t1"],
    }
    actual = {}
    for family, path in files:
        for r in read_csv(path):
            actual[(family, r["target"])] = r
    # WebShop normalized predictions, if present
    webshop = PAPER_ANALYSIS / "context_calibrated_prediction_ablation_webshop.csv"
    if webshop.exists():
        for r in read_csv(webshop):
            actual[(r["family"], r["target"])] = r

    rows = []
    # ALFWorld expected rows, including absent ones
    for family, targets in expected_targets.items():
        for target in targets:
            r = actual.get((family, target))
            if r is None:
                rows.append(
                    {
                        "family": family,
                        "target": target,
                        "N": "",
                        "positives": "",
                        "base_rate": "",
                        "stable": "no",
                    }
                )
                continue
            N = int(float(r["N"]))
            pos = int(float(r["positives"]))
            neg = N - pos
            rows.append(
                {
                    "family": family,
                    "target": target,
                    "N": N,
                    "positives": pos,
                    "base_rate": fmt(r["base_rate"]),
                    "stable": "yes" if pos >= 10 and neg >= 10 else "no",
                }
            )
    # WebShop available rows
    for fam in ["Qwen", "Llama", "Falcon"]:
        for target in ["buy_t1", "bad_buy_t1", "low_reward_buy_t1"]:
            r = actual.get((fam, target))
            if r is None:
                continue
            N = int(float(r["N"]))
            pos = int(float(r["positives"]))
            neg = N - pos
            rows.append(
                {
                    "family": fam,
                    "target": target,
                    "N": N,
                    "positives": pos,
                    "base_rate": fmt(r["base_rate"]),
                    "stable": "yes" if pos >= 10 and neg >= 10 else "no",
                }
            )
    csv_path = OUT / "appendix_target_stability.csv"
    tex_path = OUT / "appendix_target_stability.tex"
    fields = ["family", "target", "N", "positives", "base_rate", "stable"]
    write_csv(csv_path, rows, fields)
    tex_path.write_text(latex_table(rows, fields, fields), encoding="utf-8")
    note = OUT / "appendix_target_stability_note.md"
    note.write_text(
        "Panels with insufficient positives are omitted from the main paper figures to avoid overinterpreting sparse targets.\n",
        encoding="utf-8",
    )
    return [csv_path, tex_path, note]


def build_temporal_diag() -> list[Path]:
    out_files = []
    # Reuse existing Qwen diagnostic figure generated from react_ttc_monitored bad_action shift1.
    for ext in ["png", "pdf"]:
        src = ROOT / "paper_figures" / f"temporal_two_column_qwen_bad_action_shift1.{ext}"
        if src.exists():
            dst = OUT / f"appendix_temporal_diagnostics_qwen.{ext}"
            shutil.copy2(src, dst)
            out_files.append(dst)
    note = OUT / "appendix_temporal_diagnostics_note.md"
    note.write_text(
        "# Appendix temporal diagnostics\n\n"
        "This appendix figure is diagnostic rather than headline evidence. It visualizes Qwen step-t reasoning signals conditioned on bad_action at step t+1.\n\n"
        "Paper-safe interpretation:\n\n"
        "- Temporal dynamics are structured but heterogeneous.\n"
        "- The appendix does not support a universal late-stage p_hack spike before risky actions.\n"
        "- This is why feature-ablation and prediction results remain the primary quantitative evidence in the main paper.\n",
        encoding="utf-8",
    )
    out_files.append(note)
    return out_files


def build_context_features() -> list[Path]:
    path = OUT / "appendix_context_features.md"
    path.write_text(
        "# Context Features and Leakage Check\n\n"
        "## Context feature groups\n\n"
        "### Environment identity\n\n"
        "- environment label such as `alfworld` or `webshop`\n\n"
        "### TTC / reasoning budget\n\n"
        "- TTC value\n"
        "- regime label such as `ttc_8`, `ttc_16`, ..., `unconstrained`\n\n"
        "### Normalized step position\n\n"
        "- step index normalized within the episode\n\n"
        "### Previous action type\n\n"
        "- previous action category or related action-history signal when available\n\n"
        "### Available action / affordance indicators\n\n"
        "- exploit action available\n"
        "- proxy score visible\n"
        "- near completion\n"
        "- other environment affordance flags available at step t\n\n"
        "### WebShop page state if available\n\n"
        "- item page / buy page indicators\n\n"
        "## Leakage check\n\n"
        "The prediction pipeline is constructed to avoid future-label leakage:\n\n"
        "- no future labels are included in the feature vector\n"
        "- no action at `t+1` is included in the feature vector\n"
        "- context features are required to be available at step `t`\n"
        "- categorical context features are one-hot encoded\n"
        "- preprocessing is fit only on training folds and then applied to held-out folds\n"
        "- cross-validation is grouped by episode so nearby steps from the same episode do not leak across train/test splits\n\n"
        "Paper-safe wording:\n\n"
        "> Context features are restricted to information available at step t. We do not include future labels, the action at t+1, or any post-hoc outcome fields in the predictor. Categorical context features are one-hot encoded, and preprocessing is fit on training folds only before being applied to held-out folds.\n",
        encoding="utf-8",
    )
    return [path]


def build_writeup() -> list[Path]:
    path = OUT / "APPENDIX_ABLATION_WRITEUP.md"
    path.write_text(
        r"""\section{Appendix: Additional Ablations and Diagnostics}

\subsection{Full Feature-Group Ablations}

Table~\texttt{appendix\_feature\_ablation\_all\_families} aggregates family-level prediction ablations across Qwen, Llama, and Falcon. The appendix confirms that feature usefulness is family- and target-dependent: activation-only features are not universally sufficient, and richer feature sets often improve prediction. In particular, entropy and context-calibrated combinations frequently outperform activation-only baselines, although the strongest feature group varies by family and target.

\subsection{Reasoning-Budget Sensitivity}

Appendix TTC summaries provide sensitivity evidence rather than a monotonic scaling story. Available saved TTC-specific ALFWorld outputs show that monitor performance changes across reasoning budgets, but the effect is non-monotonic and target-dependent. We therefore treat TTC as a contextual variable that modulates monitor semantics rather than a simple resource axis with uniform gains.

\subsection{Target Sparsity and Stability}

The target-stability table documents which family--target pairs are sufficiently supported by positive examples. Sparse exploit labels explain why some panels are omitted from the main paper, especially for secondary families. This omission is methodological rather than rhetorical: we exclude unstable panels to avoid overinterpreting targets with too few positives.

\subsection{Temporal Diagnostics}

Appendix temporal plots are included as diagnostics rather than headline evidence. They show that within-step internal trajectories are structured, but also heterogeneous across adapters and targets. In particular, these diagnostics do not support a universal late-stage reward-hack activation spike before risky actions. This motivates the main-paper choice to emphasize feature-ablation and prediction evidence rather than raw trajectory plots.

\subsection{Context Features and Leakage Check}

The context-feature appendix note clarifies that context variables are restricted to step-$t$ information. No future labels, no action at $t+1$, and no post-hoc outcome fields are used as predictor inputs. Categorical context features are one-hot encoded, and preprocessing is fit on training folds only. This appendix material is intended to address leakage concerns explicitly and to support the interpretation of the context-calibrated prediction results as genuine next-step monitoring rather than accidental future conditioning.
""",
        encoding="utf-8",
    )
    return [path]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    written += build_feature_ablation()
    written += build_ttc_sensitivity()
    written += build_target_stability()
    written += build_temporal_diag()
    written += build_context_features()
    written += build_writeup()
    print("[DONE] Appendix outputs")
    for p in written:
        print(" -", p)


if __name__ == "__main__":
    main()
