#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path("/home/patrick/vllmPatrickMonitoring")
PAPER_ANALYSIS = ROOT / "paper_analysis_outputs"
PAPER_FIGS = ROOT / "paper_figures"
PAPER_TABLES = ROOT / "paper_final_tables"

ALFWORLD_SOURCES = {
    "Qwen": PAPER_ANALYSIS / "context_calibrated_prediction_ablation_qwen.csv",
    "Falcon": PAPER_ANALYSIS / "context_calibrated_prediction_ablation_falcon.csv",
}
WEBSHOP_SOURCE = PAPER_ANALYSIS / "context_calibrated_prediction_ablation_webshop.csv"

OUT_PNG = PAPER_FIGS / "main_alfworld_prediction_ablation.png"
OUT_PDF = PAPER_FIGS / "main_alfworld_prediction_ablation.pdf"
WEBSHOP_CSV = PAPER_TABLES / "table_webshop_prediction.csv"
WEBSHOP_TEX = PAPER_TABLES / "table_webshop_prediction.tex"

FEATURE_ORDER = [
    "p_hack_only",
    "entropy_only",
    "temporal_internal",
    "context_only",
    "p_hack_plus_entropy",
    "internal_all",
    "internal_plus_context",
]

FEATURE_LABELS = {
    "p_hack_only": "act",
    "entropy_only": "ent",
    "temporal_internal": "temp",
    "context_only": "ctx",
    "p_hack_plus_entropy": "act+ent",
    "internal_all": "int-all",
    "internal_plus_context": "int+ctx",
}

TARGET_ORDER = ["bad_action_t1", "exploit_action_t1"]
TARGET_LABELS = {
    "bad_action_t1": "Bad action",
    "exploit_action_t1": "Exploit action",
}

COLORS = {
    "p_hack_only": "#4C78A8",
    "entropy_only": "#F58518",
    "temporal_internal": "#54A24B",
    "context_only": "#B279A2",
    "p_hack_plus_entropy": "#E45756",
    "internal_all": "#72B7B2",
    "internal_plus_context": "#2E4057",
}

MIN_POSITIVES = 10


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def filter_stable(rows: list[dict], family: str, target: str) -> list[dict]:
    sub = [r for r in rows if r.get("target") == target and r.get("feature_group") in FEATURE_ORDER]
    if not sub:
        return []
    positives = max((to_float(r.get("positives")) or 0.0) for r in sub)
    negatives = max((to_float(r.get("N")) or 0.0) - (to_float(r.get("positives")) or 0.0) for r in sub)
    if positives < MIN_POSITIVES or negatives < MIN_POSITIVES:
        return []
    keyed = {r["feature_group"]: r for r in sub}
    return [keyed[fg] for fg in FEATURE_ORDER if fg in keyed]


def render_alfworld_figure() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    rows_by_family = {family: read_csv(path) for family, path in ALFWORLD_SOURCES.items()}

    fig, axes = plt.subplots(
        nrows=len(ALFWORLD_SOURCES),
        ncols=len(TARGET_ORDER),
        figsize=(13.5, 6.8),
        sharey="col",
        constrained_layout=True,
    )
    if len(ALFWORLD_SOURCES) == 1:
        axes = np.array([axes])

    for row_idx, family in enumerate(ALFWORLD_SOURCES):
        for col_idx, target in enumerate(TARGET_ORDER):
            ax = axes[row_idx, col_idx]
            stable = filter_stable(rows_by_family[family], family, target)
            if not stable:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    "insufficient data",
                    ha="center",
                    va="center",
                    fontsize=10,
                    transform=ax.transAxes,
                )
                continue

            x = np.arange(len(stable))
            vals = [to_float(r.get("AUPRC_gain")) or 0.0 for r in stable]
            colors = [COLORS[r["feature_group"]] for r in stable]
            bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
            ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle="--")
            ax.set_xticks(x)
            ax.set_xticklabels([FEATURE_LABELS[r["feature_group"]] for r in stable], rotation=35, ha="right")
            if row_idx == len(ALFWORLD_SOURCES) - 1:
                ax.set_xlabel("Feature group")
            if col_idx == 0:
                ax.set_ylabel(f"{family}\nAUPRC gain")
            positives = int(to_float(stable[0].get("positives")) or 0)
            n_total = int(to_float(stable[0].get("N")) or 0)
            ax.set_title(f"{TARGET_LABELS[target]}\n{family}  pos={positives} / N={n_total}", fontsize=11)
            max_abs = max(abs(v) for v in vals) if vals else 0.1
            pad = max(0.015, max_abs * 0.08)
            for bar, value in zip(bars, vals):
                y = value + pad if value >= 0 else value - pad
                va = "bottom" if value >= 0 else "top"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    f"{value:.3f}",
                    ha="center",
                    va=va,
                    fontsize=8,
                )

    fig.suptitle("Gameable ALFWorld next-step prediction ablations", fontsize=14, y=1.02)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def write_webshop_table() -> None:
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    rows = read_csv(WEBSHOP_SOURCE)
    fieldnames = ["family", "target", "feature_group", "AUROC", "AUPRC", "AUPRC_gain", "Recall@20", "positives"]
    with WEBSHOP_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row.get("family", ""),
                row.get("target", ""),
                row.get("feature_group", ""),
                fmt(row.get("AUROC")),
                fmt(row.get("AUPRC")),
                fmt(row.get("AUPRC_gain")),
                fmt(row.get("Recall@20")),
                row.get("positives", ""),
            ]
        )

    tex = []
    tex.append("\\begin{tabular}{lllrrrrr}")
    tex.append("\\toprule")
    tex.append("Family & Target & Feature group & AUROC & AUPRC & AUPRC gain & Recall@20 & Positives \\\\")
    tex.append("\\midrule")
    for cells in table_rows:
        tex.append(" & ".join(cells) + " \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    WEBSHOP_TEX.write_text("\n".join(tex) + "\n", encoding="utf-8")


def fmt(value: str | None) -> str:
    num = to_float(value)
    if num is None:
        return ""
    return f"{num:.3f}"


def main() -> None:
    write_webshop_table()
    print(f"[DONE] Wrote {WEBSHOP_CSV}")
    print(f"[DONE] Wrote {WEBSHOP_TEX}")
    try:
        render_alfworld_figure()
    except ModuleNotFoundError as exc:
        print(f"[WARN] Could not render ALFWorld figure: {exc}")
        print("[WARN] Install matplotlib and numpy in the active environment, then rerun this script.")
        return
    print(f"[DONE] Wrote {OUT_PNG}")
    print(f"[DONE] Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
