#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/patrick/vllmPatrickMonitoring")
SRC = ROOT / "tables_final_evidence" / "main_context_calibrated_prediction_ablation.csv"
OUT_TABLE = ROOT / "tables" / "main_context_calibrated_prediction_ablation.csv"
OUT_TEX = ROOT / "tables" / "main_context_calibrated_prediction_ablation.tex"
OUT_PNG = ROOT / "figures" / "main_context_calibrated_prediction_ablation.png"

FEATURE_ORDER = [
    "p_hack_only",
    "entropy_only",
    "p_hack_plus_entropy",
    "context_only",
    "internal_plus_context",
    "internal_all",
]
FEATURE_LABELS = {
    "p_hack_only": "activation\nonly",
    "entropy_only": "entropy\nonly",
    "p_hack_plus_entropy": "activation +\nentropy",
    "context_only": "context\nonly",
    "internal_plus_context": "internal +\ncontext",
    "internal_all": "all\ninternal",
}
FEATURE_COLORS = {
    "p_hack_only": "#d62728",
    "entropy_only": "#1f77b4",
    "p_hack_plus_entropy": "#9467bd",
    "context_only": "#7f7f7f",
    "internal_plus_context": "#e377c2",
    "internal_all": "#8c564b",
}
PANELS = [
    ("alfworld", "bad_action_t1", "ALFWorld bad_action$_{t+1}$"),
    ("alfworld", "exploit_action_t1", "ALFWorld exploit_action$_{t+1}$"),
    ("webshop", "bad_buy_t1", "WebShop bad_buy$_{t+1}$"),
    ("webshop", "low_reward_buy_t1", "WebShop low_reward_buy$_{t+1}$"),
]
METRIC_ROWS = [
    ("AUPRC_gain", "AUPRC gain over base rate", 0.0),
    ("AUROC", "AUROC", 0.5),
    ("Recall_at_20pct", "Recall@20%", None),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def fmt(x):
    try:
        return f"{float(x):.3f}"
    except Exception:
        return ""


def write_tex(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(df.columns)
    with path.open("w") as f:
        f.write("\\begin{tabular}{" + "l" * len(cols) + "}\n")
        f.write(" \\toprule\n")
        f.write(" & ".join(cols) + " \\\\\n")
        f.write(" \\midrule\n")
        for _, row in df.iterrows():
            f.write(" & ".join(fmt(row[c]) if c not in {"environment", "target", "feature_group", "best_regime_ttc"} else str(row[c]) for c in cols) + " \\\\\n")
        f.write(" \\bottomrule\n")
        f.write("\\end{tabular}\n")


def select_best_rows(df: pd.DataFrame) -> pd.DataFrame:
    stable = df[df["stable"] == True].copy()
    rows = []
    for env, target, _ in PANELS:
        sub = stable[(stable["environment"] == env) & (stable["target"] == target)].copy()
        if sub.empty:
            continue
        best_regime = (
            sub[sub["feature_group"] == "internal_plus_context"]
            .sort_values(["auprc_gain", "auroc"], ascending=False)
        )
        if best_regime.empty:
            best_regime = sub.sort_values(["auprc_gain", "auroc"], ascending=False)
        regime = best_regime.iloc[0]["ttc_or_regime"]
        chosen = (
            sub[sub["ttc_or_regime"] == regime]
            .groupby("feature_group", as_index=False)
            .agg(
                positives=("positives", "sum"),
                negatives=("negatives", "sum"),
                base_rate=("base_rate", "mean"),
                AUROC=("auroc", "mean"),
                AUPRC=("auprc", "mean"),
                AUPRC_gain=("auprc_gain", "mean"),
                Recall_at_20pct=("recall_at_20pct", "mean"),
            )
        )
        chosen["environment"] = env
        chosen["target"] = target
        chosen["best_regime_ttc"] = regime
        rows.append(chosen)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def render(best: pd.DataFrame, out_png: Path) -> None:
    configure_style()
    fig, axes = plt.subplots(
        len(METRIC_ROWS),
        len(PANELS),
        figsize=(4.0 * len(PANELS), 3.2 * len(METRIC_ROWS)),
        constrained_layout=True,
        sharex="col",
    )
    if len(METRIC_ROWS) == 1:
        axes = np.array([axes])
    if len(PANELS) == 1:
        axes = axes[:, np.newaxis]
    for c, (env, target, title) in enumerate(PANELS):
        sub = best[(best["environment"] == env) & (best["target"] == target)].copy()
        sub = sub[sub["feature_group"].isin(FEATURE_ORDER)].copy()
        if sub.empty:
            for r in range(len(METRIC_ROWS)):
                axes[r, c].set_axis_off()
            continue
        sub["feature_group"] = pd.Categorical(sub["feature_group"], FEATURE_ORDER, ordered=True)
        sub = sub.sort_values("feature_group")
        x = np.arange(len(sub))
        base = float(sub["base_rate"].iloc[0])
        regime = sub["best_regime_ttc"].iloc[0]
        for r, (metric, ylabel, refline) in enumerate(METRIC_ROWS):
            ax = axes[r, c]
            ax.bar(x, sub[metric], color=[FEATURE_COLORS.get(fg, "#777777") for fg in sub["feature_group"]], width=0.72)
            if refline is not None:
                ax.axhline(refline, color="#777777", linestyle="--", linewidth=1.0)
            ax.grid(axis="y", alpha=0.16)
            if r == 0:
                ax.set_title(f"{title}\nbase={base:.2f}, best={regime}")
                best_idx = sub[metric].astype(float).idxmax()
                best_row = sub.loc[best_idx]
                best_pos = sub.index.get_loc(best_idx)
                ax.annotate(
                    "best",
                    xy=(best_pos, best_row[metric]),
                    xytext=(best_pos, best_row[metric] + 0.03),
                    ha="center",
                    fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": "#333333", "lw": 0.8},
                )
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == len(METRIC_ROWS) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels([FEATURE_LABELS[str(fg)] for fg in sub["feature_group"]])
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=FEATURE_COLORS[fg], label=FEATURE_LABELS[fg].replace("\n", " "))
        for fg in FEATURE_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Context-calibrated internal monitoring improves next-step risk estimation", y=1.05, fontsize=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(SRC)
    best = select_best_rows(df)
    best = best.rename(columns={"feature_group": "feature_group"})
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(OUT_TABLE, index=False)
    write_tex(best, OUT_TEX)
    render(best, OUT_PNG)
    print("[DONE]")
    print(OUT_PNG)
    print(OUT_PNG.with_suffix(".pdf"))
    print(OUT_TABLE)
    print(OUT_TEX)


if __name__ == "__main__":
    main()
