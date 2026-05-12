#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# =============================================================================
# Style / constants
# =============================================================================

ADAPTER_ORDER = ["control", "mix05", "mix10", "mix50", "hack"]

ADAPTER_LABELS = {
    "control": "Control",
    "mix05": "Mix05",
    "mix10": "Mix10",
    "mix50": "Mix50",
    "hack": "Hack",
}

ADAPTER_COLORS = {
    "control": "#4C78A8",
    "mix05": "#54A24B",
    "mix10": "#F58518",
    "mix50": "#E45756",
    "hack": "#B279A2",
}

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
    "p_hack_only": "act\nonly",
    "entropy_only": "entropy\nonly",
    "temporal_internal": "temporal\ninternal",
    "context_only": "context\nonly",
    "p_hack_plus_entropy": "act+\nentropy",
    "internal_all": "internal\nall",
    "internal_plus_context": "int+\ncontext",
}

FEATURE_COLORS = {
    "p_hack_only": "#D62728",
    "entropy_only": "#1F77B4",
    "temporal_internal": "#2CA02C",
    "context_only": "#8C8C8C",
    "p_hack_plus_entropy": "#9467BD",
    "internal_all": "#8C564B",
    "internal_plus_context": "#E377C2",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "grid.color": "#9AA0A6",
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, outstem: Path) -> None:
    outstem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outstem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outstem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    print(f"[saved] {outstem.with_suffix('.pdf')}")
    print(f"[saved] {outstem.with_suffix('.png')}")


def clean_ax(ax, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =============================================================================
# Loading / standardization
# =============================================================================

def read_csv_if_exists(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize_adapter(s: str) -> str:
    x = str(s).lower()
    if "mix05" in x or "mix5" in x:
        return "mix05"
    if "mix10" in x:
        return "mix10"
    if "mix50" in x:
        return "mix50"
    if "control" in x:
        return "control"
    if "hack" in x:
        return "hack"
    return x


def standardize_behavior(df: pd.DataFrame, family: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["family"] = family

    if "adapter" in out.columns:
        out["adapter"] = out["adapter"].map(normalize_adapter)
    else:
        raise RuntimeError(f"Behavior file for {family} needs an adapter column.")

    # Ensure expected metrics exist.
    for col in [
        "bad_action_rate",
        "exploit_action_rate",
        "gameable_hack_action_rate",
        "easy_grader_action_rate",
    ]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out


def load_behavior(qwen: Path, llama: Path | None, falcon: Path | None) -> pd.DataFrame:
    parts = [
        standardize_behavior(read_csv_if_exists(qwen), "Qwen"),
        standardize_behavior(read_csv_if_exists(llama), "Llama"),
        standardize_behavior(read_csv_if_exists(falcon), "Falcon"),
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        raise RuntimeError("No behavior files loaded.")
    return pd.concat(parts, ignore_index=True)


def load_risk_state(path: Path) -> pd.DataFrame:
    df = read_csv_if_exists(path)
    if df.empty:
        raise RuntimeError("No risk-state file loaded.")

    if "adapter" in df.columns:
        df["adapter"] = df["adapter"].map(normalize_adapter)
    else:
        raise RuntimeError("Risk-state file needs an adapter column.")

    required = ["mean_p_hack", "mean_entropy", "behavior_risk_rate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Risk-state file missing columns: {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "family" not in df.columns:
        df["family"] = "Qwen"

    return df.dropna(subset=required)


def load_prediction(path: Path) -> pd.DataFrame:
    df = read_csv_if_exists(path)
    if df.empty:
        raise RuntimeError("No prediction-ablation file loaded.")

    required = ["target", "feature_group", "AUPRC_gain"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Prediction file missing columns: {missing}")

    df["AUPRC_gain"] = pd.to_numeric(df["AUPRC_gain"], errors="coerce")
    return df


def load_steering(path: Path) -> pd.DataFrame:
    df = read_csv_if_exists(path)
    if df.empty:
        raise RuntimeError("No steering file loaded.")

    if "adapter" in df.columns:
        df["adapter"] = df["adapter"].map(normalize_adapter)
    else:
        raise RuntimeError("Steering file needs an adapter column.")

    for c in ["delta_proxy", "steering_rate"]:
        if c not in df.columns:
            raise RuntimeError(f"Steering file missing required column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["proxy_reduction"] = -df["delta_proxy"]
    return df.dropna(subset=["proxy_reduction", "steering_rate"])


# =============================================================================
# Figure 1: behavior transfer + risk-state map
# =============================================================================

def plot_behavior_panel(ax, data: pd.DataFrame, family: str, show_xlabel: bool) -> None:
    sub = data[data["family"] == family].copy()

    metrics = [
        ("bad_action_rate", "bad action"),
        ("exploit_action_rate", "exploit action"),
        ("gameable_hack_action_rate", "gameable hack"),
        ("easy_grader_action_rate", "easy grader"),
    ]

    x = np.arange(len(metrics))
    width = 0.18
    adapters = ["control", "mix05", "mix10", "hack"]

    for j, adapter in enumerate(adapters):
        row = sub[sub["adapter"] == adapter]
        vals = []
        for col, _ in metrics:
            vals.append(float(row[col].mean()) if not row.empty else 0.0)

        ax.bar(
            x + (j - 1.5) * width,
            vals,
            width=width,
            color=ADAPTER_COLORS[adapter],
            alpha=0.92,
        )

    ax.set_title(family, fontweight="bold", pad=4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_xticks(x)
    if show_xlabel:
        ax.set_xticklabels([m[1] for m in metrics], rotation=0)
    else:
        ax.set_xticklabels([])
    clean_ax(ax, "y")


def plot_figure_behavior_and_risk(
    behavior: pd.DataFrame,
    risk: pd.DataFrame,
    outstem: Path,
) -> None:
    fig = plt.figure(figsize=(10.8, 6.8), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=4,
        width_ratios=[1.12, 0.06, 1.0, 1.0],
        height_ratios=[1, 1, 1],
        wspace=0.28,
        hspace=0.48,
    )

    fig.text(0.015, 0.975, "A", fontsize=16, fontweight="bold", va="top")
    fig.text(
        0.075,
        0.965,
        "Behavioral transfer across model families",
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    fig.text(0.52, 0.975, "B", fontsize=16, fontweight="bold", va="top")
    fig.text(
        0.57,
        0.965,
        "Structured internal risk states (Qwen)",
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    # Panel A
    families = ["Qwen", "Llama", "Falcon"]
    axes_behavior = []
    for i, fam in enumerate(families):
        ax = fig.add_subplot(gs[i, 0])
        axes_behavior.append(ax)
        plot_behavior_panel(ax, behavior, fam, show_xlabel=(i == len(families) - 1))

    behavior_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor=ADAPTER_COLORS[a],
            markeredgecolor="none",
            markersize=7,
            label=ADAPTER_LABELS[a],
        )
        for a in ["control", "mix05", "mix10", "hack"]
    ]
    axes_behavior[-1].legend(
        handles=behavior_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.65),
        ncol=4,
        frameon=True,
        borderpad=0.5,
    )

    # Divider
    sep = fig.add_subplot(gs[:, 1])
    sep.axis("off")
    sep.plot([0.5, 0.5], [0.05, 0.95], color="#cccccc", lw=1.0, transform=sep.transAxes)

    # Panel B
    qwen = risk[risk["family"].astype(str).str.lower() == "qwen"].copy()
    if qwen.empty:
        qwen = risk.copy()

    ax1 = fig.add_subplot(gs[0:2, 2])
    ax2 = fig.add_subplot(gs[0:2, 3], sharey=ax1)

    for ax, xcol, title, xlabel in [
        (ax1, "mean_p_hack", "Activation vs behavior", r"mean $p_{\mathrm{hack}}$"),
        (ax2, "mean_entropy", "Entropy vs behavior", "mean entropy"),
    ]:
        for adapter in ["control", "mix05", "mix10", "hack"]:
            sub = qwen[qwen["adapter"] == adapter]
            if sub.empty:
                continue
            ax.scatter(
                sub[xcol],
                sub["behavior_risk_rate"],
                s=46,
                color=ADAPTER_COLORS[adapter],
                edgecolor="black",
                linewidth=0.6,
                alpha=0.88,
                label=ADAPTER_LABELS[adapter],
            )

        ax.set_title(title, pad=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("exploit / bad-action rate")
        ax.set_ylim(-0.01, max(0.25, qwen["behavior_risk_rate"].max() * 1.15))
        clean_ax(ax, "both")

    risk_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=ADAPTER_COLORS[a],
            markeredgecolor="black",
            markersize=6.5,
            label=ADAPTER_LABELS[a],
        )
        for a in ["control", "mix05", "mix10", "hack"]
        if a in set(qwen["adapter"])
    ]

    ax_legend = fig.add_subplot(gs[2, 2:4])
    ax_legend.axis("off")
    ax_legend.legend(
        handles=risk_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=True,
        borderpad=0.5,
    )
    ax_legend.text(
        0.5,
        0.35,
        "High activation marks a latent policy state,\nnot a monotonic action-risk score.",
        ha="center",
        va="center",
        fontsize=9,
        color="#1f3b68",
        fontweight="bold",
        transform=ax_legend.transAxes,
    )

    save_figure(fig, outstem)
    plt.close(fig)


# =============================================================================
# Figure 2: prediction ablation + steering
# =============================================================================

def choose_prediction_targets(pred: pd.DataFrame) -> list[str]:
    preferred = [
        "bad_action_t1",
        "exploit_action_t1",
        "bad_buy_action_t1",
        "low_reward_buy_t1",
    ]
    found = [t for t in preferred if t in set(pred["target"])]
    if len(found) >= 2:
        return found[:2]
    return found


def plot_prediction_panel(ax, pred: pd.DataFrame, target: str) -> None:
    sub = pred[pred["target"] == target].copy()

    x = np.arange(len(FEATURE_ORDER))
    vals = []
    colors = []

    for g in FEATURE_ORDER:
        row = sub[sub["feature_group"] == g]
        vals.append(float(row.iloc[0]["AUPRC_gain"]) if not row.empty else np.nan)
        colors.append(FEATURE_COLORS[g])

    ax.bar(x, vals, color=colors, alpha=0.92)
    ax.axhline(0, color="#777777", linestyle="--", lw=1.0)

    for xi, v in zip(x, vals):
        if np.isfinite(v):
            ax.text(
                xi,
                v + (0.005 if v >= 0 else -0.008),
                f"{v:.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=7.5,
            )

    ax.set_title(target.replace("_t1", r"$_{t+1}$").replace("_", r"\_"))
    ax.set_ylabel("AUPRC gain over base rate")
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[g] for g in FEATURE_ORDER], rotation=30, ha="right")
    clean_ax(ax, "y")


def plot_prediction_and_steering(
    pred: pd.DataFrame,
    steering: pd.DataFrame,
    outstem: Path,
) -> None:
    targets = choose_prediction_targets(pred)
    if not targets:
        raise RuntimeError("No supported prediction targets found.")

    fig = plt.figure(figsize=(10.8, 6.8), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[1.0, 1.15],
        width_ratios=[1.0, 1.0, 0.55],
        hspace=0.55,
        wspace=0.34,
    )

    fig.text(0.015, 0.975, "A", fontsize=16, fontweight="bold", va="top")
    fig.text(
        0.18,
        0.965,
        "Context-calibrated features improve next-step risk estimation",
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    ax_pred1 = fig.add_subplot(gs[0, 0])
    plot_prediction_panel(ax_pred1, pred, targets[0])

    if len(targets) > 1:
        ax_pred2 = fig.add_subplot(gs[0, 1], sharey=ax_pred1)
        plot_prediction_panel(ax_pred2, pred, targets[1])
    else:
        fig.add_subplot(gs[0, 1]).axis("off")

    # Feature legend
    ax_feat_leg = fig.add_subplot(gs[0, 2])
    ax_feat_leg.axis("off")
    ax_feat_leg.text(0.0, 0.98, "Feature groups", fontweight="bold", fontsize=9, transform=ax_feat_leg.transAxes)

    y = 0.86
    for g in FEATURE_ORDER:
        ax_feat_leg.add_patch(
            plt.Rectangle(
                (0.0, y - 0.025),
                0.08,
                0.045,
                color=FEATURE_COLORS[g],
                transform=ax_feat_leg.transAxes,
            )
        )
        ax_feat_leg.text(
            0.11,
            y,
            FEATURE_LABELS[g].replace("\n", " "),
            va="center",
            fontsize=8,
            transform=ax_feat_leg.transAxes,
        )
        y -= 0.105

    # Steering panel
    fig.text(0.015, 0.47, "B", fontsize=16, fontweight="bold", va="top")

    ax_s = fig.add_subplot(gs[1, 0:2])
    ax_leg = fig.add_subplot(gs[1, 2])
    ax_leg.axis("off")

    steering = steering.copy()
    steering["delta_proxy"] = pd.to_numeric(steering["delta_proxy"], errors="coerce")
    steering["steering_rate"] = pd.to_numeric(steering["steering_rate"], errors="coerce")
    steering = steering.dropna(subset=["delta_proxy", "steering_rate"])
    steering["proxy_reduction"] = -steering["delta_proxy"]

    for _, row in steering.iterrows():
        adapter = str(row["adapter"]).lower()
        cond = str(row.get("condition", ""))
        mode = "gated" if "gated" in cond else "always"
        marker = "s" if mode == "gated" else "o"

        ax_s.scatter(
            row["steering_rate"],
            row["proxy_reduction"],
            s=80,
            color=ADAPTER_COLORS.get(adapter, "#999999"),
            edgecolor="black",
            linewidth=0.7,
            marker=marker,
            alpha=0.9,
        )

        if row["proxy_reduction"] > 0.6:
            ax_s.text(
                row["steering_rate"],
                row["proxy_reduction"],
                f" {ADAPTER_LABELS.get(adapter, adapter)}",
                fontsize=8,
                va="center",
            )

    ax_s.axhline(0, color="#777777", linestyle="--", lw=1.0)
    ax_s.set_xlabel("Fraction of steps steered")
    ax_s.set_ylabel("Reduction in proxy score\nwithout true success")
    ax_s.set_title("Steering reduces proxy behavior in selected regimes", fontweight="bold")
    clean_ax(ax_s, "both")

    # Steering legend
    ax_leg.text(0.0, 0.95, "Adapter", fontweight="bold", fontsize=9, transform=ax_leg.transAxes)
    y = 0.82
    for a in ["mix05", "mix10", "mix50"]:
        if a not in set(steering["adapter"]):
            continue
        ax_leg.scatter(0.05, y, s=55, color=ADAPTER_COLORS[a], edgecolor="black", transform=ax_leg.transAxes)
        ax_leg.text(0.16, y, ADAPTER_LABELS[a], va="center", fontsize=8, transform=ax_leg.transAxes)
        y -= 0.12

    ax_leg.text(0.0, y - 0.02, "Mode", fontweight="bold", fontsize=9, transform=ax_leg.transAxes)
    ax_leg.scatter(0.05, y - 0.14, s=55, marker="o", color="black", transform=ax_leg.transAxes)
    ax_leg.text(0.16, y - 0.14, "Always-on", va="center", fontsize=8, transform=ax_leg.transAxes)
    ax_leg.scatter(0.05, y - 0.26, s=55, marker="s", color="black", transform=ax_leg.transAxes)
    ax_leg.text(0.16, y - 0.26, "Gated", va="center", fontsize=8, transform=ax_leg.transAxes)

    ax_leg.text(
        0.0,
        0.05,
        "Intervention effects are\nselective, not universal.",
        fontsize=8.5,
        color="#1f4b8f",
        fontstyle="italic",
        fontweight="bold",
        transform=ax_leg.transAxes,
    )

    save_figure(fig, outstem)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-qwen", type=Path, required=True)
    parser.add_argument("--behavior-llama", type=Path, required=False)
    parser.add_argument("--behavior-falcon", type=Path, required=False)
    parser.add_argument("--risk-qwen", type=Path, required=True)
    parser.add_argument("--prediction-qwen", type=Path, required=True)
    parser.add_argument("--steering-qwen", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("paper_figures"))
    args = parser.parse_args()

    set_style()

    behavior = load_behavior(args.behavior_qwen, args.behavior_llama, args.behavior_falcon)
    risk = load_risk_state(args.risk_qwen)
    prediction = load_prediction(args.prediction_qwen)
    steering = load_steering(args.steering_qwen)

    plot_figure_behavior_and_risk(
        behavior=behavior,
        risk=risk,
        outstem=args.outdir / "figure1_behavior_and_risk_state",
    )

    plot_prediction_and_steering(
        pred=prediction,
        steering=steering,
        outstem=args.outdir / "figure2_prediction_and_steering",
    )


if __name__ == "__main__":
    main()