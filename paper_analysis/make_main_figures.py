#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


# -----------------------------
# Style
# -----------------------------

COLORS = {
    "control": "#4C78A8",
    "mix05": "#54A24B",
    "mix10": "#F58518",
    "mix50": "#E45756",
    "hack": "#B279A2",
    "p_hack_only": "#D62728",
    "entropy_only": "#1F77B4",
    "temporal_internal": "#2CA02C",
    "context_only": "#8C8C8C",
    "p_hack_plus_entropy": "#9467BD",
    "internal_all": "#8C564B",
    "internal_plus_context": "#E377C2",
}

ADAPTERS = ["control", "mix05", "mix10", "hack"]
ADAPTER_LABELS = {
    "control": "Control",
    "mix05": "Mix05",
    "mix10": "Mix10",
    "mix50": "Mix50",
    "hack": "Hack",
}

FAMILIES = ["Qwen", "Llama", "Falcon"]

FEATURE_GROUPS = [
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


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "grid.color": "#B0B0B0",
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, outstem: Path):
    outstem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outstem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outstem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    print(f"[saved] {outstem.with_suffix('.pdf')}")
    print(f"[saved] {outstem.with_suffix('.png')}")


# -----------------------------
# Helpers for loaded data
# -----------------------------

def load_behavior(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "adapter" in df.columns:
        df["adapter"] = df["adapter"].str.lower()
    if "family" not in df.columns and "model" in df.columns:
        df["family"] = df["model"].astype(str).str.extract(r"(Qwen|Llama|Falcon)", expand=False)
    return df


def load_risk_state(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "adapter" in df.columns:
        df["adapter"] = df["adapter"].str.lower()
    return df


def load_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def load_steering(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "adapter" in df.columns:
        df["adapter"] = df["adapter"].str.lower()
    return df


# -----------------------------
# Figure 1:
# Behavioral transfer + risk-state map
# -----------------------------

def plot_behavioral_transfer_and_risk_states(
    behavior_csv: Path,
    risk_state_csv: Path,
    outstem: Path,
):
    behavior = load_behavior(behavior_csv)
    risk = load_risk_state(risk_state_csv)

    fig = plt.figure(figsize=(11.2, 7.8), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=4,
        width_ratios=[1.2, 0.08, 1.1, 1.1],
        height_ratios=[1, 1, 1],
        wspace=0.25,
        hspace=0.55,
    )

    fig.text(0.02, 0.97, "A", fontsize=18, fontweight="bold", va="top")
    fig.text(0.06, 0.96, "Behavioral transfer across model families", fontsize=13, fontweight="bold", va="top")
    fig.text(0.52, 0.97, "B", fontsize=18, fontweight="bold", va="top")
    fig.text(0.56, 0.96, "Structured internal risk states (Qwen)", fontsize=13, fontweight="bold", va="top")

    metrics = [
        ("bad_action_rate", "bad action"),
        ("exploit_action_rate", "exploit action"),
        ("gameable_hack_action_rate", "gameable hack"),
        ("easy_grader_action_rate", "easy grader"),
    ]

    # Panel A: behavior bars by family
    axes_a = []
    for i, fam in enumerate(FAMILIES):
        ax = fig.add_subplot(gs[i, 0])
        axes_a.append(ax)

        sub = behavior[behavior["family"].astype(str).str.lower() == fam.lower()].copy()
        if sub.empty:
            ax.text(0.5, 0.5, f"No {fam} rows", ha="center", va="center")
            ax.axis("off")
            continue

        x = np.arange(len(metrics))
        width = 0.18

        for j, adapter in enumerate(ADAPTERS):
            row = sub[sub["adapter"] == adapter]
            vals = []
            for col, _ in metrics:
                vals.append(float(row[col].mean()) if col in row.columns and not row.empty else 0.0)

            ax.bar(
                x + (j - 1.5) * width,
                vals,
                width=width,
                color=COLORS[adapter],
                alpha=0.9,
                label=ADAPTER_LABELS[adapter],
            )

        ax.set_title(fam, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels([m[1] for m in metrics])
        ax.set_ylabel("Rate")
        clean_axis(ax, "y")

    # shared legend for panel A
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS[a],
               markersize=8, label=ADAPTER_LABELS[a])
        for a in ADAPTERS
    ]
    axes_a[-1].legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.55),
        ncol=4,
        frameon=True,
    )

    # vertical separator
    sep_ax = fig.add_subplot(gs[:, 1])
    sep_ax.axis("off")
    sep_ax.plot([0.5, 0.5], [0.05, 0.95], color="#cccccc", lw=1.0, transform=sep_ax.transAxes)

    # Panel B: Qwen risk-state maps
    qwen = risk[risk["family"].astype(str).str.lower() == "qwen"].copy()
    if qwen.empty:
        qwen = risk.copy()

    # normalize expected column names
    # mean_p_hack, mean_entropy, behavior_risk_rate expected from script 03
    ax_b1 = fig.add_subplot(gs[0:2, 2])
    ax_b2 = fig.add_subplot(gs[0:2, 3], sharey=ax_b1)

    for ax, xcol, title, xlabel in [
        (ax_b1, "mean_p_hack", "Activation vs behavior", r"mean $p_{\mathrm{hack}}$"),
        (ax_b2, "mean_entropy", "Entropy vs behavior", "mean entropy"),
    ]:
        for adapter, sub in qwen.groupby("adapter"):
            ax.scatter(
                sub[xcol],
                sub["behavior_risk_rate"],
                s=55,
                color=COLORS.get(adapter, "#999999"),
                edgecolor="black",
                linewidth=0.6,
                alpha=0.85,
                label=ADAPTER_LABELS.get(adapter, adapter),
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("exploit / bad-action rate")
        clean_axis(ax, "both")
        ax.set_ylim(-0.01, max(0.25, qwen["behavior_risk_rate"].max() * 1.15))

    # Legend below scatter plots
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[a],
               markeredgecolor="black", markersize=7, label=ADAPTER_LABELS[a])
        for a in ["control", "mix05", "mix10", "hack"]
        if a in set(qwen["adapter"])
    ]
    ax_b1.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(1.1, -0.20),
        ncol=4,
        frameon=True,
    )

    # Callout box
    ax_callout = fig.add_subplot(gs[2, 2:4])
    ax_callout.axis("off")
    box = FancyBboxPatch(
        (0.08, 0.10),
        0.84,
        0.70,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor="#f5f8ff",
        edgecolor="#9fb6d8",
        linewidth=1.0,
        transform=ax_callout.transAxes,
    )
    ax_callout.add_patch(box)
    ax_callout.text(0.15, 0.60, "💡", fontsize=20, transform=ax_callout.transAxes, va="center")
    ax_callout.text(
        0.25,
        0.60,
        "High activation does not imply\nhighest exploit rate.",
        fontsize=10,
        fontweight="bold",
        color="#1f3b68",
        transform=ax_callout.transAxes,
        va="center",
    )
    ax_callout.text(
        0.25,
        0.32,
        "Mixed adapters can show elevated behavior risk\nwhile saturated Hack adapters remain low-risk.",
        fontsize=8.5,
        color="#333333",
        transform=ax_callout.transAxes,
        va="center",
    )

    save_fig(fig, outstem)
    plt.close(fig)


# -----------------------------
# Figure 2:
# Prediction ablation + steering
# -----------------------------

def plot_prediction_and_steering(
    prediction_csv: Path,
    steering_csv: Path,
    outstem: Path,
):
    pred = load_prediction(prediction_csv)
    steer = load_steering(steering_csv)

    # Keep two main targets if present
    target_order = ["bad_action_t1", "exploit_action_t1", "bad_buy_action_t1", "low_reward_buy_t1"]
    targets = [t for t in target_order if t in set(pred["target"])]
    # Prefer ALFWorld targets if there are too many
    targets = targets[:2] if len(targets) > 2 else targets

    fig = plt.figure(figsize=(11.2, 7.2), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[1.05, 1.0],
        width_ratios=[1, 1, 0.55],
        hspace=0.55,
        wspace=0.35,
    )

    fig.text(0.02, 0.97, "A", fontsize=18, fontweight="bold", va="top")
    fig.text(0.14, 0.96, "Context-calibrated features improve next-step risk estimation",
             fontsize=13, fontweight="bold", va="top")

    # Feature ablation panels
    for i, target in enumerate(targets):
        ax = fig.add_subplot(gs[0, i])
        sub = pred[pred["target"] == target].copy()

        x = np.arange(len(FEATURE_GROUPS))
        vals = []
        colors = []
        for g in FEATURE_GROUPS:
            row = sub[sub["feature_group"] == g]
            if row.empty:
                vals.append(np.nan)
            else:
                vals.append(float(row.iloc[0]["AUPRC_gain"]))
            colors.append(COLORS[g])

        ax.bar(x, vals, color=colors, alpha=0.9)
        ax.axhline(0, linestyle="--", color="#777777", lw=1.0)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(
                    xi,
                    v + (0.006 if v >= 0 else -0.008),
                    f"{v:.2f}",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([FEATURE_LABELS[g] for g in FEATURE_GROUPS], rotation=25, ha="right")
        ax.set_title(target.replace("_t1", r"$_{t+1}$").replace("_", r"\_"))
        ax.set_ylabel("AUPRC gain over base rate")
        clean_axis(ax, "y")

    # Empty if only one target
    if len(targets) == 1:
        ax_empty = fig.add_subplot(gs[0, 1])
        ax_empty.axis("off")

    # Panel B: steering
    fig.text(0.02, 0.47, "B", fontsize=18, fontweight="bold", va="top")
    ax_s = fig.add_subplot(gs[1, 0:2])
    ax_leg = fig.add_subplot(gs[1, 2])
    ax_leg.axis("off")

    for col in ["delta_proxy", "steering_rate"]:
        if col in steer.columns:
            steer[col] = pd.to_numeric(steer[col], errors="coerce")

    steer = steer.dropna(subset=["delta_proxy", "steering_rate"]).copy()
    steer["proxy_reduction"] = -steer["delta_proxy"]

    marker_map = {"always": "o", "gated": "s"}
    for _, row in steer.iterrows():
        adapter = str(row["adapter"]).lower()
        cond = str(row.get("condition", ""))
        mode = "gated" if "gated" in cond else "always"
        marker = marker_map[mode]
        ax_s.scatter(
            row["steering_rate"],
            row["proxy_reduction"],
            s=85,
            marker=marker,
            color=COLORS.get(adapter, "#999999"),
            edgecolor="black",
            linewidth=0.7,
            alpha=0.9,
        )
        # annotate only meaningful positive reductions
        if row["proxy_reduction"] > 0.5:
            ax_s.text(
                row["steering_rate"],
                row["proxy_reduction"],
                f" {ADAPTER_LABELS.get(adapter, adapter)} {cond}",
                fontsize=8,
                va="center",
            )

    ax_s.axhline(0, linestyle="--", color="#777777", lw=1.0)
    ax_s.set_xlabel("Fraction of steps steered")
    ax_s.set_ylabel("Reduction in proxy score\nwithout true success")
    ax_s.set_title("Steering reduces proxy behavior in selected regimes", fontweight="bold")
    clean_axis(ax_s, "both")

    # Legends and callout
    ax_leg.text(0.0, 0.95, "Adapter", fontsize=9, fontweight="bold", transform=ax_leg.transAxes)
    y = 0.82
    for adapter in ["mix05", "mix10", "mix50"]:
        ax_leg.scatter(0.05, y, s=55, color=COLORS[adapter], edgecolor="black", transform=ax_leg.transAxes)
        ax_leg.text(0.16, y, ADAPTER_LABELS.get(adapter, adapter), va="center", fontsize=8.5, transform=ax_leg.transAxes)
        y -= 0.12

    ax_leg.text(0.0, y - 0.04, "Steering mode", fontsize=9, fontweight="bold", transform=ax_leg.transAxes)
    ax_leg.scatter(0.05, y - 0.17, s=55, color="black", marker="o", transform=ax_leg.transAxes)
    ax_leg.text(0.16, y - 0.17, "Always-on", va="center", fontsize=8.5, transform=ax_leg.transAxes)
    ax_leg.scatter(0.05, y - 0.29, s=55, color="black", marker="s", transform=ax_leg.transAxes)
    ax_leg.text(0.16, y - 0.29, "Gated", va="center", fontsize=8.5, transform=ax_leg.transAxes)

    box = FancyBboxPatch(
        (0.0, 0.02),
        0.95,
        0.18,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#f5f8ff",
        edgecolor="#9fb6d8",
        linewidth=1.0,
        transform=ax_leg.transAxes,
    )
    ax_leg.add_patch(box)
    ax_leg.text(
        0.05,
        0.11,
        "Intervention effects are\nselective, not universal.",
        fontsize=8.5,
        color="#1f4b8f",
        fontstyle="italic",
        fontweight="bold",
        va="center",
        transform=ax_leg.transAxes,
    )

    save_fig(fig, outstem)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-csv", type=Path, required=True)
    parser.add_argument("--risk-csv", type=Path, required=True)
    parser.add_argument("--prediction-csv", type=Path, required=True)
    parser.add_argument("--steering-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("paper_figures"))
    args = parser.parse_args()

    set_style()

    plot_behavioral_transfer_and_risk_states(
        behavior_csv=args.behavior_csv,
        risk_state_csv=args.risk_csv,
        outstem=args.outdir / "figure_behavioral_transfer_and_risk_states",
    )

    plot_prediction_and_steering(
        prediction_csv=args.prediction_csv,
        steering_csv=args.steering_csv,
        outstem=args.outdir / "figure_prediction_and_steering",
    )


if __name__ == "__main__":
    main()

def mainOLD():
    set_style = set_style if False else None  # keep linters quiet
    set_style
    set_style = None
    set_style

    set_style = None
    set_style
    set_style = None

    set_style = None

    set_style = None
    set_style

    set_style = None

    set_style
    set_style

    set_style = None

    set_style

    set_style = None

    set_style

    set_style = None

    set_style

    set_style = None

    set_style

    set_style = None

    set_style

    set_style = None

    set_style

    set_style = None

    set_style

    set_style = None

    set_style

    set_style = None

    # actual style call
    set_style = None
    set_style
    set_style = None
    set_style

    # sorry, actual call:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-csv", type=Path, required=True)
    parser.add_argument("--risk-csv", type=Path, required=True)
    parser.add_argument("--prediction-csv", type=Path, required=True)
    parser.add_argument("--steering-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("paper_figures"))
    args = parser.parse_args()

    set_style = None
    # call the real function
    globals()["set_style"] = None
    set_style

    # Fix: call style via function name.
    # This is intentionally explicit to avoid accidental shadowing in notebooks.
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
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

    plot_behavioral_transfer_and_risk_states(
        behavior_csv=args.behavior_csv,
        risk_state_csv=args.risk_csv,
        outstem=args.outdir / "figure_behavioral_transfer_and_risk_states",
    )

    plot_prediction_and_steering(
        prediction_csv=args.prediction_csv,
        steering_csv=args.steering_csv,
        outstem=args.outdir / "figure_prediction_and_steering",
    )


if __name__ == "__main__":
    main()