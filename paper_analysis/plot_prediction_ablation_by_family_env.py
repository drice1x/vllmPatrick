#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


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

FAMILY_ORDER = ["Qwen", "Llama", "Falcon"]

TARGET_ORDER = [
    "bad_action_t1",
    "exploit_action_t1",
    "buy_t1",
    "bad_buy_t1",
    "low_reward_buy_t1",
]

TARGET_LABELS = {
    "bad_action_t1": r"ALFWorld bad action$_{t+1}$",
    "exploit_action_t1": r"ALFWorld exploit action$_{t+1}$",
    "buy_t1": r"WebShop buy$_{t+1}$",
    "bad_buy_t1": r"WebShop bad buy$_{t+1}$",
    "low_reward_buy_t1": r"WebShop low-reward buy$_{t+1}$",
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
            "grid.color": "#9AA0A6",
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def infer_family_from_path(path: Path) -> str:
    s = path.name.lower()
    if "qwen" in s:
        return "Qwen"
    if "llama" in s:
        return "Llama"
    if "falcon" in s:
        return "Falcon"
    return "Unknown"


def normalize_target(t: str) -> str:
    t = str(t)
    # normalize common forms
    t = t.replace("bad_buy_action_t1", "bad_buy_t1")
    t = t.replace("low_reward_buy_action_t1", "low_reward_buy_t1")
    t = t.replace("buy_action_t1", "buy_t1")
    return t


def infer_environment(target: str) -> str:
    t = normalize_target(target)
    if "buy" in t:
        return "WebShop"
    return "ALFWorld"


def normalize_feature_group(g: str) -> str:
    g = str(g)
    # robust aliases
    aliases = {
        "activation_only": "p_hack_only",
        "act_only": "p_hack_only",
        "phack_only": "p_hack_only",
        "p_hack": "p_hack_only",
        "entropy": "entropy_only",
        "temporal": "temporal_internal",
        "context": "context_only",
        "act_plus_entropy": "p_hack_plus_entropy",
        "activation_plus_entropy": "p_hack_plus_entropy",
        "internal": "internal_all",
        "all_internal": "internal_all",
        "internal_context": "internal_plus_context",
        "internal+context": "internal_plus_context",
        "int_plus_context": "internal_plus_context",
    }
    return aliases.get(g, g)


def read_prediction_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "family" not in df.columns:
        df["family"] = infer_family_from_path(path)
    if "target" not in df.columns:
        raise ValueError(f"{path} missing 'target' column")
    if "feature_group" not in df.columns:
        raise ValueError(f"{path} missing 'feature_group' column")
    if "AUPRC_gain" not in df.columns:
        raise ValueError(f"{path} missing 'AUPRC_gain' column")

    df["target"] = df["target"].map(normalize_target)
    df["feature_group"] = df["feature_group"].map(normalize_feature_group)
    df["environment"] = df["target"].map(infer_environment)
    df["AUPRC_gain"] = pd.to_numeric(df["AUPRC_gain"], errors="coerce")

    for col in ["AUROC", "AUPRC", "Recall@20", "positives", "N", "base_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_all_prediction(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not p.exists():
            print(f"[WARN] missing: {p}")
            continue
        frames.append(read_prediction_csv(p))

    if not frames:
        raise RuntimeError("No prediction CSVs loaded.")

    df = pd.concat(frames, ignore_index=True)

    # remove duplicate rows if the same CSV was supplied twice
    df = df.drop_duplicates(subset=["family", "target", "feature_group"], keep="first")
    return df


def choose_targets(df: pd.DataFrame, include_targets: list[str] | None) -> list[str]:
    if include_targets:
        targets = [normalize_target(t) for t in include_targets]
        return [t for t in targets if t in set(df["target"])]

    available = set(df["target"])
    targets = [t for t in TARGET_ORDER if t in available]
    # add any extra targets after known ones
    targets += sorted(available - set(targets))
    return targets


def plot_family_env_grid(
    df: pd.DataFrame,
    outstem: Path,
    include_targets: list[str] | None = None,
    families: list[str] | None = None,
    metric: str = "AUPRC_gain",
) -> None:
    targets = choose_targets(df, include_targets)
    if not targets:
        raise RuntimeError("No targets available to plot.")

    if families is None:
        families = [f for f in FAMILY_ORDER if f in set(df["family"])]
        families += sorted(set(df["family"]) - set(families))
    else:
        families = [f for f in families if f in set(df["family"])]

    if not families:
        raise RuntimeError("No families available to plot.")

    n_rows = len(families)
    n_cols = len(targets)

    fig_w = max(10.5, 3.2 * n_cols)
    fig_h = max(3.0, 2.6 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        sharey=True,
        constrained_layout=True,
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    # global y limits
    vals = df[metric].dropna().to_numpy()
    if len(vals) == 0:
        y_min, y_max = -0.05, 0.2
    else:
        y_min = min(-0.02, float(np.nanmin(vals)) - 0.03)
        y_max = max(0.08, float(np.nanmax(vals)) + 0.06)

    for r, family in enumerate(families):
        for c, target in enumerate(targets):
            ax = axes[r, c]
            sub = df[(df["family"] == family) & (df["target"] == target)].copy()

            x = np.arange(len(FEATURE_ORDER))
            heights = []
            colors = []

            for g in FEATURE_ORDER:
                row = sub[sub["feature_group"] == g]
                if row.empty:
                    heights.append(np.nan)
                else:
                    heights.append(float(row.iloc[0][metric]))
                colors.append(FEATURE_COLORS[g])

            ax.bar(x, heights, color=colors, alpha=0.92)
            ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, axis="y")

            if r == 0:
                ax.set_title(TARGET_LABELS.get(target, target.replace("_", r"\_")), pad=6)

            if c == 0:
                ax.set_ylabel(f"{family}\n{metric.replace('_', ' ')}")
            else:
                ax.set_ylabel("")

            ax.set_xticks(x)
            if r == n_rows - 1:
                ax.set_xticklabels([FEATURE_LABELS[g] for g in FEATURE_ORDER], rotation=35, ha="right")
            else:
                ax.set_xticklabels([])

            # add values only if readable
            for xi, h in zip(x, heights):
                if np.isfinite(h):
                    ax.text(
                        xi,
                        h + (0.006 if h >= 0 else -0.008),
                        f"{h:.2f}",
                        ha="center",
                        va="bottom" if h >= 0 else "top",
                        fontsize=7,
                    )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=FEATURE_COLORS[g], markersize=8,
               label=FEATURE_LABELS[g].replace("\n", " "))
        for g in FEATURE_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(len(FEATURE_ORDER), 7),
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )

    fig.suptitle(
        "Context-calibrated features improve next-step risk estimation",
        fontsize=13,
        fontweight="bold",
        y=1.09,
    )

    outstem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outstem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outstem.with_suffix(".png"), dpi=420, bbox_inches="tight")
    print(f"[saved] {outstem.with_suffix('.pdf')}")
    print(f"[saved] {outstem.with_suffix('.png')}")


def plot_compact_by_environment(
    df: pd.DataFrame,
    outstem: Path,
    families: list[str] | None = None,
) -> None:
    """
    Alternative compact figure:
    one panel per environment/target, pooled across families by showing one row per family.
    """
    plot_family_env_grid(
        df,
        outstem=outstem,
        families=families,
        include_targets=None,
        metric="AUPRC_gain",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-csvs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more context_calibrated_prediction_ablation_*.csv files.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("paper_figures"))
    parser.add_argument("--families", type=str, default="Qwen,Llama,Falcon")
    parser.add_argument(
        "--targets",
        type=str,
        default="",
        help="Optional comma-separated targets, e.g. bad_action_t1,exploit_action_t1,bad_buy_t1,low_reward_buy_t1",
    )
    args = parser.parse_args()

    set_style()

    df = load_all_prediction(args.prediction_csvs)

    families = [x.strip() for x in args.families.split(",") if x.strip()]
    targets = [x.strip() for x in args.targets.split(",") if x.strip()] if args.targets.strip() else None

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Save combined table for inspection.
    combined_csv = args.outdir / "combined_prediction_ablation_all_families.csv"
    df.to_csv(combined_csv, index=False)
    print(f"[saved] {combined_csv}")

    plot_family_env_grid(
        df,
        outstem=args.outdir / "prediction_ablation_by_family_environment",
        families=families,
        include_targets=targets,
        metric="AUPRC_gain",
    )


if __name__ == "__main__":
    main()