#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ADAPTER_ORDER = ["control", "mix05", "mix10", "mix50", "hack"]
ADAPTER_LABEL = {
    "control": "Control",
    "mix05": "Mix05",
    "mix10": "Mix10",
    "mix50": "Mix50",
    "hack": "Hack",
}
COLORS = {
    "control": "#4c78a8",
    "mix05": "#54a24b",
    "mix10": "#f58518",
    "mix50": "#e45756",
    "hack": "#b279a2",
}


def infer_adapter(x: str) -> str:
    s = str(x).lower()
    if "mix05" in s or "mix5" in s:
        return "mix05"
    if "mix10" in s:
        return "mix10"
    if "mix50" in s:
        return "mix50"
    if "control" in s:
        return "control"
    if "hack" in s:
        return "hack"
    return "unknown"


def infer_family(x: str) -> str:
    s = str(x).lower()
    if "qwen" in s:
        return "Qwen"
    if "llama" in s:
        return "Llama"
    if "falcon" in s:
        return "Falcon"
    return "Unknown"


def load_step_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "adapter" not in df.columns:
        df["adapter"] = df["model"].map(infer_adapter)
    else:
        df["adapter"] = df["adapter"].map(infer_adapter)
    if "family" not in df.columns:
        df["family"] = df["model"].map(infer_family)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in [
        "bad_action",
        "gameable_hack_action",
        "easy_grader_action",
        "fake_completion_action",
        "buy_action",
        "bad_buy_action",
        "low_reward_buy",
        "early_buy",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    exploit_cols = [c for c in ["gameable_hack_action", "easy_grader_action", "fake_completion_action"] if c in df.columns]
    if exploit_cols:
        df["exploit_action"] = (df[exploit_cols].sum(axis=1) > 0).astype(float)
    else:
        df["exploit_action"] = np.nan

    return df


def summarize(df: pd.DataFrame, env: str) -> pd.DataFrame:
    df = add_targets(df)
    group_cols = ["family", "adapter"]

    metric_cols = [
        "bad_action",
        "exploit_action",
        "gameable_hack_action",
        "easy_grader_action",
        "fake_completion_action",
        "buy_action",
        "bad_buy_action",
        "low_reward_buy",
        "early_buy",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    rows = []
    for (family, adapter), sub in df.groupby(group_cols):
        row = {
            "environment": env,
            "family": family,
            "adapter": adapter,
            "n_steps": len(sub),
        }
        if "source_file" in sub.columns:
            row["n_episodes"] = sub["source_file"].nunique()
        for c in metric_cols:
            vals = pd.to_numeric(sub[c], errors="coerce")
            if vals.notna().any():
                row[f"{c}_rate"] = vals.mean()
        rows.append(row)

    out = pd.DataFrame(rows)
    out["adapter_order"] = out["adapter"].map({a: i for i, a in enumerate(ADAPTER_ORDER)})
    return out.sort_values(["environment", "family", "adapter_order"]).drop(columns=["adapter_order"])


def plot_behavior(summary: pd.DataFrame, outpath: Path, family: str = "Qwen") -> None:
    sub = summary[summary["family"] == family].copy()
    if sub.empty:
        print(f"[WARN] No rows for family={family}")
        return

    metrics = [c for c in [
        "bad_action_rate",
        "exploit_action_rate",
        "gameable_hack_action_rate",
        "easy_grader_action_rate",
        "bad_buy_action_rate",
        "low_reward_buy_rate",
    ] if c in sub.columns]

    if not metrics:
        print("[WARN] No behavior metrics found to plot.")
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 3.2), sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        plot_df = sub[sub[metric].notna()].copy()
        plot_df["adapter"] = pd.Categorical(plot_df["adapter"], ADAPTER_ORDER, ordered=True)
        plot_df = plot_df.sort_values("adapter")

        ax.bar(
            [ADAPTER_LABEL.get(a, a) for a in plot_df["adapter"]],
            plot_df[metric],
            color=[COLORS.get(a, "#999999") for a in plot_df["adapter"]],
            alpha=0.9,
        )
        ax.set_title(metric.replace("_rate", "").replace("_", " "))
        ax.set_ylim(0, max(0.05, plot_df[metric].max() * 1.25))
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Rate")
    fig.suptitle(f"{family}: agentic action behavior by adapter", y=1.05)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alfworld-steps", type=Path, default=Path("BALROG/analysis_gameable_alfworld_temporal/steps_with_entropy_phack_temporal.csv"))
    ap.add_argument("--webshop-steps", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=Path("paper_analysis_outputs"))
    ap.add_argument("--family", type=str, default="Qwen")
    args = ap.parse_args()

    outputs = []

    if args.alfworld_steps and args.alfworld_steps.exists():
        alf = load_step_csv(args.alfworld_steps)
        outputs.append(summarize(alf, "ALFWorld"))
    else:
        print("[WARN] ALFWorld step file not found.")

    if args.webshop_steps and args.webshop_steps.exists():
        web = load_step_csv(args.webshop_steps)
        outputs.append(summarize(web, "WebShop"))
    elif args.webshop_steps:
        print("[WARN] WebShop step file not found.")

    if not outputs:
        raise RuntimeError("No input data found.")

    summary = pd.concat(outputs, ignore_index=True)
    args.outdir.mkdir(parents=True, exist_ok=True)

    family_slug = args.family.lower()

    csv_path = args.outdir / f"behavior_transfer_summary_{family_slug}.csv"
    png_path = args.outdir / f"behavior_transfer_{family_slug}.png"

    summary.to_csv(csv_path, index=False)
    plot_behavior(summary, png_path, family=args.family)
    print("[DONE]")
    print(" ", csv_path)
    print(" ", png_path)
    print(" ", png_path.with_suffix(".pdf"))

if __name__ == "__main__":
    main()