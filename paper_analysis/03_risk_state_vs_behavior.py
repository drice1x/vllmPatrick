#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


COLORS = {
    "control": "#4c78a8",
    "mix05": "#54a24b",
    "mix10": "#f58518",
    "mix50": "#e45756",
    "hack": "#b279a2",
}


def infer_adapter(x):
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


def infer_family(x):
    s = str(x).lower()
    if "qwen" in s:
        return "Qwen"
    if "llama" in s:
        return "Llama"
    if "falcon" in s:
        return "Falcon"
    return "Unknown"


def normalize_ttc(x):
    s = str(x)
    if "react_baseline" in s:
        return "react_baseline"
    m = re.search(r"rtok(\d+)", s)
    if m:
        return f"ttc_{m.group(1)}"
    m = re.search(r"(\d+)", s)
    if m:
        return f"ttc_{m.group(1)}"
    return s


def load(path: Path, env: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "adapter" not in df.columns:
        df["adapter"] = df["model"].map(infer_adapter)
    else:
        df["adapter"] = df["adapter"].map(infer_adapter)
    df["family"] = df["model"].map(infer_family)
    df["environment"] = env
    if "ttc" in df.columns:
        df["ttc_label"] = df["ttc"].map(normalize_ttc)
    else:
        df["ttc_label"] = "unknown"
    return df


def add_metrics(df):
    df = df.copy()
    for c in [
        "bad_action",
        "gameable_hack_action",
        "easy_grader_action",
        "fake_completion_action",
        "bad_buy_action",
        "low_reward_buy",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    exploit_cols = [c for c in ["gameable_hack_action", "easy_grader_action", "fake_completion_action"] if c in df.columns]
    if exploit_cols:
        df["exploit_or_bad_rate_target"] = (df[exploit_cols].sum(axis=1) > 0).astype(float)
    elif "bad_buy_action" in df.columns:
        df["exploit_or_bad_rate_target"] = df["bad_buy_action"]
    elif "bad_action" in df.columns:
        df["exploit_or_bad_rate_target"] = df["bad_action"]
    else:
        df["exploit_or_bad_rate_target"] = np.nan

    return df


def summarize(df):
    df = add_metrics(df)
    p_col = "reasoning_p_hack" if "reasoning_p_hack" in df.columns else None
    if p_col is None:
        bins = [c for c in df.columns if c.startswith("reasoning_p_bin")]
        if bins:
            df["reasoning_p_hack"] = df[bins].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            p_col = "reasoning_p_hack"

    e_col = "reasoning_entropy_mean" if "reasoning_entropy_mean" in df.columns else None
    if e_col is None:
        bins = [c for c in df.columns if c.startswith("reasoning_entropy_bin")]
        if bins:
            df["reasoning_entropy_mean"] = df[bins].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            e_col = "reasoning_entropy_mean"

    rows = []
    for key, sub in df.groupby(["environment", "family", "adapter", "ttc_label"]):
        env, fam, adapter, ttc = key
        rows.append({
            "environment": env,
            "family": fam,
            "adapter": adapter,
            "ttc": ttc,
            "n": len(sub),
            "mean_p_hack": pd.to_numeric(sub[p_col], errors="coerce").mean() if p_col else np.nan,
            "mean_entropy": pd.to_numeric(sub[e_col], errors="coerce").mean() if e_col else np.nan,
            "behavior_risk_rate": pd.to_numeric(sub["exploit_or_bad_rate_target"], errors="coerce").mean(),
        })
    return pd.DataFrame(rows)


def plot(summary, outpath: Path, family="Qwen"):
    sub = summary[summary["family"] == family].copy()
    if sub.empty:
        raise RuntimeError(f"No rows for {family}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    for ax, xcol, title in [
        (axes[0], "mean_p_hack", "Activation vs behavior"),
        (axes[1], "mean_entropy", "Entropy vs behavior"),
    ]:
        for _, row in sub.iterrows():
            a = row["adapter"]
            marker = "s" if row["environment"] == "WebShop" else "o"
            ax.scatter(
                row[xcol],
                row["behavior_risk_rate"],
                s=80,
                color=COLORS.get(a, "#999999"),
                marker=marker,
                edgecolor="black",
                alpha=0.8,
            )
            if a in {"mix50", "hack"} and row["environment"] == "ALFWorld":
                ax.text(row[xcol], row["behavior_risk_rate"], f" {a}", fontsize=8)

        ax.set_title(title)
        ax.set_xlabel(xcol.replace("_", " "))
        ax.set_ylabel("Exploit / bad-action rate")
        ax.grid(alpha=0.2)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alfworld", type=Path, default=Path("BALROG/analysis_gameable_alfworld_temporal/steps_with_entropy_phack_temporal.csv"))
    ap.add_argument("--webshop", type=Path, default=None)
    ap.add_argument("--family", type=str, default="Qwen")
    ap.add_argument("--outdir", type=Path, default=Path("paper_analysis_outputs"))
    args = ap.parse_args()

    dfs = []
    if args.alfworld.exists():
        dfs.append(load(args.alfworld, "ALFWorld"))
    if args.webshop and args.webshop.exists():
        dfs.append(load(args.webshop, "WebShop"))
    if not dfs:
        raise RuntimeError("No data found.")

    df = pd.concat(dfs, ignore_index=True)
    summ = summarize(df)

    args.outdir.mkdir(parents=True, exist_ok=True)
    family_slug = args.family.lower()

    csv_path = args.outdir / f"risk_state_vs_behavior_summary_{family_slug}.csv"
    png_path = args.outdir / f"risk_state_vs_behavior_{family_slug}.png"

    summ.to_csv(csv_path, index=False)
    plot(summ, png_path, family=args.family)

    print("[DONE]")
    print(" ", csv_path)
    print(" ", png_path)
    print(" ", png_path.with_suffix(".pdf"))

if __name__ == "__main__":
    main()