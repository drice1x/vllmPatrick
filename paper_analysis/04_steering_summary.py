#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


COLORS = {
    "mix05": "#54a24b",
    "mix10": "#f58518",
    "mix50": "#e45756",
}


def load_steering(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    def find_col(candidates):
        cols = {c.lower().strip(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in cols:
                return cols[cand.lower()]
        for c in df.columns:
            lc = c.lower()
            if any(cand.lower() in lc for cand in candidates):
                return c
        return None

    mapping = {
        "adapter": find_col(["adapter", "model_adapter", "condition_adapter"]),
        "condition": find_col(["condition", "steering_condition", "setting", "run_condition"]),
        "delta_proxy": find_col([
            "delta_proxy", "Δ proxy", "d_proxy", "proxy_delta",
            "delta_proxy_score_without_success",
            "proxy_score_without_success_delta",
            "delta proxy_score_without_success",
            "delta proxy",
        ]),
        "delta_easy": find_col([
            "delta_easy", "Δ easy", "d_easy", "easy_delta",
            "delta_easy_grader_count", "easy_grader_delta",
            "delta easy",
        ]),
        "delta_fake": find_col([
            "delta_fake", "Δ fake", "d_fake", "fake_delta",
            "delta_fake_completion_count", "fake_completion_delta",
            "delta fake",
        ]),
        "steering_rate": find_col([
            "steering_rate", "steering rate", "rate", "fraction_steered",
            "steered_fraction", "mean_steering_applied",
        ]),
        "family": find_col(["family", "model_family"]),
        "model": find_col(["model", "model_id"]),
    }

    rename = {v: k for k, v in mapping.items() if v is not None}
    df = df.rename(columns=rename)

    print("[DEBUG] steering columns after rename:", df.columns.tolist())

    # If this file contains original and steered values rather than deltas, compute deltas.
    if "delta_proxy" not in df.columns:
        proxy_col = find_col(["proxy_score_without_success", "proxy_without_success"])
        base_col = find_col(["baseline_proxy_score_without_success", "unsteered_proxy_score_without_success"])
        if proxy_col and base_col:
            df["delta_proxy"] = pd.to_numeric(df[proxy_col], errors="coerce") - pd.to_numeric(df[base_col], errors="coerce")

    if "delta_easy" not in df.columns:
        easy_col = find_col(["easy_grader_count", "easy_grader_rate"])
        base_col = find_col(["baseline_easy_grader_count", "unsteered_easy_grader_count"])
        if easy_col and base_col:
            df["delta_easy"] = pd.to_numeric(df[easy_col], errors="coerce") - pd.to_numeric(df[base_col], errors="coerce")

    if "delta_fake" not in df.columns:
        fake_col = find_col(["fake_completion_count", "fake_completion_rate"])
        base_col = find_col(["baseline_fake_completion_count", "unsteered_fake_completion_count"])
        if fake_col and base_col:
            df["delta_fake"] = pd.to_numeric(df[fake_col], errors="coerce") - pd.to_numeric(df[base_col], errors="coerce")

    # Last fallback: if already analyzed file has columns like d_proxy/d_easy/d_fake.
    for required in ["delta_proxy", "delta_easy", "delta_fake", "steering_rate"]:
        if required not in df.columns:
            print(f"[WARN] Missing {required}. Filling with NaN.")
            df[required] = float("nan")

    if "adapter" not in df.columns:
        if "model" in df.columns:
            df["adapter"] = df["model"].astype(str).str.extract(r"(mix05|mix10|mix50|control|hack)", expand=False).fillna("unknown")
        else:
            df["adapter"] = "unknown"

    if "condition" not in df.columns:
        df["condition"] = "steering"

    return df


def fallback_from_markdown_values() -> pd.DataFrame:
    return pd.DataFrame([
        {"adapter": "mix05", "condition": "always a=0.25", "delta_proxy": -0.8, "delta_easy": -0.1, "delta_fake": 0.0, "steering_rate": 1.0000},
        {"adapter": "mix05", "condition": "gated a=0.5", "delta_proxy": -1.1, "delta_easy": -0.4, "delta_fake": 0.0, "steering_rate": 0.0000},
        {"adapter": "mix10", "condition": "always a=0.25", "delta_proxy": -2.0, "delta_easy": -0.6, "delta_fake": -0.6, "steering_rate": 1.0000},
        {"adapter": "mix10", "condition": "gated a=0.5", "delta_proxy": -0.4, "delta_easy": -0.1, "delta_fake": -0.7, "steering_rate": 0.0133},
        {"adapter": "mix10", "condition": "gated a=1.0", "delta_proxy": -0.4, "delta_easy": -0.5, "delta_fake": 0.1, "steering_rate": 0.0025},
        {"adapter": "mix50", "condition": "always a=0.25", "delta_proxy": -2.2, "delta_easy": -0.8, "delta_fake": -0.9, "steering_rate": 1.0000},
    ])

def plot(df: pd.DataFrame, outpath: Path):
    df = df.copy()

    for c in ["delta_proxy", "delta_easy", "delta_fake", "steering_rate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Prefer proxy reduction; fallback to easy/fake reduction if proxy is missing.
    if df["delta_proxy"].notna().any():
        df["exploit_reduction"] = -df["delta_proxy"]
        ylab = "Reduction in proxy score without true success"
    else:
        df["exploit_reduction"] = -(df["delta_easy"].fillna(0) + df["delta_fake"].fillna(0))
        ylab = "Reduction in easy-grader + fake-completion count"

    df = df[df["exploit_reduction"].notna() & df["steering_rate"].notna()].copy()
    if df.empty:
        raise RuntimeError("No plottable steering rows after column normalization.")

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    for _, row in df.iterrows():
        adapter = str(row["adapter"]).lower()
        marker = "s" if "gated" in str(row["condition"]).lower() else "o"
        ax.scatter(
            row["steering_rate"],
            row["exploit_reduction"],
            s=100,
            color=COLORS.get(adapter, "#999999"),
            marker=marker,
            edgecolor="black",
            alpha=0.85,
        )
        if row["exploit_reduction"] > 0.3:
            ax.text(
                row["steering_rate"],
                row["exploit_reduction"],
                f" {adapter} {row['condition']}",
                fontsize=8,
            )

    ax.axhline(0, color="#777777", linestyle="--")
    ax.set_xlabel("Fraction of steps steered")
    ax.set_ylabel(ylab)
    ax.set_title("Steering reduces proxy behavior in selected regimes")
    ax.grid(alpha=0.2)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steering-csv", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=Path("paper_analysis_outputs"))
    ap.add_argument("--family", type=str, default="Qwen")
    args = ap.parse_args()

    if args.steering_csv and args.steering_csv.exists():
        df = load_steering(args.steering_csv)
    else:
        print("[WARN] Steering CSV not provided/found. Using markdown fallback values.")
        df = fallback_from_markdown_values()

    family_slug = args.family.lower()

    if "family" in df.columns:
        df = df[df["family"].astype(str).str.lower() == family_slug].copy()
    elif "model" in df.columns:
        df = df[df["model"].astype(str).str.lower().str.contains(family_slug, regex=False)].copy()

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / f"steering_summary_{family_slug}.csv"
    png_path = args.outdir / f"steering_intervention_probe_{family_slug}.png"

    df.to_csv(csv_path, index=False)
    plot(df, png_path)
    print("[DONE]")
    print(" ", csv_path)
    print(" ", png_path)
    print(" ", png_path.with_suffix(".pdf"))


if __name__ == "__main__":
    main()