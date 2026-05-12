#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ADAPTER_ORDER = ["control", "mix05", "mix10", "mix50", "hack"]


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


def normalize_ttc_label(x) -> str:
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


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1 / (1 + np.exp(-z))


def fit_logreg(X, y, lr=0.1, steps=600, l2=1e-3):
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(steps):
        p = sigmoid(Xb @ w)
        grad = (Xb.T @ (p - y)) / len(y)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return w


def predict_logreg(X, w):
    Xb = np.column_stack([np.ones(len(X)), X])
    return sigmoid(Xb @ w)


def average_precision(y, score):
    order = np.argsort(-score)
    y_sorted = y[order]
    n_pos = y.sum()
    if n_pos <= 0:
        return np.nan
    cum_pos = np.cumsum(y_sorted)
    precision = cum_pos / (np.arange(len(y_sorted)) + 1)
    return float((precision * y_sorted).sum() / n_pos)


def auroc(y, score):
    y = np.asarray(y)
    score = np.asarray(score)
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    # Mann-Whitney U style
    ranks = pd.Series(score).rank(method="average").to_numpy()
    r_pos = ranks[y == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def recall_at_frac(y, score, frac=0.2):
    n = len(y)
    k = max(1, int(np.ceil(frac * n)))
    idx = np.argsort(-score)[:k]
    denom = y.sum()
    if denom <= 0:
        return np.nan
    return float(y[idx].sum() / denom)


def load_dataset(path: Path, environment: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "adapter" not in df.columns:
        df["adapter"] = df["model"].map(infer_adapter)
    else:
        df["adapter"] = df["adapter"].map(infer_adapter)
    if "family" not in df.columns:
        df["family"] = df["model"].map(infer_family)
    if "ttc" in df.columns:
        df["ttc_label"] = df["ttc"].map(normalize_ttc_label)
    else:
        df["ttc_label"] = "unknown"
    df["environment"] = environment

    if "step" in df.columns:
        df["step_num"] = pd.to_numeric(df["step"], errors="coerce")
    elif "step_num" not in df.columns:
        df["step_num"] = np.nan

    return df


def add_shifted_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["source_file", "step_num"]).copy()

    for c in [
        "bad_action",
        "gameable_hack_action",
        "easy_grader_action",
        "fake_completion_action",
        "buy_action",
        "bad_buy_action",
        "low_reward_buy",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    exploit_cols = [c for c in ["gameable_hack_action", "easy_grader_action", "fake_completion_action"] if c in df.columns]
    if exploit_cols:
        df["exploit_action"] = (df[exploit_cols].sum(axis=1) > 0).astype(float)

    for target in ["bad_action", "exploit_action", "bad_buy_action", "low_reward_buy", "buy_action"]:
        if target in df.columns:
            df[f"{target}_t1"] = df.groupby("source_file")[target].shift(-1)

    return df


def feature_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    p_cols = [c for c in [
        "reasoning_p_hack",
        "reasoning_p_hack_last",
        "reasoning_p_hack_peak",
        "reasoning_p_hack_late_change",
        "reasoning_p_hack_late_slope",
        "reasoning_p_hack_centroid",
    ] if c in df.columns]

    # fallback names
    if "reasoning_p_hack" not in p_cols and "reasoning_p_bin0" in df.columns:
        p_cols += [c for c in df.columns if c.startswith("reasoning_p_bin")]

    e_cols = [c for c in [
        "reasoning_entropy_mean",
        "reasoning_entropy_max",
        "reasoning_entropy_late_change",
        "reasoning_entropy_late_slope",
    ] if c in df.columns]
    if "reasoning_entropy_mean" not in e_cols and "reasoning_entropy_bin0" in df.columns:
        e_cols += [c for c in df.columns if c.startswith("reasoning_entropy_bin")]

    temporal_cols = [c for c in p_cols + e_cols if "late" in c or "slope" in c or "centroid" in c]

    context_cols = []
    for c in ["step_num"]:
        if c in df.columns:
            context_cols.append(c)

    # One-hot-like context: adapter, environment, TTC
    for cat in ["environment", "ttc_label"]:
        if cat in df.columns:
            dummies = pd.get_dummies(df[cat].astype(str), prefix=cat)
            for dc in dummies.columns:
                df[dc] = dummies[dc].astype(float)
            context_cols += list(dummies.columns)

    return {
        "p_hack_only": p_cols,
        "entropy_only": e_cols,
        "temporal_internal": temporal_cols,
        "p_hack_plus_entropy": list(dict.fromkeys(p_cols + e_cols)),
        "context_only": context_cols,
        "internal_all": list(dict.fromkeys(p_cols + e_cols + temporal_cols)),
        "internal_plus_context": list(dict.fromkeys(p_cols + e_cols + temporal_cols + context_cols)),
    }


def grouped_cv_predict(df, cols, target_col, group_col="source_file"):
    valid = df[[target_col] + cols + [group_col]].copy()
    for c in cols:
        valid[c] = pd.to_numeric(valid[c], errors="coerce")
    valid[target_col] = pd.to_numeric(valid[target_col], errors="coerce")
    valid = valid.dropna(subset=[target_col])
    valid = valid[valid[target_col].isin([0, 1])].copy()
    valid[cols] = valid[cols].fillna(valid[cols].median(numeric_only=True)).fillna(0.0)

    y = valid[target_col].to_numpy(float)
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return None

    pred = np.full(len(valid), np.nan)
    groups = valid[group_col].astype(str).to_numpy()
    X = valid[cols].to_numpy(float)

    for g in np.unique(groups):
        test = groups == g
        train = ~test
        if len(np.unique(y[train])) < 2:
            continue

        mu = X[train].mean(axis=0)
        sd = X[train].std(axis=0)
        sd[sd == 0] = 1.0

        Xtr = (X[train] - mu) / sd
        Xte = (X[test] - mu) / sd

        w = fit_logreg(Xtr, y[train])
        pred[test] = predict_logreg(Xte, w)

    ok = np.isfinite(pred)
    if ok.sum() < 5 or len(np.unique(y[ok])) < 2:
        return None

    y_ok = y[ok]
    p_ok = pred[ok]
    base = float(y_ok.mean())
    ap = average_precision(y_ok, p_ok)
    return {
        "N": int(len(y_ok)),
        "positives": int(y_ok.sum()),
        "base_rate": base,
        "AUROC": auroc(y_ok, p_ok),
        "AUPRC": ap,
        "AUPRC_gain": ap - base,
        "Recall@20": recall_at_frac(y_ok, p_ok, 0.2),
    }


def evaluate(df, targets):
    df = add_shifted_targets(df)
    feats = feature_columns(df)
    rows = []

    for target in targets:
        if target not in df.columns:
            print("[WARN] missing target", target)
            continue
        for group_name, cols in feats.items():
            cols = [c for c in cols if c in df.columns]
            if not cols:
                continue
            res = grouped_cv_predict(df, cols, target)
            if res is None:
                continue
            rows.append({
                "target": target,
                "feature_group": group_name,
                "n_features": len(cols),
                **res,
            })

    return pd.DataFrame(rows)


def plot_results(res: pd.DataFrame, outpath: Path):
    targets = list(res["target"].unique())
    groups = [
        "p_hack_only",
        "entropy_only",
        "temporal_internal",
        "context_only",
        "p_hack_plus_entropy",
        "internal_all",
        "internal_plus_context",
    ]
    labels = {
        "p_hack_only": "act\nonly",
        "entropy_only": "entropy\nonly",
        "temporal_internal": "temporal\nint",
        "context_only": "context\nonly",
        "p_hack_plus_entropy": "act+\nentropy",
        "internal_all": "int\nall",
        "internal_plus_context": "int+\ncontext",
    }
    colors = {
        "p_hack_only": "#d62728",
        "entropy_only": "#1f77b4",
        "temporal_internal": "#2ca02c",
        "context_only": "#7f7f7f",
        "p_hack_plus_entropy": "#9467bd",
        "internal_all": "#8c564b",
        "internal_plus_context": "#e377c2",
    }

    fig, axes = plt.subplots(1, len(targets), figsize=(4.1 * len(targets), 3.2), sharey=True)
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        sub = res[res["target"] == target].copy()
        xs = []
        ys = []
        cs = []
        xt = []
        for i, g in enumerate(groups):
            row = sub[sub["feature_group"] == g]
            if row.empty:
                xs.append(i)
                ys.append(np.nan)
                cs.append("#eeeeee")
            else:
                val = float(row.iloc[0]["AUPRC_gain"])
                xs.append(i)
                ys.append(val)
                cs.append(colors[g])
                ax.text(
                    i,
                    val + 0.01 if val >= 0 else val - 0.02,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=8,
                )
            xt.append(labels[g])

        ax.bar(xs, ys, color=cs, alpha=0.9)
        ax.axhline(0, color="#777777", linestyle="--", linewidth=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(xt, rotation=35, ha="right")
        ax.set_title(target.replace("_t1", r"$_{t+1}$").replace("_", r"\_"))
        ax.grid(axis="y", alpha=0.15)

    axes[0].set_ylabel("AUPRC gain over base rate")
    fig.suptitle("Context-calibrated features improve next-step risk estimation", y=1.05)
    fig.tight_layout()
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

    if args.alfworld and args.alfworld.exists():
        alf = load_dataset(args.alfworld, "ALFWorld")
        alf = alf[alf["family"] == args.family].copy()
        dfs.append(alf)

    if args.webshop and args.webshop.exists():
        web = load_dataset(args.webshop, "WebShop")
        web = web[web["family"] == args.family].copy()
        dfs.append(web)

    if not dfs:
        raise RuntimeError("No datasets found.")

    df = pd.concat(dfs, ignore_index=True)

    targets = []
    for t in ["bad_action_t1", "exploit_action_t1", "bad_buy_action_t1", "low_reward_buy_t1"]:
        # add_shifted_targets creates these
        targets.append(t)

    res = evaluate(df, targets)
    if res.empty:
        raise RuntimeError("No stable prediction results.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    family_slug = args.family.lower()

    env_slug = "allenv"
    if args.webshop is None:
        env_slug = "alfworld"
    elif args.alfworld is None:
        env_slug = "webshop"

    csv_path = args.outdir / f"context_calibrated_prediction_ablation_{family_slug}_{env_slug}.csv"
    png_path = args.outdir / f"context_calibrated_prediction_ablation_{family_slug}_{env_slug}.png"

    #csv_path = args.outdir / f"context_calibrated_prediction_ablation_{family_slug}.csv"
    #png_path = args.outdir / f"context_calibrated_prediction_ablation_{family_slug}.png"

    res.to_csv(csv_path, index=False)
    plot_results(res, png_path)

    print("[DONE]")
    print(res.to_string(index=False))
    print("Saved:", csv_path)
    print("Saved:", png_path)


if __name__ == "__main__":
    main()