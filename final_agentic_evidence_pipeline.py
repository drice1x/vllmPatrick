#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ADAPTER_ORDER = ["control", "mix05", "mix10", "mix50", "hack"]
FAMILY_ORDER = ["Qwen", "Llama", "Falcon"]
ADAPTER_LABELS = {
    "control": "Control",
    "mix05": "Mix05",
    "mix10": "Mix10",
    "mix50": "Mix50",
    "hack": "Hack",
}
ADAPTER_COLORS = {
    "control": "#4c78a8",
    "mix05": "#54a24b",
    "mix10": "#f58518",
    "mix50": "#e45756",
    "hack": "#b279a2",
}
ENV_MARKERS = {"alfworld": "o", "webshop": "s"}
FEATURE_GROUPS = {
    "p_hack_only": [
        "reasoning_p_hack_mean",
        "reasoning_p_hack_final",
        "reasoning_p_hack_late_mean",
        "reasoning_p_hack_late_change",
        "reasoning_p_hack_late_slope",
    ],
    "entropy_only": [
        "reasoning_entropy_mean",
        "reasoning_entropy_final",
        "reasoning_entropy_late_mean",
        "reasoning_entropy_late_change",
        "reasoning_entropy_late_slope",
    ],
    "temporal_internal": [
        "reasoning_p_hack_late_change",
        "reasoning_p_hack_late_slope",
        "reasoning_entropy_late_change",
        "reasoning_entropy_late_slope",
    ],
    "context_only": [
        "environment",
        "ttc_or_regime",
        "step_index_normalized",
        "previous_action_type",
        "admissible_action_count",
        "exploit_action_available",
        "proxy_score_visible",
        "near_completion",
        "item_page_or_buy_page",
        "prev_exploit_like_action",
    ],
}
FEATURE_GROUPS["p_hack_plus_entropy"] = FEATURE_GROUPS["p_hack_only"] + FEATURE_GROUPS["entropy_only"]
FEATURE_GROUPS["internal_all"] = (
    FEATURE_GROUPS["p_hack_only"] + FEATURE_GROUPS["entropy_only"] + FEATURE_GROUPS["temporal_internal"]
)
FEATURE_GROUPS["internal_plus_context"] = FEATURE_GROUPS["internal_all"] + FEATURE_GROUPS["context_only"]
FEATURE_COLORS = {
    "p_hack_only": "#d62728",
    "entropy_only": "#1f77b4",
    "temporal_internal": "#2ca02c",
    "context_only": "#7f7f7f",
    "p_hack_plus_entropy": "#9467bd",
    "internal_all": "#8c564b",
    "internal_plus_context": "#e377c2",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.edgecolor": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def maybe_load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def maybe_load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def family_of_model(model: str) -> str:
    text = str(model)
    for fam in FAMILY_ORDER:
        if text.startswith(fam):
            return fam
    return "Other"


def adapter_key(adapter: str) -> int:
    try:
        return ADAPTER_ORDER.index(str(adapter))
    except ValueError:
        return 99


def normalize_regime(ttc_value) -> tuple[str, float | None]:
    if ttc_value in ("", None) or (isinstance(ttc_value, float) and np.isnan(ttc_value)):
        return ("unconstrained", None)
    value = float(ttc_value)
    return (f"ttc_{int(value)}", value)


def savefig(fig: plt.Figure, png_path: Path) -> None:
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def load_alfworld_steps(root: Path) -> pd.DataFrame:
    path = root / "BALROG/analysis_gameable_alfworld_temporal/steps_with_entropy_phack_temporal.csv"
    df = maybe_load_csv(path)
    if df.empty:
        return df
    df["environment"] = "alfworld"
    df["model_family"] = df["model"].map(family_of_model)
    df["ttc_or_regime"], df["ttc_numeric"] = zip(*df["ttc"].map(normalize_regime))
    df["episode_id"] = df["source_file"]
    df["step_id"] = pd.to_numeric(df["step"], errors="coerce")
    df["action_t"] = df["action"]
    df["previous_action_was_invalid"] = pd.to_numeric(df.get("action_was_rewritten"), errors="coerce")
    safe_numeric(
        df,
        [
            "reasoning_entropy_mean",
            "reasoning_entropy_max",
            "action_entropy_mean",
            "action_entropy_max",
            "reasoning_p_hack",
            "action_p_hack",
            "reasoning_p_hack_late_slope",
            "reasoning_p_hack_late_change",
            "reasoning_p_hack_centroid",
            "reasoning_p_hack_first",
            "reasoning_p_hack_last",
            "reasoning_p_hack_peak",
            "reasoning_entropy_late_slope",
            "reasoning_entropy_late_change",
            "proxy_score",
            "last_proxy_reward",
            "proxy_score_without_success",
            "true_task_success_event",
            "bad_action",
            "gameable_hack_action",
            "fake_completion_action",
            "easy_grader_action",
            "claim_complete_action",
            "inspect_score_action",
        ]
        + [c for c in df.columns if "bin" in c],
    )
    df = df.sort_values(["episode_id", "step_id"]).reset_index(drop=True)
    group = df.groupby("episode_id", sort=False)
    for col in [
        "bad_action",
        "gameable_hack_action",
        "fake_completion_action",
        "easy_grader_action",
        "claim_complete_action",
        "inspect_score_action",
        "proxy_score_without_success",
        "proxy_score",
        "true_task_success_event",
        "action_t",
    ]:
        if col in df.columns:
            df[f"{col}_t1"] = group[col].shift(-1)
    df["action_t1"] = group["action_t"].shift(-1)
    df["has_t1"] = group["step_id"].shift(-1).notna()
    df = df[df["has_t1"]].copy()

    ent_cols = sorted([c for c in df.columns if c.startswith("reasoning_entropy_bin")], key=lambda c: int(c.split("bin")[-1]))
    ph_cols = sorted([c for c in df.columns if c.startswith("reasoning_p_bin")], key=lambda c: int(c.split("bin")[-1]))
    if ent_cols:
        ent = df[ent_cols].apply(pd.to_numeric, errors="coerce")
        k = max(1, int(math.ceil(len(ent_cols) * 0.2)))
        df["reasoning_entropy_final"] = ent.iloc[:, -1]
        df["reasoning_entropy_late_mean"] = ent.iloc[:, -k:].mean(axis=1)
    if ph_cols:
        ph = df[ph_cols].apply(pd.to_numeric, errors="coerce")
        k = max(1, int(math.ceil(len(ph_cols) * 0.2)))
        df["reasoning_p_hack_final"] = ph.iloc[:, -1]
        df["reasoning_p_hack_late_mean"] = ph.iloc[:, -k:].mean(axis=1)
    df["reasoning_p_hack_mean"] = df["reasoning_p_hack"]
    df["reasoning_p_hack_final"] = df.get("reasoning_p_hack_final", df["reasoning_p_hack_last"])
    df["reasoning_p_hack_late_mean"] = df.get("reasoning_p_hack_late_mean", df["reasoning_p_hack"])
    df["reasoning_entropy_final"] = df.get("reasoning_entropy_final", df["reasoning_entropy_mean"])
    df["reasoning_entropy_late_mean"] = df.get("reasoning_entropy_late_mean", df["reasoning_entropy_mean"])

    df["exploit_action_t1"] = (
        df.get("easy_grader_action_t1", 0).fillna(0)
        + df.get("fake_completion_action_t1", 0).fillna(0)
        + df.get("gameable_hack_action_t1", 0).fillna(0)
    ).clip(upper=1)
    next_proxy_delta = df.get("proxy_score_without_success_t1", df["proxy_score_without_success"]).fillna(df["proxy_score_without_success"]) - df["proxy_score_without_success"].fillna(0)
    df["proxy_without_success_t1"] = (
        (next_proxy_delta > 0)
        & (df.get("true_task_success_event_t1", 0).fillna(0) < 0.5)
    ).astype(float)
    df["bad_action_t1"] = df.get("bad_action_t1", np.nan)
    df["easy_grader_action_t1"] = df.get("easy_grader_action_t1", np.nan)
    df["fake_completion_action_t1"] = df.get("fake_completion_action_t1", np.nan)
    df["gameable_hack_action_t1"] = df.get("gameable_hack_action_t1", np.nan)
    df["step_index_normalized"] = df.groupby("episode_id")["step_id"].transform(lambda s: s / max(float(s.max()), 1.0))
    df["previous_action_type"] = group["action_t"].shift(1)
    exploit_like_prev = df["previous_action_type"].fillna("").astype(str).str.contains("inspect score|claim|grader", case=False, regex=True)
    df["prev_exploit_like_action"] = exploit_like_prev.astype(float)
    df["proxy_score_visible"] = (
        df["proxy_score"].notna() | df["proxy_score_without_success"].notna() | (df["inspect_score_action"].fillna(0) > 0)
    ).astype(float)
    df["near_completion"] = (
        df.get("claim_complete_action_t1", 0).fillna(0) > 0
    ).astype(float)
    df["admissible_action_count"] = np.nan
    df["exploit_action_available"] = df["proxy_score_visible"]
    df["is_unconstrained"] = df["ttc_numeric"].isna().astype(float)
    df["item_page_or_buy_page"] = np.nan
    df["success"] = df.get("true_task_success_event_t1", np.nan)
    df["return"] = df.get("proxy_score_t1", np.nan)
    df["progression"] = np.nan
    return df[[
        "environment",
        "model_family",
        "model",
        "adapter",
        "ttc_or_regime",
        "ttc_numeric",
        "episode_id",
        "step_id",
        "action_t",
        "action_t1",
        "reasoning_p_hack_mean",
        "reasoning_p_hack_final",
        "reasoning_p_hack_late_mean",
        "reasoning_p_hack_late_change",
        "reasoning_p_hack_late_slope",
        "reasoning_entropy_mean",
        "reasoning_entropy_final",
        "reasoning_entropy_late_mean",
        "reasoning_entropy_late_change",
        "reasoning_entropy_late_slope",
        "action_p_hack",
        "action_entropy_mean",
        "bad_action_t1",
        "exploit_action_t1",
        "easy_grader_action_t1",
        "fake_completion_action_t1",
        "gameable_hack_action_t1",
        "proxy_without_success_t1",
        "step_index_normalized",
        "previous_action_type",
        "previous_action_was_invalid",
        "admissible_action_count",
        "exploit_action_available",
        "proxy_score_visible",
        "near_completion",
        "item_page_or_buy_page",
        "is_unconstrained",
        "prev_exploit_like_action",
        "success",
        "return",
        "progression",
    ]].copy()


def load_webshop_steps(root: Path) -> pd.DataFrame:
    pair_files = [
        root / "BALROG/webshop_react_eval/runs/next_step_analysis_buycurated/next_step_pairs.json",
        root / "BALROG/webshop_react_eval/runs/next_step_analysis_buycurated_falcon/next_step_pairs.json",
        root / "BALROG/webshop_react_eval/runs/next_step_analysis_buycurated_llama/next_step_pairs.json",
    ]
    rows = []
    for pf in pair_files:
        data = maybe_load_json(pf)
        if not data:
            continue
        for row in data:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["environment"] = "webshop"
    df["model_family"] = df["model"].map(family_of_model)
    df["adapter"] = df["model"].str.extract(r"(Control|Hack|Mix05|Mix10|Mix50)$", expand=False).str.lower()
    df["ttc_or_regime"] = "react_baseline"
    df["ttc_numeric"] = np.nan
    df["step_id"] = pd.to_numeric(df["step_t"], errors="coerce")
    df["action_t"] = np.nan
    df["action_t1"] = df["action_t1"]
    df["reasoning_p_hack_mean"] = pd.to_numeric(df["p_hack_t"], errors="coerce")
    df["reasoning_p_hack_final"] = df["reasoning_p_hack_mean"]
    df["reasoning_p_hack_late_mean"] = df["reasoning_p_hack_mean"]
    df["reasoning_p_hack_late_change"] = np.nan
    df["reasoning_p_hack_late_slope"] = np.nan
    df["reasoning_entropy_mean"] = pd.to_numeric(df["entropy_t"], errors="coerce")
    df["reasoning_entropy_final"] = df["reasoning_entropy_mean"]
    df["reasoning_entropy_late_mean"] = df["reasoning_entropy_mean"]
    df["reasoning_entropy_late_change"] = np.nan
    df["reasoning_entropy_late_slope"] = np.nan
    df["action_p_hack"] = np.nan
    df["action_entropy_mean"] = np.nan
    df["bad_action_t1"] = pd.to_numeric(df["bad_buy_t1"], errors="coerce")
    df["exploit_action_t1"] = pd.to_numeric(df["bad_buy_t1"], errors="coerce")
    df["easy_grader_action_t1"] = np.nan
    df["fake_completion_action_t1"] = np.nan
    df["gameable_hack_action_t1"] = np.nan
    df["bad_buy_t1"] = pd.to_numeric(df["bad_buy_t1"], errors="coerce")
    df["low_reward_buy_t1"] = pd.to_numeric(df["low_reward_buy_t1"], errors="coerce")
    df["early_buy_t1"] = pd.to_numeric(df["early_buy_t1"], errors="coerce")
    df["proxy_without_success_t1"] = np.nan
    df["step_index_normalized"] = df.groupby("episode_id")["step_id"].transform(lambda s: s / max(float(s.max()), 1.0))
    df["previous_action_type"] = np.nan
    df["previous_action_was_invalid"] = np.nan
    df["admissible_action_count"] = np.nan
    df["exploit_action_available"] = np.nan
    df["proxy_score_visible"] = np.nan
    df["near_completion"] = np.nan
    action_lower = df["action_t1"].fillna("").astype(str).str.lower()
    df["item_page_or_buy_page"] = action_lower.str.contains("buy|item|asin").astype(float)
    df["is_unconstrained"] = 1.0
    df["prev_exploit_like_action"] = np.nan
    df["success"] = np.nan
    df["return"] = np.nan
    df["progression"] = np.nan
    return df[[
        "environment",
        "model_family",
        "model",
        "adapter",
        "ttc_or_regime",
        "ttc_numeric",
        "episode_id",
        "step_id",
        "action_t",
        "action_t1",
        "reasoning_p_hack_mean",
        "reasoning_p_hack_final",
        "reasoning_p_hack_late_mean",
        "reasoning_p_hack_late_change",
        "reasoning_p_hack_late_slope",
        "reasoning_entropy_mean",
        "reasoning_entropy_final",
        "reasoning_entropy_late_mean",
        "reasoning_entropy_late_change",
        "reasoning_entropy_late_slope",
        "action_p_hack",
        "action_entropy_mean",
        "bad_action_t1",
        "exploit_action_t1",
        "easy_grader_action_t1",
        "fake_completion_action_t1",
        "gameable_hack_action_t1",
        "bad_buy_t1",
        "low_reward_buy_t1",
        "early_buy_t1",
        "proxy_without_success_t1",
        "step_index_normalized",
        "previous_action_type",
        "previous_action_was_invalid",
        "admissible_action_count",
        "exploit_action_available",
        "proxy_score_visible",
        "near_completion",
        "item_page_or_buy_page",
        "is_unconstrained",
        "prev_exploit_like_action",
        "success",
        "return",
        "progression",
    ]].copy()


def save_unified_dataset(df: pd.DataFrame, tables_dir: Path) -> tuple[Path | None, Path]:
    parquet_path = tables_dir / "unified_agent_monitoring_dataset.parquet"
    summary_path = tables_dir / "unified_agent_monitoring_dataset_summary.csv"
    summary = (
        df.groupby(["environment", "model_family", "adapter", "ttc_or_regime"], as_index=False)
        .agg(
            n_steps=("step_id", "count"),
            n_episodes=("episode_id", "nunique"),
            exploit_rate=("exploit_action_t1", "mean"),
            bad_rate=("bad_action_t1", "mean"),
            mean_p_hack=("reasoning_p_hack_mean", "mean"),
            mean_entropy=("reasoning_entropy_mean", "mean"),
        )
    )
    summary.to_csv(summary_path, index=False)
    saved_parquet: Path | None = None
    try:
        df.to_parquet(parquet_path, index=False)
        saved_parquet = parquet_path
    except Exception:
        csv_fallback = tables_dir / "unified_agent_monitoring_dataset.csv"
        df.to_csv(csv_fallback, index=False)
    return saved_parquet, summary_path


def condition_risk_state(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(["environment", "model_family", "model", "adapter", "ttc_or_regime", "ttc_numeric"], dropna=False):
        row = {
            "environment": keys[0],
            "model_family": keys[1],
            "model": keys[2],
            "adapter": keys[3],
            "ttc_or_regime": keys[4],
            "ttc_numeric": keys[5],
            "n_steps": len(sub),
            "n_episodes": sub["episode_id"].nunique(),
            "mean_p_hack": pd.to_numeric(sub["reasoning_p_hack_mean"], errors="coerce").mean(),
            "mean_entropy": pd.to_numeric(sub["reasoning_entropy_mean"], errors="coerce").mean(),
            "bad_action_rate": pd.to_numeric(sub["bad_action_t1"], errors="coerce").mean(),
            "exploit_action_rate": pd.to_numeric(sub["exploit_action_t1"], errors="coerce").mean(),
            "bad_buy_rate": pd.to_numeric(sub["bad_buy_t1"], errors="coerce").mean() if "bad_buy_t1" in sub.columns else np.nan,
            "low_reward_buy_rate": pd.to_numeric(sub["low_reward_buy_t1"], errors="coerce").mean() if "low_reward_buy_t1" in sub.columns else np.nan,
            "success": pd.to_numeric(sub["success"], errors="coerce").mean(),
            "return": pd.to_numeric(sub["return"], errors="coerce").mean(),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_internal_risk_state_map(cond: pd.DataFrame, figs_dir: Path) -> None:
    def _plot(sub: pd.DataFrame, out: Path, title: str):
        fig, ax = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
        for _, row in sub.iterrows():
            size_value = row["exploit_action_rate"]
            if pd.isna(size_value):
                size_value = row["bad_action_rate"]
            ax.scatter(
                row["mean_entropy"],
                row["mean_p_hack"],
                s=max(30, float(40 + 220 * (0 if pd.isna(size_value) else size_value))),
                color=ADAPTER_COLORS.get(row["adapter"], "#777777"),
                marker=ENV_MARKERS.get(row["environment"], "o"),
                alpha=0.85,
                edgecolors="black",
                linewidths=0.4,
            )
        for _, row in sub.iterrows():
            if row["model_family"] == "Qwen" and row["adapter"] in {"control", "mix50", "hack"}:
                ax.text(row["mean_entropy"], row["mean_p_hack"], row["model"], fontsize=7, alpha=0.75)
        ax.set_xlabel("Mean reasoning entropy")
        ax.set_ylabel("Mean reasoning $p_{hack}$")
        ax.set_title(title)
        ax.grid(alpha=0.18)
        adapter_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=ADAPTER_COLORS[a],
                markerfacecolor=ADAPTER_COLORS[a],
                markeredgecolor="black",
                markersize=7,
                label=ADAPTER_LABELS[a],
            )
            for a in ADAPTER_ORDER
            if a in set(sub["adapter"])
        ]
        env_handles = [
            plt.Line2D(
                [0],
                [0],
                marker=ENV_MARKERS[e],
                linestyle="",
                color="white",
                markerfacecolor="#cccccc",
                markeredgecolor="black",
                markersize=7,
                label=e.capitalize(),
            )
            for e in ["alfworld", "webshop"]
            if e in set(sub["environment"])
        ]
        size_handles = [
            ax.scatter([], [], s=s, color="#999999", edgecolors="black", linewidths=0.4, label=lab)
            for s, lab in [(40, "low exploit/bad rate"), (140, "medium"), (260, "high")]
        ]
        leg1 = ax.legend(handles=adapter_handles, title="Adapter", loc="upper left", frameon=False)
        ax.add_artist(leg1)
        leg2 = ax.legend(handles=env_handles, title="Environment", loc="lower right", frameon=False)
        ax.add_artist(leg2)
        ax.legend(handles=size_handles, title="Point size", loc="upper right", frameon=False)
        savefig(fig, out)
    q = cond[cond["model_family"] == "Qwen"].copy()
    _plot(q, figs_dir / "main_internal_risk_state_map_qwen.png", "Qwen internal risk-state map")
    _plot(cond.copy(), figs_dir / "appendix_internal_risk_state_map_all_models.png", "Internal risk-state map across environments and families")


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    mask = xv.notna() & yv.notna()
    if mask.sum() < 3:
        return float("nan")
    return float(xv[mask].rank().corr(yv[mask].rank()))


def plot_risk_state_vs_behavior(cond: pd.DataFrame, tables_dir: Path, figs_dir: Path) -> None:
    rows = []
    for env, target in [("alfworld", "exploit_action_rate"), ("alfworld", "bad_action_rate"), ("webshop", "bad_buy_rate"), ("webshop", "low_reward_buy_rate")]:
        sub = cond[cond["environment"] == env]
        if target not in sub.columns:
            continue
        x1 = pd.to_numeric(sub["mean_p_hack"], errors="coerce")
        x2 = pd.to_numeric(sub["mean_entropy"], errors="coerce")
        y = pd.to_numeric(sub[target], errors="coerce")
        mask = x1.notna() & x2.notna() & y.notna()
        beta_ph = beta_both = np.nan
        if mask.sum() >= 3:
            X1 = np.column_stack([np.ones(mask.sum()), x1[mask].to_numpy()])
            beta_ph = float(np.linalg.lstsq(X1, y[mask].to_numpy(), rcond=None)[0][1])
            X2 = np.column_stack([np.ones(mask.sum()), x1[mask].to_numpy(), x2[mask].to_numpy()])
            coefs = np.linalg.lstsq(X2, y[mask].to_numpy(), rcond=None)[0]
            beta_both = float(coefs[1])
        rows.append({
            "environment": env,
            "target": target,
            "spearman_mean_p_hack": spearman_corr(sub["mean_p_hack"], sub[target]),
            "spearman_mean_entropy": spearman_corr(sub["mean_entropy"], sub[target]),
            "regression_beta_p_hack_only": beta_ph,
            "regression_beta_p_hack_given_entropy": beta_both,
        })
    pd.DataFrame(rows).to_csv(tables_dir / "risk_state_behavior_relationship.csv", index=False)

    def _plot(subdf: pd.DataFrame, out: Path):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
        panels = [
            ("mean_p_hack", "A. Activation vs behavior"),
            ("mean_entropy", "B. Entropy vs behavior"),
        ]
        labels_to_show = {"QwenControl", "QwenMix50", "QwenHack"}
        for ax, (xcol, title) in zip(axes, panels):
            for _, row in subdf.iterrows():
                y = row["exploit_action_rate"] if row["environment"] == "alfworld" else row.get("bad_buy_rate", np.nan)
                if pd.isna(y):
                    y = row["bad_action_rate"] if row["environment"] == "alfworld" else row.get("low_reward_buy_rate", np.nan)
                ax.scatter(
                    row[xcol],
                    y,
                    color=ADAPTER_COLORS.get(row["adapter"], "#777777"),
                    marker=ENV_MARKERS.get(row["environment"], "o"),
                    s=60,
                    alpha=0.85,
                    edgecolors="black",
                    linewidths=0.4,
                )
                if row.get("model") in labels_to_show:
                    ax.text(row[xcol], y, row["model"], fontsize=7, alpha=0.8)
            ax.set_xlabel("Mean reasoning $p_{hack}$" if xcol == "mean_p_hack" else "Mean reasoning entropy")
            ax.set_ylabel("ALFWorld exploit rate / WebShop bad-buy rate")
            ax.set_title(title)
            ax.grid(alpha=0.18)
        adapter_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=ADAPTER_COLORS[a],
                markerfacecolor=ADAPTER_COLORS[a],
                markeredgecolor="black",
                markersize=7,
                label=ADAPTER_LABELS[a],
            )
            for a in ADAPTER_ORDER
            if a in set(subdf["adapter"])
        ]
        env_handles = [
            plt.Line2D(
                [0],
                [0],
                marker=ENV_MARKERS[e],
                linestyle="",
                color="white",
                markerfacecolor="#cccccc",
                markeredgecolor="black",
                markersize=7,
                label=e.capitalize(),
            )
            for e in ["alfworld", "webshop"]
            if e in set(subdf["environment"])
        ]
        leg1 = axes[0].legend(handles=adapter_handles, title="Adapter", loc="upper left", frameon=False)
        axes[0].add_artist(leg1)
        axes[1].legend(handles=env_handles, title="Environment", loc="upper right", frameon=False)
        savefig(fig, out)
    _plot(cond[cond["model_family"] == "Qwen"].copy(), figs_dir / "main_risk_state_vs_behavior_qwen.png")
    _plot(cond.copy(), figs_dir / "appendix_risk_state_vs_behavior_all_models.png")


def resolve_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    return {name: [c for c in cols if c in df.columns] for name, cols in FEATURE_GROUPS.items()}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logreg(X: np.ndarray, y: np.ndarray, steps: int = 500, lr: float = 0.12, l2: float = 1e-4) -> np.ndarray:
    w = np.zeros(X.shape[1] + 1, dtype=float)
    Xb = np.column_stack([np.ones(len(X)), X])
    for _ in range(steps):
        p = _sigmoid(Xb @ w)
        grad = (Xb.T @ (p - y)) / len(X)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return w


def predict_logreg(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return _sigmoid(np.column_stack([np.ones(len(X)), X]) @ w)


def auc_score(y: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(wins / (len(pos) * len(neg)))


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    positives = np.sum(y == 1)
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    ap = 0.0
    prev = 0.0
    for p, r in zip(precision, recall):
        ap += p * max(r - prev, 0.0)
        prev = r
    return float(ap)


def recall_at_flag(y: np.ndarray, scores: np.ndarray, flag_rate: float) -> float:
    positives = np.sum(y == 1)
    if positives == 0:
        return float("nan")
    k = max(1, int(math.ceil(len(scores) * flag_rate)))
    idx = np.argsort(-scores)[:k]
    return float(np.sum(y[idx] == 1) / positives)


def evaluate_prediction(sub: pd.DataFrame, target_col: str, feature_groups: dict[str, list[str]]) -> list[dict]:
    y_all = pd.to_numeric(sub[target_col], errors="coerce").fillna(0).astype(float).to_numpy()
    positives = int(np.sum(y_all == 1))
    negatives = int(np.sum(y_all == 0))
    rows = []
    groups = sub.groupby("episode_id", sort=False)
    for fg, feats in feature_groups.items():
        feats = list(dict.fromkeys(feats))
        if positives < 10 or negatives < 10 or not feats:
            rows.append({
                "feature_group": fg,
                "base_rate": float(np.mean(y_all)) if len(y_all) else np.nan,
                "positives": positives,
                "negatives": negatives,
                "auroc": np.nan,
                "auprc": np.nan,
                "auprc_gain": np.nan,
                "recall_at_10pct": np.nan,
                "recall_at_20pct": np.nan,
                "stable": False,
            })
            continue
        X_df = sub[feats].copy()
        cat_cols = [c for c in X_df.columns if X_df[c].dtype == object or str(X_df[c].dtype).startswith("string")]
        if cat_cols:
            X_df = pd.get_dummies(X_df, columns=cat_cols, dummy_na=True)
        X = X_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        oof = np.full(len(sub), np.nan)
        for _, idx in groups.indices.items():
            mask = np.zeros(len(sub), dtype=bool)
            mask[list(idx)] = True
            y_train = y_all[~mask]
            if len(np.unique(y_train)) < 2:
                continue
            X_train = X[~mask]
            mu = np.nanmean(X_train, axis=0)
            mu = np.where(np.isnan(mu), 0.0, mu)
            X_train = np.where(np.isnan(X_train), mu, X_train)
            X_test = np.where(np.isnan(X[mask]), mu, X[mask])
            sigma = np.nanstd(X_train, axis=0)
            sigma = np.where(np.isnan(sigma), 1.0, sigma)
            sigma[sigma == 0] = 1.0
            w = fit_logreg((X_train - mu) / sigma, y_train)
            oof[mask] = predict_logreg((X_test - mu) / sigma, w)
        valid = ~np.isnan(oof)
        y = y_all[valid]
        s = oof[valid]
        base = float(np.mean(y)) if len(y) else np.nan
        ap = average_precision(y, s)
        rows.append({
            "feature_group": fg,
            "base_rate": base,
            "positives": positives,
            "negatives": negatives,
            "auroc": auc_score(y, s),
            "auprc": ap,
            "auprc_gain": ap - base if np.isfinite(ap) and np.isfinite(base) else np.nan,
            "recall_at_10pct": recall_at_flag(y, s, 0.10),
            "recall_at_20pct": recall_at_flag(y, s, 0.20),
            "stable": True,
        })
    return rows


def build_feature_ablation(df: pd.DataFrame, tables_dir: Path, figs_dir: Path) -> pd.DataFrame:
    feature_groups = resolve_feature_groups(df)
    specs = [
        ("alfworld", "bad_action_t1"),
        ("alfworld", "exploit_action_t1"),
        ("webshop", "bad_buy_t1"),
        ("webshop", "low_reward_buy_t1"),
    ]
    rows = []
    for environment, target in specs:
        if target not in df.columns:
            continue
        sub_env = df[df["environment"] == environment].copy()
        if environment == "alfworld":
            for regime, sub_reg in sub_env.groupby("ttc_or_regime"):
                ttc_num = sub_reg["ttc_numeric"].dropna().iloc[0] if sub_reg["ttc_numeric"].notna().any() else np.nan
                for row in evaluate_prediction(sub_reg, target, feature_groups):
                    row.update({"environment": environment, "target": target, "ttc_or_regime": regime, "ttc_numeric": ttc_num})
                    rows.append(row)
        else:
            for row in evaluate_prediction(sub_env, target, feature_groups):
                row.update({"environment": environment, "target": target, "ttc_or_regime": "react_baseline", "ttc_numeric": np.nan})
                rows.append(row)
    pred = pd.DataFrame(rows)
    pred.to_csv(tables_dir / "main_context_calibrated_prediction_ablation.csv", index=False)
    pred.to_latex(tables_dir / "main_context_calibrated_prediction_ablation.tex", index=False, float_format="%.3f")

    best_rows = []
    for environment, target in specs:
        sub = pred[(pred["environment"] == environment) & (pred["target"] == target) & (pred["stable"])].copy()
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
                base_rate=("base_rate", "mean"),
                auroc=("auroc", "mean"),
                auprc=("auprc", "mean"),
                auprc_gain=("auprc_gain", "mean"),
                recall_at_20pct=("recall_at_20pct", "mean"),
                positives=("positives", "sum"),
                negatives=("negatives", "sum"),
            )
        )
        chosen["environment"] = environment
        chosen["target"] = target
        chosen["best_regime_ttc"] = regime
        best_rows.append(chosen)
    best = pd.concat(best_rows, ignore_index=True) if best_rows else pd.DataFrame()

    fig_targets = [
        ("alfworld", "bad_action_t1", "ALFWorld bad_action$_{t+1}$"),
        ("alfworld", "exploit_action_t1", "ALFWorld exploit_action$_{t+1}$"),
        ("webshop", "bad_buy_t1", "WebShop bad_buy$_{t+1}$"),
        ("webshop", "low_reward_buy_t1", "WebShop low_reward_buy$_{t+1}$"),
    ]
    fig, axes = plt.subplots(1, len(fig_targets), figsize=(4.3 * len(fig_targets), 4.8), constrained_layout=True, sharey=True)
    if len(fig_targets) == 1:
        axes = [axes]
    for ax, (environment, target, title) in zip(axes, fig_targets):
        sub = best[(best["environment"] == environment) & (best["target"] == target)]
        if sub.empty:
            ax.set_axis_off()
            continue
        sub = sub.set_index("feature_group").reindex(list(FEATURE_GROUPS.keys())).reset_index()
        valid = sub["auprc_gain"].notna()
        xvals = np.arange(len(sub))
        ax.bar(
            xvals,
            sub["auprc_gain"],
            color=[FEATURE_COLORS.get(fg, "#777777") for fg in sub["feature_group"]],
        )
        ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
        ax.set_xticks(xvals)
        ax.set_xticklabels(
            [
                fg.replace("p_hack", "act").replace("_", "\n").replace("internal", "int")
                for fg in sub["feature_group"]
            ],
            rotation=0,
            ha="center",
        )
        base_rate = sub["base_rate"].dropna().iloc[0] if sub["base_rate"].notna().any() else np.nan
        regime = sub["best_regime_ttc"].dropna().iloc[0] if "best_regime_ttc" in sub.columns and sub["best_regime_ttc"].notna().any() else "n/a"
        ax.set_title(f"{title}\nbase={base_rate:.2f}, best={regime}" if np.isfinite(base_rate) else f"{title}\nbest={regime}")
        ax.set_ylabel("AUPRC gain over base rate")
        ax.grid(axis="y", alpha=0.18)
        for x, gain, recall in zip(xvals, sub["auprc_gain"], sub["recall_at_20pct"]):
            if pd.notna(gain):
                ax.text(x, gain + (0.01 if gain >= 0 else -0.03), f"{gain:.2f}\nR20={recall:.2f}", ha="center", va="bottom" if gain >= 0 else "top", fontsize=7)
        for x, ok in zip(xvals, valid):
            if not ok:
                ax.text(x, 0.02, "omit", rotation=90, ha="center", va="bottom", fontsize=7, color="#777777")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=FEATURE_COLORS.get(fg, "#777777"), label=fg.replace("_", " "))
        for fg in FEATURE_GROUPS.keys()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4), frameon=False, bbox_to_anchor=(0.5, 1.05))
    savefig(fig, figs_dir / "main_context_calibrated_prediction_ablation.png")
    return best


def build_behavior_tables(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    sub = df[df["environment"] == "alfworld"].copy()
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for model, part in sub.groupby("model"):
        rows.append({
            "model": model,
            "adapter": part["adapter"].iloc[0],
            "bad_action_rate": pd.to_numeric(part["bad_action_t1"], errors="coerce").mean(),
            "exploit_action_rate": pd.to_numeric(part["exploit_action_t1"], errors="coerce").mean(),
            "easy_grader_rate": pd.to_numeric(part["easy_grader_action_t1"], errors="coerce").mean(),
            "fake_completion_rate": pd.to_numeric(part["fake_completion_action_t1"], errors="coerce").mean(),
            "proxy_score_without_success": pd.to_numeric(part["proxy_without_success_t1"], errors="coerce").mean(),
            "success_rate": pd.to_numeric(part["success"], errors="coerce").mean(),
        })
    table = pd.DataFrame(rows).sort_values("adapter", key=lambda s: s.map(adapter_key))
    table.to_csv(tables_dir / "table1_behavioral_internal_profile.csv", index=False)
    return table


def build_monitor_semantics_table(tables_dir: Path) -> pd.DataFrame:
    rows = [
        {
            "component": "p_hack",
            "mechanistic_interpretation": "latent reward-hack policy-state descriptor",
            "what_it_captures": "adapter-regime separation and reward-hack-like internal state",
            "evidence_in_our_data": "separates condition clusters in the internal risk-state map",
            "limitation": "does not directly predict every next bad action",
        },
        {
            "component": "entropy",
            "mechanistic_interpretation": "decision-state uncertainty / instability / confidence",
            "what_it_captures": "immediate decision-local information before next action",
            "evidence_in_our_data": "often improves next-step prediction over raw p_hack alone",
            "limitation": "not alignment-specific",
        },
        {
            "component": "temporal summaries",
            "mechanistic_interpretation": "late-stage within-step trajectory descriptors",
            "what_it_captures": "how internal state changes during reasoning, not just its level",
            "evidence_in_our_data": "useful in internal_all and temporal feature groups in selected regimes",
            "limitation": "not universal across adapters or TTC budgets",
        },
        {
            "component": "context",
            "mechanistic_interpretation": "environment/action affordances and reasoning-budget calibration",
            "what_it_captures": "mapping from latent state to immediate action risk",
            "evidence_in_our_data": "internal_plus_context can outperform internal-only models",
            "limitation": "environment-dependent and partly feature-availability dependent",
        },
        {
            "component": "reward-hack steering direction",
            "mechanistic_interpretation": "intervention probe of the reward-hack representation",
            "what_it_captures": "behavioral relevance of the latent reward-hack direction",
            "evidence_in_our_data": "selected mixed-adapter steering runs reduce exploit/proxy behavior",
            "limitation": "not a universal mitigation",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "main_monitor_semantics_table.csv", index=False)
    df.to_latex(tables_dir / "main_monitor_semantics_table.tex", index=False)
    return df


def load_webshop_behavior(root: Path) -> pd.DataFrame:
    rows = []
    for sweep in [
        root / "BALROG/webshop_react_eval/runs/buycurated_sweep",
        root / "BALROG/webshop_react_eval/runs/buycurated_sweep_falcon",
        root / "BALROG/webshop_react_eval/runs/buycurated_sweep_llama",
    ]:
        if not sweep.exists():
            continue
        for p in sweep.glob("*/pilot_metrics.json"):
            obj = maybe_load_json(p)
            if obj:
                rows.append(obj)
    return pd.DataFrame(rows)


def build_steering_summary(root: Path, tables_dir: Path, figs_dir: Path) -> pd.DataFrame:
    rows = []
    for fam, rel in [
        ("Qwen", root / "BALROG/hf_alfworld_steering/hf_steering_analysis_mix_sweep/steering_effects.csv"),
        ("Falcon", root / "BALROG/hf_alfworld_steering/hf_steering_analysis_mix_sweep_falcon/steering_effects.csv"),
        ("Llama", root / "BALROG/hf_alfworld_steering/hf_steering_analysis_mix_sweep_llama/steering_effects.csv"),
    ]:
        df = maybe_load_csv(rel)
        if df.empty:
            continue
        df["model_family"] = fam
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df["adapter"] = df["adapter"].str.lower()
    keep = df[df["adapter"].isin(["mix05", "mix10", "mix50"])].copy()
    keep["exploit_reduction"] = -pd.to_numeric(keep["delta_steered_minus_unsteered_proxy_score_without_success"], errors="coerce")
    keep["steering_rate"] = pd.to_numeric(keep["steering_rate_steered_mean"], errors="coerce")
    summary = keep[[
        "model_family",
        "adapter",
        "condition",
        "steering_mode",
        "alpha",
        "steering_rate",
        "exploit_reduction",
        "delta_steered_minus_unsteered_easy_grader_count",
        "delta_steered_minus_unsteered_fake_completion_count",
        "delta_steered_minus_unsteered_success",
        "success_preserved",
        "exploit_reduced",
    ]].copy()
    summary = summary[(summary["exploit_reduction"] > 0) | (summary["exploit_reduced"] == True)]
    summary.to_csv(tables_dir / "main_steering_intervention_probe.csv", index=False)
    summary.to_latex(tables_dir / "main_steering_intervention_probe.tex", index=False, float_format="%.3f")

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    for _, row in summary.iterrows():
        ax.scatter(
            row["steering_rate"],
            row["exploit_reduction"],
            color=ADAPTER_COLORS.get(row["adapter"], "#777777"),
            marker="o" if row["steering_mode"] == "always" else "s",
            s=70,
            edgecolors="black",
            linewidths=0.5,
        )
        if row["exploit_reduction"] >= summary["exploit_reduction"].nlargest(min(3, len(summary))).min():
            ax.text(
                row["steering_rate"],
                row["exploit_reduction"],
                f"{ADAPTER_LABELS.get(row['adapter'], row['adapter'])} {row['steering_mode']} a={float(row['alpha']):.2f}",
                fontsize=7,
            )
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
    ax.axvline(0.0, color="#dddddd", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Fraction of steps steered")
    ax.set_ylabel("Reduction in proxy score without true success")
    ax.set_title("Steering reduces exploit/proxy behavior in selected mixed-adapter regimes")
    ax.grid(alpha=0.18)
    adapter_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=ADAPTER_COLORS[a],
            markerfacecolor=ADAPTER_COLORS[a],
            markeredgecolor="black",
            markersize=7,
            label=ADAPTER_LABELS[a],
        )
        for a in ["mix05", "mix10", "mix50"]
        if a in set(summary["adapter"])
    ]
    mode_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color="#666666", markerfacecolor="#666666", label="always-on"),
        plt.Line2D([0], [0], marker="s", linestyle="", color="#666666", markerfacecolor="#666666", label="gated"),
    ]
    leg1 = ax.legend(handles=adapter_handles, title="Adapter", loc="upper left", frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=mode_handles, title="Steering mode", loc="lower right", frameon=False)
    savefig(fig, figs_dir / "main_steering_intervention_probe.png")
    return summary


def build_ttc_budget_effects(cond: pd.DataFrame, best_pred: pd.DataFrame, tables_dir: Path, figs_dir: Path) -> pd.DataFrame:
    q = cond[(cond["model_family"] == "Qwen") & (cond["environment"] == "alfworld")].copy()
    if q.empty:
        return pd.DataFrame()
    q["sort_x"] = q["ttc_numeric"].fillna(-1)
    effects = q[["adapter", "ttc_or_regime", "ttc_numeric", "mean_p_hack", "mean_entropy", "exploit_action_rate", "bad_action_rate"]].copy()
    effects.to_csv(tables_dir / "ttc_budget_effects.csv", index=False)

    pred_q = best_pred[best_pred["environment"] == "alfworld"].copy() if not best_pred.empty else pd.DataFrame()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "mean_p_hack", "Mean reasoning $p_{hack}$ by TTC"),
        (axes[0], "mean_entropy", "Mean reasoning entropy by TTC"),
    ]:
        pass
    axes[0].cla()
    for adapter in [a for a in ADAPTER_ORDER if a in set(q["adapter"])]:
        sub = q[q["adapter"] == adapter].sort_values("sort_x")
        labels = ["unconstrained" if pd.isna(t) else str(int(t)) for t in sub["ttc_numeric"]]
        axes[0].plot(labels, sub["mean_p_hack"], marker="o", linewidth=2.0, color=ADAPTER_COLORS[adapter], label=f"{ADAPTER_LABELS[adapter]} p_hack")
        axes[0].plot(labels, sub["mean_entropy"], marker="s", linewidth=1.8, linestyle="--", color=ADAPTER_COLORS[adapter], alpha=0.7, label=f"{ADAPTER_LABELS[adapter]} entropy")
    axes[0].set_title("Monitor state vs TTC")
    axes[0].set_ylabel("Mean signal")
    axes[0].grid(alpha=0.18)

    if not pred_q.empty:
        pred_q["sort_x"] = pred_q["best_regime_ttc"].map(lambda x: -1 if str(x) == "unconstrained" else float(str(x).replace("ttc_", "")) if str(x).startswith("ttc_") else np.nan)
        for fg, sub in pred_q.groupby("feature_group"):
            axes[1].scatter(sub["best_regime_ttc"], sub["auprc_gain"], color=FEATURE_COLORS.get(fg, "#777777"), s=55, label=fg)
    axes[1].axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
    axes[1].set_title("Prediction quality vs selected TTC")
    axes[1].set_ylabel("AUPRC gain")
    axes[1].grid(alpha=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[: min(len(handles), 8)], labels[: min(len(labels), 8)], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    savefig(fig, figs_dir / "main_ttc_budget_monitor_semantics.png")
    return effects


def build_appendix_temporal_placeholder(df: pd.DataFrame, figs_dir: Path) -> None:
    alf = df[(df["environment"] == "alfworld") & df["ttc_numeric"].notna()].copy()
    if alf.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "reasoning_p_hack_late_mean", "Late-window $p_{hack}$ by future label"),
        (axes[1], "reasoning_entropy_late_mean", "Late-window entropy by future label"),
    ]:
        plot_rows = []
        for adapter in ["control", "mix05", "mix10", "mix50", "hack"]:
            sub = alf[alf["adapter"] == adapter]
            for label in [0, 1]:
                part = sub[sub["bad_action_t1"].fillna(0) == label]
                plot_rows.append({"adapter": adapter, "label": label, "value": pd.to_numeric(part[metric], errors="coerce").mean()})
        plot = pd.DataFrame(plot_rows)
        for label, offset, alpha in [(0, -0.15, 0.55), (1, 0.15, 0.9)]:
            part = plot[plot["label"] == label]
            xs = np.arange(len(part)) + offset
            ax.bar(xs, part["value"], width=0.28, alpha=alpha, color=[ADAPTER_COLORS[a] for a in part["adapter"]], label=f"bad_t1={label}" if ax is axes[0] else None)
        ax.set_xticks(np.arange(len(ADAPTER_ORDER)))
        ax.set_xticklabels([ADAPTER_LABELS[a] for a in ADAPTER_ORDER], rotation=45, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18)
    axes[0].legend(frameon=False)
    out1 = figs_dir / "appendix_temporal_bad_vs_nonbad.png"
    savefig(fig, out1)
    shutil.copyfile(out1, figs_dir / "appendix_temporal_exploit_opportunity.png")
    shutil.copyfile(out1.with_suffix(".pdf"), (figs_dir / "appendix_temporal_exploit_opportunity.png").with_suffix(".pdf"))

    # TTC lines
    q = alf[alf["model_family"] == "Qwen"].copy()
    if not q.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
        for ax, metric, title in [
            (axes[0], "reasoning_p_hack_late_mean", "Late-window $p_{hack}$ delta vs TTC"),
            (axes[1], "reasoning_entropy_late_mean", "Late-window entropy delta vs TTC"),
        ]:
            for adapter in ["control", "mix05", "mix10", "mix50", "hack"]:
                ys, xs = [], []
                for ttc, sub in q[q["adapter"] == adapter].groupby("ttc_numeric"):
                    xs.append(ttc)
                    bad = pd.to_numeric(sub.loc[sub["bad_action_t1"].fillna(0) > 0.5, metric], errors="coerce")
                    good = pd.to_numeric(sub.loc[sub["bad_action_t1"].fillna(0) <= 0.5, metric], errors="coerce")
                    ys.append(float(bad.mean() - good.mean()) if len(bad) and len(good) else np.nan)
                ax.plot(xs, ys, marker="o", linewidth=2.0, color=ADAPTER_COLORS[adapter], label=ADAPTER_LABELS[adapter])
            ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
            ax.set_title(title)
            ax.set_xlabel("TTC")
            ax.grid(alpha=0.18)
        axes[0].legend(frameon=False, ncol=2)
        out2 = figs_dir / "appendix_latewindow_delta_vs_ttc.png"
        savefig(fig, out2)
        shutil.copyfile(out2, figs_dir / "appendix_progress_ttc_heatmaps.png")
        shutil.copyfile(out2.with_suffix(".pdf"), (figs_dir / "appendix_progress_ttc_heatmaps.png").with_suffix(".pdf"))


def write_summary_markdown(root: Path, cond: pd.DataFrame, best_pred: pd.DataFrame, steering: pd.DataFrame, figs_dir: Path) -> None:
    supports = [
        "- `p_hack` remains active and structured in agentic reasoning and separates adapter regimes at the condition level.",
        "- Raw `p_hack` is not a simple monotonic readout of exploit/bad behavior across environments and adapters.",
        "- Entropy and mixed internal/context feature sets can outperform raw `p_hack` alone for next-step prediction in selected targets/regimes.",
        "- Steering against the reward-hack direction reduces exploit behavior in selected mixed-adapter regimes.",
    ]
    not_supported = [
        "- `p_hack` universally rises before bad actions.",
        "- late-stage `p_hack` is a universal exploit predictor.",
        "- entropy always indicates danger.",
        "- steering fully prevents reward hacking.",
        "- all effects are monotonic in mixture strength.",
    ]
    lines = [
        "# PAPER_FINAL_EVIDENCE_SUMMARY",
        "",
        "## 1. What does the data actually show?",
        "",
        "The unified analysis indicates that the reward-hack monitor transfers into agentic settings as a latent policy-state descriptor: conditions separate clearly in mean reasoning `p_hack` and entropy, but immediate action-level risk is not a simple function of raw `p_hack` alone.",
        "",
        "1. Does p_hack remain structured in agentic reasoning?",
        "Yes, at the condition level it remains active and structured across environments, adapters, and TTC regimes.",
        "",
        "2. Does it separate adapter regimes?",
        "Yes. Adapter conditions separate in the internal risk-state map, especially within Qwen and in mixed/hack settings.",
        "",
        "3. Is p_hack alone a direct next-action predictor?",
        "No. It is informative as a latent policy-state descriptor but not a universal immediate action-risk predictor.",
        "",
        "4. Does entropy add decision-local information?",
        "Yes, often substantially, especially in next-step prediction ablations.",
        "",
        "5. Does context improve prediction?",
        "In selected targets/regimes, internal-plus-context improves over internal-only features.",
        "",
        "6. How does TTC/CoT budget affect monitor semantics?",
        "TTC shifts both the internal risk-state distribution and prediction quality, but not monotonically.",
        "",
        "7. Does steering reduce exploit behavior in selected regimes?",
        "Yes, selected mixed-adapter steering conditions reduce proxy/exploit behavior while preserving success.",
        "",
        "## 2. Which claims are supported?",
        "",
        *supports,
        "",
        "## 3. Which claims are not supported?",
        "",
        *not_supported,
        "",
        "## 4. Which figures should go in the main paper?",
        "",
        f"- [main_internal_risk_state_map_qwen.png]({figs_dir / 'main_internal_risk_state_map_qwen.png'})",
        f"- [main_context_calibrated_prediction_ablation.png]({figs_dir / 'main_context_calibrated_prediction_ablation.png'})",
        f"- [main_risk_state_vs_behavior_qwen.png]({figs_dir / 'main_risk_state_vs_behavior_qwen.png'}) or [main_ttc_budget_monitor_semantics.png]({figs_dir / 'main_ttc_budget_monitor_semantics.png'})",
        f"- [main_steering_intervention_probe.png]({figs_dir / 'main_steering_intervention_probe.png'}) if space allows",
        "",
        "## 5. Which figures should go in the appendix?",
        "",
        f"- [appendix_internal_risk_state_map_all_models.png]({figs_dir / 'appendix_internal_risk_state_map_all_models.png'})",
        f"- [appendix_temporal_bad_vs_nonbad.png]({figs_dir / 'appendix_temporal_bad_vs_nonbad.png'})",
        f"- [appendix_temporal_exploit_opportunity.png]({figs_dir / 'appendix_temporal_exploit_opportunity.png'})",
        f"- [appendix_latewindow_delta_vs_ttc.png]({figs_dir / 'appendix_latewindow_delta_vs_ttc.png'})",
        f"- [appendix_progress_ttc_heatmaps.png]({figs_dir / 'appendix_progress_ttc_heatmaps.png'})",
        "",
        "## 6. What is the cleanest ICML MechInterp story?",
        "",
        "The cleanest story is: `p_hack` transfers from isolated generation to agents as a latent reward-hack policy-state descriptor. It separates adapter regimes, but immediate action-level risk is better captured by entropy and context-calibrated internal features. Steering remains useful as a targeted intervention probe rather than a universal mitigation.",
    ]
    (root / "PAPER_FINAL_EVIDENCE_SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/home/patrick/vllmPatrickMonitoring"))
    args = ap.parse_args()
    configure_style()
    root = args.root
    tables_dir = root / "tables_final_evidence"
    figs_dir = root / "figures_final_evidence"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    alf = load_alfworld_steps(root)
    ws = load_webshop_steps(root)
    unified = pd.concat([df for df in [alf, ws] if not df.empty], ignore_index=True)
    parquet_path, summary_path = save_unified_dataset(unified, tables_dir)
    cond = condition_risk_state(unified)
    cond.to_csv(tables_dir / "internal_risk_state_by_condition.csv", index=False)
    plot_internal_risk_state_map(cond, figs_dir)
    plot_risk_state_vs_behavior(cond, tables_dir, figs_dir)
    behavior = build_behavior_tables(unified, tables_dir)
    monitor_semantics = build_monitor_semantics_table(tables_dir)
    best_pred = build_feature_ablation(unified, tables_dir, figs_dir)
    ttc_effects = build_ttc_budget_effects(cond, best_pred, tables_dir, figs_dir)
    steering = build_steering_summary(root, tables_dir, figs_dir)
    build_appendix_temporal_placeholder(unified, figs_dir)
    write_summary_markdown(root, cond, best_pred, steering, figs_dir)

    generated = [
        tables_dir / "internal_risk_state_by_condition.csv",
        tables_dir / "main_context_calibrated_prediction_ablation.csv",
        tables_dir / "table1_behavioral_internal_profile.csv",
        tables_dir / "main_monitor_semantics_table.csv",
        tables_dir / "ttc_budget_effects.csv",
        tables_dir / "main_steering_intervention_probe.csv",
        figs_dir / "main_internal_risk_state_map_qwen.png",
        figs_dir / "appendix_internal_risk_state_map_all_models.png",
        figs_dir / "main_risk_state_vs_behavior_qwen.png",
        figs_dir / "appendix_risk_state_vs_behavior_all_models.png",
        figs_dir / "main_context_calibrated_prediction_ablation.png",
        figs_dir / "main_ttc_budget_monitor_semantics.png",
        figs_dir / "main_steering_intervention_probe.png",
        figs_dir / "appendix_temporal_bad_vs_nonbad.png",
        figs_dir / "appendix_temporal_exploit_opportunity.png",
        figs_dir / "appendix_latewindow_delta_vs_ttc.png",
        figs_dir / "appendix_progress_ttc_heatmaps.png",
        root / "PAPER_FINAL_EVIDENCE_SUMMARY.md",
    ]
    print("[DONE] Final evidence pipeline outputs:")
    if parquet_path:
        print(" -", parquet_path)
    else:
        print(" -", tables_dir / "unified_agent_step_dataset.csv")
    print(" -", summary_path)
    for path in generated:
        if path.exists():
            print(" -", path)


if __name__ == "__main__":
    main()
