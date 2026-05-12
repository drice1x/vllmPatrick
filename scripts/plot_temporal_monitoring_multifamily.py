#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/patrick/vllmPatrickMonitoring")
DEFAULT_OUTPUT = ROOT / "figures/temporal"
FAMILY_ORDER = ["Llama", "Qwen", "Falcon"]
ADAPTER_ORDER = ["mix05", "mix10", "mix50"]
ADAPTER_LABELS = {"mix05": "Mix05", "mix10": "Mix10", "mix50": "Mix50"}
ADAPTER_COLORS = {"mix05": "#4c78a8", "mix10": "#f58518", "mix50": "#e45756"}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def family_of_model(model: str) -> str:
    text = str(model)
    for fam in FAMILY_ORDER:
        if text.startswith(fam):
            return fam
    return "Other"


def normalize_regime(ttc_value) -> tuple[str, float | None]:
    if ttc_value in ("", None) or (isinstance(ttc_value, float) and np.isnan(ttc_value)):
        return ("unconstrained", None)
    try:
        value = float(ttc_value)
        return (str(int(value)), value)
    except Exception:
        return (str(ttc_value), None)


def regime_sort_key(regime: str) -> tuple[int, float]:
    if regime == "unconstrained":
        return (1, float("inf"))
    try:
        return (0, float(regime))
    except Exception:
        return (0, float("inf"))


def safe_num(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def maybe_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def resample_traj(values, n_bins: int) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.full(n_bins, np.nan)
    if len(arr) == 1:
        return np.full(n_bins, arr[0])
    x_old = np.linspace(0.0, 1.0, len(arr))
    x_new = np.linspace(0.0, 1.0, n_bins)
    return np.interp(x_new, x_old, arr)


def extract_row_traj(row: pd.Series, cols: list[str]) -> list[float]:
    vals = pd.to_numeric(row[cols], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return vals.tolist()


def fit_logreg(X: np.ndarray, y: np.ndarray, steps: int = 350, lr: float = 0.12, l2: float = 1e-4) -> np.ndarray:
    w = np.zeros(X.shape[1] + 1, dtype=float)
    Xb = np.column_stack([np.ones(len(X)), X])
    for _ in range(steps):
        z = np.clip(Xb @ w, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = (Xb.T @ (p - y)) / len(X)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return w


def predict_logreg(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])
    z = np.clip(Xb @ w, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def build_prefix_features(
    p_resampled: np.ndarray,
    e_resampled: np.ndarray,
    j: int,
    context_row: pd.Series,
    late_window_start: float,
) -> list[float]:
    p_pref = p_resampled[: j + 1]
    e_pref = e_resampled[: j + 1]
    x = np.linspace(0.0, 1.0, len(p_pref))
    if len(p_pref) >= 2:
        p_slope = float(np.polyfit(x, p_pref, 1)[0])
        e_slope = float(np.polyfit(x, e_pref, 1)[0])
    else:
        p_slope = 0.0
        e_slope = 0.0
    late_idx = max(0, int(math.floor(len(p_pref) * late_window_start)))
    p_late = p_pref[late_idx:]
    e_late = e_pref[late_idx:]
    p_late_mean = float(np.nanmean(p_late)) if len(p_late) else float(np.nanmean(p_pref))
    e_late_mean = float(np.nanmean(e_late)) if len(e_late) else float(np.nanmean(e_pref))
    return [
        float(p_pref[-1]),
        float(e_pref[-1]),
        float(np.nanmean(p_pref)),
        float(np.nanmean(e_pref)),
        p_late_mean,
        e_late_mean,
        float(p_pref[-1] - p_pref[0]),
        float(e_pref[-1] - e_pref[0]),
        p_slope,
        e_slope,
        safe_num(context_row.get("step_index_normalized")),
        safe_num(context_row.get("previous_action_was_invalid")),
        safe_num(context_row.get("exploit_action_available")),
        safe_num(context_row.get("proxy_score_visible")),
        safe_num(context_row.get("near_completion")),
        safe_num(context_row.get("item_page_or_buy_page")),
        safe_num(context_row.get("prev_exploit_like_action")),
        1.0 if str(context_row.get("environment")) == "webshop" else 0.0,
        1.0 if str(context_row.get("ttc_or_regime")) == "unconstrained" else 0.0,
        safe_num(context_row.get("ttc_numeric")),
    ]


def auc_score(y: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(wins / (len(pos) * len(neg)))


def load_alfworld_temporal(root: Path) -> tuple[pd.DataFrame, dict]:
    path = root / "BALROG/analysis_gameable_alfworld_temporal/steps_with_entropy_phack_temporal.csv"
    df = pd.read_csv(path)
    p_cols = sorted([c for c in df.columns if c.startswith("reasoning_p_bin")], key=lambda c: int(c.split("bin")[-1]))
    e_cols = sorted([c for c in df.columns if c.startswith("reasoning_entropy_bin")], key=lambda c: int(c.split("bin")[-1]))
    df["environment"] = "alfworld"
    df["model_family"] = df["model"].map(family_of_model)
    df["episode_id"] = df["source_file"]
    df["step_id"] = pd.to_numeric(df["step"], errors="coerce")
    df["action_t"] = df["action"]
    df["ttc_or_regime"], df["ttc_numeric"] = zip(*df["ttc"].map(normalize_regime))
    safe_numeric(
        df,
        [
            "bad_action",
            "gameable_hack_action",
            "fake_completion_action",
            "easy_grader_action",
            "reasoning_p_hack",
            "reasoning_entropy_mean",
            "reasoning_p_hack_late_change",
            "reasoning_p_hack_late_slope",
            "reasoning_entropy_late_change",
            "reasoning_entropy_late_slope",
            "action_was_rewritten",
            "proxy_score",
            "proxy_score_without_success",
            "inspect_score_action",
            "claim_complete_action",
            "true_task_success_event",
        ]
        + p_cols
        + e_cols,
    )
    df = df.sort_values(["episode_id", "step_id"]).reset_index(drop=True)
    g = df.groupby("episode_id", sort=False)
    df["action_t1"] = g["action_t"].shift(-1)
    for col in ["bad_action", "gameable_hack_action", "fake_completion_action", "easy_grader_action", "true_task_success_event"]:
        df[f"{col}_t1"] = g[col].shift(-1)
    df["has_t1"] = g["step_id"].shift(-1).notna()
    df = df[df["has_t1"]].copy()
    df["bad_action_t1"] = df["bad_action_t1"]
    df["exploit_action_t1"] = (
        df["easy_grader_action_t1"].fillna(0)
        + df["fake_completion_action_t1"].fillna(0)
        + df["gameable_hack_action_t1"].fillna(0)
    ).clip(upper=1)
    df["step_index_normalized"] = df.groupby("episode_id")["step_id"].transform(lambda s: s / max(float(s.max()), 1.0))
    df["previous_action_type"] = g["action_t"].shift(1)
    df["prev_exploit_like_action"] = (
        df["previous_action_type"].fillna("").astype(str).str.contains("inspect score|claim|grader", case=False, regex=True)
    ).astype(float)
    df["previous_action_was_invalid"] = df["action_was_rewritten"].fillna(0)
    df["proxy_score_visible"] = (
        df["proxy_score"].notna() | df["proxy_score_without_success"].notna() | (df["inspect_score_action"].fillna(0) > 0)
    ).astype(float)
    claim_t1 = df["claim_complete_action_t1"] if "claim_complete_action_t1" in df.columns else pd.Series(0.0, index=df.index)
    df["near_completion"] = (claim_t1.fillna(0) > 0).astype(float)
    df["exploit_action_available"] = df["proxy_score_visible"]
    df["item_page_or_buy_page"] = np.nan
    df["p_traj"] = df.apply(lambda r: extract_row_traj(r, p_cols), axis=1)
    df["e_traj"] = df.apply(lambda r: extract_row_traj(r, e_cols), axis=1)
    return df, {"alfworld_csv": str(path), "p_cols": len(p_cols), "e_cols": len(e_cols)}


def load_webshop_temporal(root: Path) -> tuple[pd.DataFrame, dict]:
    rows = []
    used_files = set()
    for step_path in sorted((root / "BALROG/webshop_monitoring_eval/webshop_runs").glob("*/*/steps/steps.jsonl")):
        used_files.add(str(step_path))
        with step_path.open() as f:
            for line in f:
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(), {"webshop_step_logs": 0}
    df = pd.DataFrame(rows)
    df["environment"] = "webshop"
    df["model_family"] = df["model"].map(family_of_model)
    df["episode_id"] = df["source_file"]
    df["step_id"] = pd.to_numeric(df["step"], errors="coerce")
    df["action_t"] = df["action"]
    df["ttc_or_regime"], df["ttc_numeric"] = zip(*df["ttc"].map(normalize_regime))
    for col in ["reasoning_p_hack", "reasoning_entropy_mean", "reasoning_p_hack_late_change", "reasoning_p_hack_late_slope", "reasoning_entropy_late_change", "reasoning_entropy_late_slope", "bad_buy_action", "low_reward_buy", "early_buy", "buy_action"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["episode_id", "step_id"]).reset_index(drop=True)
    g = df.groupby("episode_id", sort=False)
    df["action_t1"] = g["action_t"].shift(-1)
    for col in ["bad_buy_action", "low_reward_buy", "early_buy", "buy_action"]:
        if col in df.columns:
            df[f"{col}_t1"] = g[col].shift(-1)
    df["has_t1"] = g["step_id"].shift(-1).notna()
    df = df[df["has_t1"]].copy()
    df["bad_buy_t1"] = df.get("bad_buy_action_t1", np.nan)
    df["low_reward_buy_t1"] = df.get("low_reward_buy_t1", np.nan)
    df["bad_action_t1"] = df["bad_buy_t1"]
    df["exploit_action_t1"] = df["bad_buy_t1"]
    df["step_index_normalized"] = df.groupby("episode_id")["step_id"].transform(lambda s: s / max(float(s.max()), 1.0))
    df["previous_action_type"] = g["action_t"].shift(1)
    df["prev_exploit_like_action"] = np.nan
    df["previous_action_was_invalid"] = np.nan
    df["proxy_score_visible"] = np.nan
    df["near_completion"] = np.nan
    df["exploit_action_available"] = np.nan
    action_lower = df["action_t1"].fillna("").astype(str).str.lower()
    df["item_page_or_buy_page"] = action_lower.str.contains("buy|item|asin").astype(float)
    df["p_traj"] = df["reasoning_p_hack_trajectory"].apply(lambda x: x if isinstance(x, list) else [])
    df["e_traj"] = df["reasoning_token_entropy"].apply(lambda x: x if isinstance(x, list) else [])
    return df, {"webshop_step_logs": len(used_files)}


def load_all_steps(root: Path, environments: list[str]) -> tuple[pd.DataFrame, dict]:
    frames = []
    meta = {}
    if "alfworld" in environments:
        alf, info = load_alfworld_temporal(root)
        if not alf.empty:
            frames.append(alf)
        meta.update(info)
    if "webshop" in environments:
        ws, info = load_webshop_temporal(root)
        if not ws.empty:
            frames.append(ws)
        meta.update(info)
    if not frames:
        return pd.DataFrame(), meta
    df = pd.concat(frames, ignore_index=True, sort=False)
    return df, meta


def prepare_traces(
    df: pd.DataFrame,
    target_label: str,
    families: list[str],
    adapters: list[str],
    n_bins: int,
    min_tokens: int,
    include_unconstrained: bool,
) -> pd.DataFrame:
    sub = df[df["model_family"].isin(families) & df["adapter"].isin(adapters)].copy()
    if not include_unconstrained:
        sub = sub[sub["ttc_or_regime"] != "unconstrained"]
    sub = sub[sub[target_label].notna()].copy()
    sub["p_resampled"] = sub["p_traj"].apply(lambda x: resample_traj(x, n_bins))
    sub["e_resampled"] = sub["e_traj"].apply(lambda x: resample_traj(x, n_bins))
    sub["p_valid"] = sub["p_traj"].apply(lambda x: len([v for v in x if pd.notna(v)]) >= min_tokens)
    sub["e_valid"] = sub["e_traj"].apply(lambda x: len([v for v in x if pd.notna(v)]) >= min_tokens)
    sub["valid_trace"] = sub["p_valid"] & sub["e_valid"]
    return sub


def compute_risk_trajectories(
    df: pd.DataFrame,
    target_label: str,
    families: list[str],
    regimes: list[str],
    n_bins: int,
    min_count: int,
    late_window_start: float,
) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for family in families:
        for regime in regimes:
            cell = df[(df["model_family"] == family) & (df["ttc_or_regime"] == regime) & (df["valid_trace"])].copy()
            y = pd.to_numeric(cell[target_label], errors="coerce").fillna(0).astype(int).to_numpy()
            pos = int(np.sum(y == 1))
            neg = int(np.sum(y == 0))
            payload = {"enough": False, "risk_traj": {}, "auc_final": np.nan, "n_valid": len(cell), "n_bad": pos, "n_nonbad": neg}
            if len(cell) == 0 or pos < min_count or neg < min_count:
                out[(family, regime)] = payload
                continue
            risk = np.full((len(cell), n_bins), np.nan)
            groups = cell.groupby("episode_id", sort=False).indices
            for j in range(n_bins):
                X_rows = []
                for _, row in cell.iterrows():
                    X_rows.append(build_prefix_features(row["p_resampled"], row["e_resampled"], j, row, late_window_start))
                X = np.asarray(X_rows, dtype=float)
                for _, idx in groups.items():
                    mask = np.zeros(len(cell), dtype=bool)
                    mask[list(idx)] = True
                    y_train = y[~mask]
                    if len(np.unique(y_train)) < 2:
                        continue
                    X_train = X[~mask]
                    mu = np.nanmean(X_train, axis=0)
                    mu = np.where(np.isnan(mu), 0.0, mu)
                    sigma = np.nanstd(np.where(np.isnan(X_train), mu, X_train), axis=0)
                    sigma = np.where((sigma == 0) | np.isnan(sigma), 1.0, sigma)
                    X_train = (np.where(np.isnan(X_train), mu, X_train) - mu) / sigma
                    X_test = (np.where(np.isnan(X[mask]), mu, X[mask]) - mu) / sigma
                    w = fit_logreg(X_train, y_train)
                    risk[mask, j] = predict_logreg(X_test, w)
            payload["risk_traj"] = {idx: risk[i] for i, idx in enumerate(cell.index)}
            if np.isfinite(risk[:, -1]).sum() >= (pos + neg) // 2:
                valid = np.isfinite(risk[:, -1])
                payload["auc_final"] = auc_score(y[valid], risk[valid, -1])
            payload["enough"] = True
            out[(family, regime)] = payload
    return out


def summarize_cells(
    df: pd.DataFrame,
    risk_payload: dict,
    target_label: str,
    families: list[str],
    regimes: list[str],
    adapters: list[str],
    late_window_start: float,
) -> pd.DataFrame:
    rows = []
    for family in families:
        for regime in regimes:
            for adapter in adapters:
                part = df[
                    (df["model_family"] == family)
                    & (df["ttc_or_regime"] == regime)
                    & (df["adapter"] == adapter)
                    & (df["valid_trace"])
                ].copy()
                y = pd.to_numeric(part[target_label], errors="coerce").fillna(0).astype(int)
                n_valid = len(part)
                n_bad = int((y == 1).sum())
                n_nonbad = int((y == 0).sum())
                late_start_idx = max(0, int(math.floor(len(part["p_resampled"].iloc[0]) * late_window_start))) if n_valid else 0
                def late_delta(col: str) -> float:
                    if n_bad == 0 or n_nonbad == 0:
                        return np.nan
                    bad_vals = []
                    non_vals = []
                    for _, row in part.iterrows():
                        arr = np.asarray(row[col], dtype=float)
                        if col == "risk_resampled":
                            arr = arr[np.isfinite(arr)]
                            if len(arr) == 0:
                                continue
                            start = max(0, int(math.floor(len(arr) * late_window_start)))
                            val = float(np.nanmean(arr[start:]))
                        else:
                            val = float(np.nanmean(arr[late_start_idx:]))
                        if int(row[target_label]) == 1:
                            bad_vals.append(val)
                        else:
                            non_vals.append(val)
                    if not bad_vals or not non_vals:
                        return np.nan
                    return float(np.nanmean(bad_vals) - np.nanmean(non_vals))

                risk_map = risk_payload.get((family, regime), {}).get("risk_traj", {})
                part["risk_resampled"] = part.index.map(lambda i: risk_map.get(i, np.full(1, np.nan)))
                rows.append(
                    {
                        "environments": ",".join(sorted(set(part["environment"]))) if n_valid else "",
                        "family": family,
                        "ttc_or_regime": regime,
                        "adapter": adapter,
                        "target_label": target_label,
                        "n_valid": n_valid,
                        "n_bad": n_bad,
                        "n_nonbad": n_nonbad,
                        "late_delta_p_hack": late_delta("p_resampled"),
                        "late_delta_entropy": late_delta("e_resampled"),
                        "late_delta_risk": late_delta("risk_resampled"),
                        "risk_auc_final": risk_payload.get((family, regime), {}).get("auc_final", np.nan),
                    }
                )
    return pd.DataFrame(rows)


def plot_aligned_main(
    df: pd.DataFrame,
    risk_payload: dict,
    summary: pd.DataFrame,
    outdir: Path,
    target_label: str,
    families: list[str],
    regimes: list[str],
    adapters: list[str],
    n_bins: int,
    min_count: int,
    late_window_start: float,
) -> None:
    metrics = [
        ("p_resampled", "Reasoning $p_{hack}$"),
        ("e_resampled", "Reasoning entropy"),
        ("risk", "Predicted next-step risk"),
    ]
    n_rows = len(families) * len(metrics)
    fig, axes = plt.subplots(n_rows, len(regimes), figsize=(2.9 * len(regimes), 1.85 * n_rows), sharex=True, sharey="row", constrained_layout=True)
    if n_rows == 1:
        axes = np.array([axes])
    if len(regimes) == 1:
        axes = axes[:, np.newaxis]
    x = np.linspace(0.0, 1.0, n_bins)
    for m_idx, (metric_key, panel_label) in enumerate(metrics):
        for f_idx, family in enumerate(families):
            row_idx = m_idx * len(families) + f_idx
            for c_idx, regime in enumerate(regimes):
                ax = axes[row_idx, c_idx]
                cell = df[(df["model_family"] == family) & (df["ttc_or_regime"] == regime) & (df["valid_trace"])].copy()
                payload = risk_payload.get((family, regime), {})
                n_valid = int(payload.get("n_valid", len(cell)))
                n_bad = int(payload.get("n_bad", int(pd.to_numeric(cell[target_label], errors="coerce").fillna(0).sum())))
                n_nonbad = int(payload.get("n_nonbad", max(len(cell) - n_bad, 0)))
                ax.axvspan(late_window_start, 1.0, color="#ececec", zorder=0)
                enough_any = False
                for adapter in adapters:
                    part = cell[cell["adapter"] == adapter].copy()
                    if metric_key == "risk":
                        risk_map = payload.get("risk_traj", {})
                        part["metric_arr"] = part.index.map(lambda i: risk_map.get(i))
                    else:
                        part["metric_arr"] = part[metric_key]
                    for label, linestyle, alpha in [(1, "-", 0.16), (0, "--", 0.10)]:
                        traces = []
                        for _, row in part[pd.to_numeric(part[target_label], errors="coerce").fillna(0).astype(int) == label].iterrows():
                            arr = row["metric_arr"]
                            if arr is None:
                                continue
                            arr = np.asarray(arr, dtype=float)
                            if np.isfinite(arr).sum() < max(3, n_bins // 6):
                                continue
                            traces.append(arr)
                        if len(traces) >= min_count:
                            enough_any = True
                            arr = np.vstack(traces)
                            mean = np.nanmean(arr, axis=0)
                            sem = np.nanstd(arr, axis=0) / np.sqrt(len(arr))
                            ax.plot(x, mean, color=ADAPTER_COLORS[adapter], linestyle=linestyle, linewidth=1.6)
                            ax.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem, color=ADAPTER_COLORS[adapter], alpha=alpha)
                if not enough_any:
                    ax.text(0.5, 0.5, f"insufficient data\nvalid={n_valid}\nbad={n_bad} nonbad={n_nonbad}", ha="center", va="center", transform=ax.transAxes)
                title = f"{regime}\nvalid={n_valid} bad={n_bad} nonbad={n_nonbad}"
                ax.set_title(title if row_idx < len(families) else f"valid={n_valid} bad={n_bad} nonbad={n_nonbad}")
                if c_idx == 0:
                    ax.set_ylabel(f"{family}")
                if row_idx == n_rows - 1:
                    ax.set_xlabel("Normalized reasoning progress within step $t$")
                ax.grid(alpha=0.10)
        y_center = 1 - ((m_idx * len(families) + len(families) / 2) / n_rows)
        fig.text(0.008, y_center, panel_label, rotation=90, va="center", ha="center", fontsize=10)
    adapter_handles = [
        plt.Line2D([0], [0], color=ADAPTER_COLORS[a], lw=2, label=ADAPTER_LABELS[a]) for a in adapters
    ]
    risk_handles = [
        plt.Line2D([0], [0], color="black", lw=2, linestyle="-", label="risky next action"),
        plt.Line2D([0], [0], color="black", lw=2, linestyle="--", label="non-risky next action"),
    ]
    fig.legend(handles=adapter_handles + risk_handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"Aligned pre-action internal dynamics before next-step risk\nTarget: {target_label}", y=1.03, fontsize=12)
    png = outdir / "aligned_temporal_phack_entropy_risk_main.png"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_concat_metric(
    df: pd.DataFrame,
    target_label: str,
    metric_col: str,
    ylabel: str,
    out_png: Path,
    families: list[str],
    regimes: list[str],
    adapters: list[str],
    min_count: int,
) -> None:
    fig, axes = plt.subplots(len(families), len(regimes), figsize=(2.9 * len(regimes), 2.0 * len(families)), sharey=True, constrained_layout=True)
    if len(families) == 1:
        axes = np.array([axes])
    if len(regimes) == 1:
        axes = axes[:, np.newaxis]
    for r, family in enumerate(families):
        for c, regime in enumerate(regimes):
            ax = axes[r, c]
            cell = df[(df["model_family"] == family) & (df["ttc_or_regime"] == regime) & (df["valid_trace"])].copy()
            cursor = 0
            valid = 0
            bad = 0
            nonbad = 0
            for adapter in adapters:
                part = cell[cell["adapter"] == adapter].sort_values(["episode_id", "step_id"])
                for _, row in part.iterrows():
                    arr = np.asarray(row[metric_col], dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if len(arr) < 2:
                        continue
                    xs = np.arange(cursor, cursor + len(arr))
                    ax.plot(xs, arr, color=ADAPTER_COLORS[adapter], linewidth=0.9, alpha=0.9)
                    risky = int(safe_num(row[target_label]) > 0.5)
                    ax.axvline(xs[-1], color="#d62728" if risky else "#999999", linestyle="--", linewidth=0.5, alpha=0.7)
                    cursor = xs[-1] + 2
                    valid += 1
                    bad += risky
                    nonbad += 1 - risky
            if valid < min_count:
                ax.text(0.5, 0.5, f"insufficient data\nvalid={valid}\nbad={bad} nonbad={nonbad}", ha="center", va="center", transform=ax.transAxes)
            title = f"{regime}\nvalid={valid} bad={bad} nonbad={nonbad}"
            ax.set_title(title if r == 0 else f"valid={valid} bad={bad} nonbad={nonbad}")
            if c == 0:
                ax.set_ylabel(f"{family}\n{ylabel}")
            if r == len(families) - 1:
                ax.set_xlabel("Concatenated token index")
            ax.grid(alpha=0.10)
    handles = [plt.Line2D([0], [0], color=ADAPTER_COLORS[a], lw=2, label=ADAPTER_LABELS[a]) for a in adapters]
    handles += [
        plt.Line2D([0], [0], color="#d62728", lw=1.5, linestyle="--", label="boundary before risky action"),
        plt.Line2D([0], [0], color="#999999", lw=1.5, linestyle="--", label="boundary before non-risky action"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Concatenated raw {ylabel.lower()} traces with next-action boundaries\nTarget: {target_label}", y=1.03, fontsize=12)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_summary_md(path: Path, used_files: dict, summary: pd.DataFrame, target_label: str) -> None:
    enough = summary[(summary["n_bad"] > 0) & (summary["n_nonbad"] > 0)].copy()
    top_risk = enough.sort_values("late_delta_risk", ascending=False).head(6)
    top_entropy = enough.reindex(enough["late_delta_entropy"].abs().sort_values(ascending=False).index).head(6)
    missing = summary[(summary["n_valid"] == 0) | (summary["n_bad"] == 0) | (summary["n_nonbad"] == 0)]
    lines = [
        "# Temporal Monitoring Summary",
        "",
        f"Target label: `{target_label}`",
        "",
        "## Files used",
        "",
    ]
    for k, v in used_files.items():
        lines.append(f"- `{k}`: `{v}`")
    lines += [
        "",
        "## Strongest late-window risk separation",
        "",
    ]
    if top_risk.empty:
        lines.append("- No family/regime/adapter cell had enough data for stable risk separation.")
    else:
        for _, row in top_risk.iterrows():
            lines.append(
                f"- `{row['family']}` / `{row['ttc_or_regime']}` / `{row['adapter']}`: "
                f"late Δrisk={row['late_delta_risk']:.3f}, late Δp={row['late_delta_p_hack']:.3f}, late Δent={row['late_delta_entropy']:.3f}"
            )
    lines += [
        "",
        "## Largest entropy separations",
        "",
    ]
    if top_entropy.empty:
        lines.append("- No stable entropy separations available.")
    else:
        for _, row in top_entropy.iterrows():
            lines.append(
                f"- `{row['family']}` / `{row['ttc_or_regime']}` / `{row['adapter']}`: "
                f"late Δentropy={row['late_delta_entropy']:.3f}"
            )
    lines += [
        "",
        "## Missing-data issues",
        "",
    ]
    if missing.empty:
        lines.append("- All family/regime/adapter cells had at least some valid risky and non-risky traces.")
    else:
        for _, row in missing.head(20).iterrows():
            lines.append(
                f"- `{row['family']}` / `{row['ttc_or_regime']}` / `{row['adapter']}`: "
                f"valid={int(row['n_valid'])}, bad={int(row['n_bad'])}, nonbad={int(row['n_nonbad'])}"
            )
    lines += [
        "",
        "## Recommendation",
        "",
        "- Best figure for the paper: `aligned_temporal_phack_entropy_risk_main`",
        "- Appendix-only figures: concatenated raw boundary plots",
        "- If entropy or calibrated risk separates more clearly than `p_hack`, that directly supports the paper thesis.",
        "- If `p_hack` separation is heterogeneous or weak, say so explicitly; that is still consistent with the paper narrative.",
    ]
    path.write_text("\n".join(lines))


def print_terminal_summary(used_files: dict, summary: pd.DataFrame) -> None:
    enough = summary[(summary["n_valid"] > 0) & (summary["n_bad"] > 0) & (summary["n_nonbad"] > 0)].copy()
    top = enough.reindex(enough["late_delta_risk"].abs().sort_values(ascending=False).index).head(8) if not enough.empty else pd.DataFrame()
    print("[INFO] Files used")
    for k, v in used_files.items():
        print(f"  {k}: {v}")
    print("[INFO] Cells with enough data")
    print(f"  {len(enough)} / {len(summary)} family-regime-adapter cells")
    if not top.empty:
        print("[INFO] Largest late-window separations")
        for _, row in top.iterrows():
            print(
                f"  {row['family']} {row['ttc_or_regime']} {row['adapter']}: "
                f"Δrisk={row['late_delta_risk']:.3f}, Δp={row['late_delta_p_hack']:.3f}, Δent={row['late_delta_entropy']:.3f}"
            )
    missing = summary[(summary["n_valid"] == 0) | (summary["n_bad"] == 0) | (summary["n_nonbad"] == 0)]
    if not missing.empty:
        print("[WARN] Missing/sparse cells")
        for _, row in missing.head(15).iterrows():
            print(
                f"  {row['family']} {row['ttc_or_regime']} {row['adapter']}: "
                f"valid={int(row['n_valid'])}, bad={int(row['n_bad'])}, nonbad={int(row['n_nonbad'])}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_label", default="exploit_action_t1")
    ap.add_argument("--environments", nargs="+", default=["alfworld"])
    ap.add_argument("--families", nargs="+", default=FAMILY_ORDER)
    ap.add_argument("--adapters", nargs="+", default=ADAPTER_ORDER)
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--min_count", type=int, default=2)
    ap.add_argument("--n_bins", type=int, default=50)
    ap.add_argument("--late_window_start", type=float, default=0.8)
    ap.add_argument("--include_unconstrained", action="store_true")
    ap.add_argument("--min_tokens", type=int, default=2)
    args = ap.parse_args()

    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, used_files = load_all_steps(ROOT, args.environments)
    if df.empty:
        raise SystemExit("No temporal step data found for the requested environments.")

    traces = prepare_traces(
        df,
        target_label=args.target_label,
        families=args.families,
        adapters=args.adapters,
        n_bins=args.n_bins,
        min_tokens=args.min_tokens,
        include_unconstrained=args.include_unconstrained,
    )
    regimes = sorted(traces["ttc_or_regime"].dropna().unique().tolist(), key=regime_sort_key)
    risk_payload = compute_risk_trajectories(
        traces,
        target_label=args.target_label,
        families=args.families,
        regimes=regimes,
        n_bins=args.n_bins,
        min_count=args.min_count,
        late_window_start=args.late_window_start,
    )
    summary = summarize_cells(
        traces,
        risk_payload,
        target_label=args.target_label,
        families=args.families,
        regimes=regimes,
        adapters=args.adapters,
        late_window_start=args.late_window_start,
    )
    summary_csv = args.output_dir / "temporal_counts_and_deltas.csv"
    summary_md = args.output_dir / "temporal_counts_and_deltas.md"
    summary.to_csv(summary_csv, index=False)
    write_summary_md(summary_md, used_files, summary, args.target_label)

    plot_aligned_main(
        traces,
        risk_payload,
        summary,
        args.output_dir,
        target_label=args.target_label,
        families=args.families,
        regimes=regimes,
        adapters=args.adapters,
        n_bins=args.n_bins,
        min_count=args.min_count,
        late_window_start=args.late_window_start,
    )
    plot_concat_metric(
        traces,
        args.target_label,
        "p_resampled",
        "p_hack",
        args.output_dir / "concat_phack_boundaries_appendix.png",
        args.families,
        regimes,
        args.adapters,
        args.min_count,
    )
    plot_concat_metric(
        traces,
        args.target_label,
        "e_resampled",
        "entropy",
        args.output_dir / "concat_entropy_boundaries_appendix.png",
        args.families,
        regimes,
        args.adapters,
        args.min_count,
    )
    print_terminal_summary(used_files, summary)
    print("[DONE]")
    print(args.output_dir / "aligned_temporal_phack_entropy_risk_main.png")
    print(args.output_dir / "concat_phack_boundaries_appendix.png")
    print(args.output_dir / "concat_entropy_boundaries_appendix.png")
    print(summary_csv)
    print(summary_md)


if __name__ == "__main__":
    main()
