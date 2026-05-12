#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILY_ORDER = ["Llama", "Qwen", "Falcon"]
ADAPTER_ORDER = ["mix05", "mix10", "mix50"]
ADAPTER_LABELS = {"mix05": "Mix05", "mix10": "Mix10", "mix50": "Mix50", "hack": "Hack"}
ADAPTER_COLORS = {"mix05": "#4c78a8", "mix10": "#f58518", "mix50": "#e45756", "hack": "#8c564b"}


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


def family_of_model(model: str) -> str:
    text = str(model)
    for fam in FAMILY_ORDER:
        if text.startswith(fam):
            return fam
    return "Other"


def regime_value(ttc) -> tuple[str, float | None]:
    val = pd.to_numeric(pd.Series([ttc]), errors="coerce").iloc[0]
    if pd.isna(val):
        return ("unconstrained", None)
    return (str(int(val)), float(val))


def load_steps(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    df["model_family"] = df["model"].map(family_of_model)
    df["step_num"] = pd.to_numeric(df["step"], errors="coerce")
    df["regime"], df["ttc_numeric"] = zip(*df["ttc"].map(regime_value))
    p_cols = sorted([c for c in df.columns if c.startswith("reasoning_p_bin")], key=lambda c: int(c.split("bin")[-1]))
    for c in p_cols + ["bad_action"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["source_file", "step_num"]).reset_index(drop=True)
    df["next_bad_action"] = df.groupby("source_file", sort=False)["bad_action"].shift(-1)
    df["has_next"] = df.groupby("source_file", sort=False)["step_num"].shift(-1).notna()
    df = df[df["has_next"]].copy()
    return df, p_cols


def available_regimes(df: pd.DataFrame) -> list[str]:
    nums = sorted(df["ttc_numeric"].dropna().unique())
    out = [str(int(x)) for x in nums]
    if (df["regime"] == "unconstrained").any():
        out.append("unconstrained")
    return out


def filter_cell(df: pd.DataFrame, family: str, regime: str, include_hack: bool = False) -> pd.DataFrame:
    adapters = ADAPTER_ORDER + (["hack"] if include_hack else [])
    sub = df[(df["model_family"] == family) & (df["adapter"].isin(adapters))].copy()
    if regime == "unconstrained":
        sub = sub[sub["regime"] == "unconstrained"]
    else:
        sub = sub[sub["ttc_numeric"] == float(regime)]
    return sub


def resample_traj(vals: np.ndarray, n_points: int) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    mask = np.isfinite(vals)
    vals = vals[mask]
    if len(vals) == 0:
        return np.full(n_points, np.nan)
    if len(vals) == 1:
        return np.full(n_points, vals[0])
    x_old = np.linspace(0.0, 1.0, len(vals))
    x_new = np.linspace(0.0, 1.0, n_points)
    return np.interp(x_new, x_old, vals)


def count_valid_traces(sub: pd.DataFrame, p_cols: list[str], min_points: int = 2) -> tuple[int, int, int]:
    valid = 0
    bad = 0
    nonbad = 0
    for _, row in sub.iterrows():
        vals = pd.to_numeric(row[p_cols], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < min_points:
            continue
        valid += 1
        if float(row.get("next_bad_action") or 0.0) > 0.5:
            bad += 1
        else:
            nonbad += 1
    return valid, bad, nonbad


def draw_raw_grid(df: pd.DataFrame, p_cols: list[str], out_png: Path, include_hack: bool = False) -> dict:
    regimes = available_regimes(df)
    fig, axes = plt.subplots(len(FAMILY_ORDER), len(regimes), figsize=(3.2 * len(regimes), 2.6 * len(FAMILY_ORDER)), sharex=False, sharey=True, constrained_layout=True)
    if len(FAMILY_ORDER) == 1:
        axes = np.array([axes])
    if len(regimes) == 1:
        axes = axes[:, np.newaxis]
    stats = {}
    for r, family in enumerate(FAMILY_ORDER):
        for c, regime in enumerate(regimes):
            ax = axes[r, c]
            sub = filter_cell(df, family, regime, include_hack=include_hack)
            key = (family, regime)
            valid_n, valid_bad, valid_nonbad = count_valid_traces(sub, p_cols, min_points=2)
            stats[key] = {
                "rows": len(sub),
                "valid_traces": valid_n,
                "bad": valid_bad,
                "nonbad": valid_nonbad,
            }
            title = f"{regime}\nvalid={valid_n} bad={valid_bad} nonbad={valid_nonbad}"
            if valid_n < 3:
                ax.text(
                    0.5,
                    0.5,
                    f"insufficient data\nvalid={valid_n}\nbad={valid_bad} nonbad={valid_nonbad}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(title if r == 0 else f"valid={valid_n} bad={valid_bad} nonbad={valid_nonbad}")
                if c == 0:
                    ax.set_ylabel(family)
                ax.set_ylim(0.0, 1.0)
                continue
            cursor = 0
            for adapter in ADAPTER_ORDER + (["hack"] if include_hack else []):
                part = sub[sub["adapter"] == adapter].sort_values(["source_file", "step_num"])
                for _, row in part.iterrows():
                    vals = pd.to_numeric(row[p_cols], errors="coerce").to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if len(vals) < 2:
                        continue
                    xs = np.arange(cursor, cursor + len(vals))
                    ax.plot(xs, vals, color=ADAPTER_COLORS[adapter], linewidth=1.1, alpha=0.95)
                    line_color = "#d62728" if float(row.get("next_bad_action") or 0.0) > 0.5 else "#999999"
                    ax.axvline(xs[-1], color=line_color, linestyle="--", linewidth=0.6, alpha=0.7)
                    cursor = xs[-1] + 2
            ax.set_ylim(0.0, 1.0)
            ax.set_title(title if r == 0 else f"valid={valid_n} bad={valid_bad} nonbad={valid_nonbad}")
            if c == 0:
                ax.set_ylabel(family + "\n$p_{hack}$")
            if r == len(FAMILY_ORDER) - 1:
                ax.set_xlabel("Concatenated token index")
            ax.grid(alpha=0.12)
    handles = [plt.Line2D([0], [0], color=ADAPTER_COLORS[a], lw=2, label=ADAPTER_LABELS[a]) for a in ADAPTER_ORDER + (["hack"] if include_hack else [])]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Concatenated reward-hack activation trajectories before next action", y=1.04, fontsize=12)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return stats


def draw_aligned_grid(df: pd.DataFrame, p_cols: list[str], out_png: Path, include_hack: bool = False, n_points: int = 32) -> dict:
    regimes = available_regimes(df)
    fig, axes = plt.subplots(len(FAMILY_ORDER), len(regimes), figsize=(3.2 * len(regimes), 2.8 * len(FAMILY_ORDER)), sharex=True, sharey=True, constrained_layout=True)
    if len(FAMILY_ORDER) == 1:
        axes = np.array([axes])
    if len(regimes) == 1:
        axes = axes[:, np.newaxis]
    stats = {}
    x = np.linspace(-1.0, 0.0, n_points)
    for r, family in enumerate(FAMILY_ORDER):
        for c, regime in enumerate(regimes):
            ax = axes[r, c]
            sub = filter_cell(df, family, regime, include_hack=include_hack)
            valid_n, valid_bad, valid_nonbad = count_valid_traces(sub, p_cols, min_points=2)
            cell_stats = {}
            enough = False
            for adapter in ADAPTER_ORDER + (["hack"] if include_hack else []):
                part = sub[sub["adapter"] == adapter]
                cell_stats[adapter] = {}
                for label, style, alpha in [(1, "-", 0.18), (0, "--", 0.10)]:
                    traces = []
                    for _, row in part[part["next_bad_action"].fillna(0).round().astype(int) == label].iterrows():
                        vals = pd.to_numeric(row[p_cols], errors="coerce").to_numpy(dtype=float)
                        res = resample_traj(vals, n_points)
                        if np.isfinite(res).sum() >= max(4, n_points // 4):
                            traces.append(res)
                    if len(traces) >= 2:
                        enough = True
                        arr = np.vstack(traces)
                        mean = np.nanmean(arr, axis=0)
                        sem = np.nanstd(arr, axis=0) / np.sqrt(len(arr))
                        ax.plot(x, mean, linestyle=style, color=ADAPTER_COLORS[adapter], linewidth=1.8)
                        ax.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem, color=ADAPTER_COLORS[adapter], alpha=alpha)
                    cell_stats[adapter][label] = len(traces)
            stats[(family, regime)] = {
                "cell": {
                    "rows": len(sub),
                    "valid_traces": valid_n,
                    "bad": valid_bad,
                    "nonbad": valid_nonbad,
                },
                "adapters": cell_stats,
            }
            if not enough:
                ax.text(
                    0.5,
                    0.5,
                    f"insufficient data\nvalid={valid_n}\nbad={valid_bad} nonbad={valid_nonbad}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.axvspan(-0.2, 0.0, color="#ececec", zorder=0)
            ax.set_ylim(0.0, 1.0)
            title = f"{regime}\nvalid={valid_n} bad={valid_bad} nonbad={valid_nonbad}"
            ax.set_title(title if r == 0 else f"valid={valid_n} bad={valid_bad} nonbad={valid_nonbad}")
            if c == 0:
                ax.set_ylabel(family + "\n$p_{hack}$")
            if r == len(FAMILY_ORDER) - 1:
                ax.set_xlabel("Normalized position before action")
            ax.grid(alpha=0.12)
    handles = []
    for adapter in ADAPTER_ORDER + (["hack"] if include_hack else []):
        handles.append(plt.Line2D([0], [0], color=ADAPTER_COLORS[adapter], lw=2, label=ADAPTER_LABELS[adapter]))
    handles.append(plt.Line2D([0], [0], color="black", lw=2, linestyle="-", label="bad next action"))
    handles.append(plt.Line2D([0], [0], color="black", lw=2, linestyle="--", label="non-bad next action"))
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 5), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Aligned pre-action reward-hack activation trajectories", y=1.04, fontsize=12)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return stats


def write_notes(path: Path, raw_stats: dict, aligned_stats: dict) -> None:
    lines = [
        "# p_hack Concatenated Grid Notes",
        "",
        "## Summary",
        "",
        "This figure package is exploratory. It is designed to visually inspect whether reward-hack activation tends to increase before bad actions, without forcing a universal rise story.",
        "",
        "## Raw concatenated plot",
        "",
        "- The raw concatenated grid is useful for qualitative inspection of local fluctuations and trajectory density.",
        "- It is visually rich, but can become cluttered when many short reasoning traces are concatenated.",
        "- It is best treated as appendix material unless a very clear pattern is visible in a small number of cells.",
        "",
        "## Endpoint-aligned plot",
        "",
        "- The aligned-end grid is more scientifically interpretable for the paper.",
        "- It directly compares bad vs non-bad next-action trajectories near the action boundary.",
        "- If any pre-bad-action increase exists, it should appear more clearly here than in the raw concatenated view.",
        "",
        "## Cell-level observations",
        "",
    ]
    apparent = []
    weak = []
    for (family, regime), cell_payload in aligned_stats.items():
        adapter_stats = cell_payload["adapters"]
        cell = cell_payload["cell"]
        deltas = []
        notes = []
        for adapter, counts in adapter_stats.items():
            n_bad = counts.get(1, 0)
            n_good = counts.get(0, 0)
            if n_bad >= 2 and n_good >= 2:
                notes.append(f"{ADAPTER_LABELS.get(adapter, adapter)}: bad={n_bad}, nonbad={n_good}")
        if notes:
            lines.append(
                f"- `{family}` / `{regime}`: valid={cell['valid_traces']}, bad={cell['bad']}, nonbad={cell['nonbad']}; "
                + "; ".join(notes)
            )
            apparent.append((family, regime))
        else:
            weak.append((family, regime))
    lines += [
        "",
        "## Interpretation",
        "",
        "If visible pre-bad-action increases are present, they are likely to be heterogeneous across families and TTC regimes rather than universal.",
        "This is consistent with the broader paper thesis: reward-hack activation remains structured, but immediate action risk is context-dependent and often requires entropy and context for prediction.",
        "",
        "## Recommendation",
        "",
        "- Better figure for the ICML paper: `phack_preaction_aligned_grid_all_models`",
        "- Better figure for appendix/qualitative reference: `phack_concatenated_grid_all_models`",
        "- If the aligned figure does not show a clean universal pre-bad-action rise, say so explicitly. That outcome is still consistent with the paper’s main message.",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-csv", type=Path, default=Path("/home/patrick/vllmPatrickMonitoring/BALROG/analysis_gameable_alfworld_temporal/steps_with_entropy_phack_temporal.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("/home/patrick/vllmPatrickMonitoring/figures/main"))
    ap.add_argument("--include-hack", action="store_true")
    args = ap.parse_args()

    configure_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df, p_cols = load_steps(args.steps_csv)
    raw_path = args.outdir / "phack_concatenated_grid_all_models.png"
    aligned_path = args.outdir / "phack_preaction_aligned_grid_all_models.png"
    raw_stats = draw_raw_grid(df, p_cols, raw_path, include_hack=args.include_hack)
    aligned_stats = draw_aligned_grid(df, p_cols, aligned_path, include_hack=args.include_hack)
    notes_path = args.outdir / "phack_concatenated_grid_all_models_notes.md"
    write_notes(notes_path, raw_stats, aligned_stats)
    print("[DONE]")
    print(raw_path)
    print(raw_path.with_suffix(".pdf"))
    print(aligned_path)
    print(aligned_path.with_suffix(".pdf"))
    print(notes_path)


if __name__ == "__main__":
    main()
