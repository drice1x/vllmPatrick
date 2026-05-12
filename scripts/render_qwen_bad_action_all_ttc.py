#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path("/home/patrick/vllmPatrickMonitoring")
SRC = ROOT / "tables" / "alfworld_prediction_by_ttc_adapter_featuregroup.csv"
OUT_DIR = ROOT / "paper_figures"

ADAPTERS = ["control", "mix05", "mix10", "mix50", "hack"]
ADAPTER_LABELS = {
    "control": "Control",
    "mix05": "Mix05",
    "mix10": "Mix10",
    "mix50": "Mix50",
    "hack": "Hack",
}
FEATURE_ORDER = ["p_hack-only", "entropy-only", "temporal-p_hack-only", "combined"]
FEATURE_LABELS = {
    "p_hack-only": "act",
    "entropy-only": "ent",
    "temporal-p_hack-only": "temp",
    "combined": "all",
}
FEATURE_COLORS = {
    "p_hack-only": "#4C78A8",
    "entropy-only": "#F58518",
    "temporal-p_hack-only": "#54A24B",
    "combined": "#E45756",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def regime_key(name: str) -> tuple[int, float | str]:
    s = str(name).strip().lower()
    if s in {"", "nan"}:
        return (99, s)
    if "unconstrained" in s or "direct" in s or "react" in s:
        return (98, s)
    try:
        val = float(s.replace("ttc_", ""))
        return (0, val)
    except Exception:
        return (50, s)


def regime_label(name: str) -> str:
    s = str(name)
    if s.startswith("ttc_"):
        return s.replace("ttc_", "")
    return s


def metric_config(name: str) -> tuple[str, str, float, Path, Path]:
    if name == "auprc_gain":
        return (
            "ap_gain",
            "AUPRC gain over base rate",
            0.0,
            OUT_DIR / "prediction_ablation_qwen_bad_action_all_ttc_gain.png",
            OUT_DIR / "prediction_ablation_qwen_bad_action_all_ttc_gain.pdf",
        )
    if name == "recall20":
        return (
            "recall_at_20pct",
            "Recall@20%",
            0.0,
            OUT_DIR / "prediction_ablation_qwen_bad_action_all_ttc_recall20.png",
            OUT_DIR / "prediction_ablation_qwen_bad_action_all_ttc_recall20.pdf",
        )
    return (
        "auroc",
        "AUROC",
        0.5,
        OUT_DIR / "prediction_ablation_qwen_bad_action_all_ttc_auroc.png",
        OUT_DIR / "prediction_ablation_qwen_bad_action_all_ttc_auroc.pdf",
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["auroc", "auprc_gain", "recall20"], default="auroc")
    ap.add_argument("--min-positives", type=int, default=5)
    return ap.parse_args()


def render(rows: list[dict], metric_name: str, ylabel: str, baseline: float, out_png: Path, out_pdf: Path, min_positives: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qwen = [
        r for r in rows
        if r.get("family") == "Qwen"
        and r.get("stable") == "True"
        and (fnum(r.get("positives")) or 0) >= min_positives
        and r.get("feature_group") in FEATURE_ORDER
    ]
    regimes = sorted({r["regime"] for r in qwen}, key=regime_key)
    x = np.arange(len(regimes))

    fig, axes = plt.subplots(len(ADAPTERS), 1, figsize=(11.0, 10.5), sharex=True, sharey=True, constrained_layout=True)
    for idx, adapter in enumerate(ADAPTERS):
        ax = axes[idx]
        sub = [r for r in qwen if r.get("adapter") == adapter]
        if not sub:
            ax.axis("off")
            ax.text(0.5, 0.5, "no stable rows", ha="center", va="center", transform=ax.transAxes)
            continue
        positives = sorted({int(fnum(r.get("positives")) or 0) for r in sub})
        ns = sorted({int(fnum(r.get("n")) or 0) for r in sub})
        ax.axhline(baseline, color="0.55", linestyle="--", linewidth=1.0)
        for fg in FEATURE_ORDER:
            pts = []
            for regime in regimes:
                row = next((r for r in sub if r.get("regime") == regime and r.get("feature_group") == fg), None)
                pts.append(fnum(row.get(metric_name)) if row else np.nan)
            ax.plot(
                x,
                pts,
                marker="o",
                linewidth=2.0,
                markersize=5,
                color=FEATURE_COLORS[fg],
                label=FEATURE_LABELS[fg],
            )
        ax.set_ylabel(f"{ADAPTER_LABELS[adapter]}\n{ylabel}")
        pos_txt = positives[0] if len(positives) == 1 else f"{min(positives)}-{max(positives)}"
        n_txt = ns[0] if len(ns) == 1 else f"{min(ns)}-{max(ns)}"
        ax.set_title(f"{ADAPTER_LABELS[adapter]}   pos={pos_txt}   N={n_txt}", fontsize=10)
        ax.grid(axis="y", alpha=0.2)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([regime_label(r) for r in regimes])
    axes[-1].set_xlabel("TTC / reasoning budget")
    handles = [plt.Line2D([0], [0], color=FEATURE_COLORS[f], marker="o", linewidth=2.0, label=FEATURE_LABELS[f]) for f in FEATURE_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, title="Feature group")
    fig.suptitle("Qwen bad_action prediction by adapter across TTC", fontsize=14, y=1.01)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_csv(SRC)
    metric_name, ylabel, baseline, out_png, out_pdf = metric_config(args.metric)
    try:
        render(rows, metric_name, ylabel, baseline, out_png, out_pdf, args.min_positives)
    except ModuleNotFoundError as exc:
        print(f"[WARN] Could not render figure: {exc}")
    print(f"[DONE] Wrote target figure paths: {out_png}, {out_pdf}")


if __name__ == "__main__":
    main()
