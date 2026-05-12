#!/usr/bin/env python3
from __future__ import annotations

import csv
import argparse
from pathlib import Path


ROOT = Path("/home/patrick/vllmPatrickMonitoring")
BAD_SRC = ROOT / "tables" / "alfworld_prediction_by_ttc_adapter_featuregroup.csv"
EXPLOIT_SRC = ROOT / "tables_rescue" / "qwen_exploit_prediction_by_ttc_adapter_featuregroup.csv"
OUT_DIR = ROOT / "paper_figures"
OUT_CSV = OUT_DIR / "prediction_ablation_qwen_by_adapter_summary.csv"

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
MIN_POSITIVES = 5


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


def select_rows(rows: list[dict], adapter: str, target: str) -> tuple[str | None, list[dict]]:
    stable = []
    for r in rows:
        if r.get("adapter") != adapter:
            continue
        if target == "bad_action_t1":
            if r.get("family") != "Qwen":
                continue
            if r.get("stable") != "True":
                continue
        else:
            if r.get("stable") != "True":
                continue
        pos = fnum(r.get("positives")) or 0.0
        if pos < MIN_POSITIVES:
            continue
        if r.get("feature_group") not in FEATURE_ORDER:
            continue
        stable.append(r)
    if not stable:
        return None, []
    grouped: dict[str, list[dict]] = {}
    for r in stable:
        grouped.setdefault(r["regime"], []).append(r)
    candidates = []
    for regime, sub in grouped.items():
        keyed = {r["feature_group"]: r for r in sub}
        if "combined" in keyed:
            score = fnum(keyed["combined"].get("ap_gain", keyed["combined"].get("auprc_gain"))) or -1e9
            au = fnum(keyed["combined"].get("auroc")) or -1e9
        else:
            score = max((fnum(r.get("ap_gain", r.get("auprc_gain"))) or -1e9) for r in sub)
            au = max((fnum(r.get("auroc")) or -1e9) for r in sub)
        candidates.append((score, au, regime, keyed))
    candidates.sort(reverse=True)
    _, _, regime, keyed = candidates[0]
    chosen = [keyed[fg] for fg in FEATURE_ORDER if fg in keyed]
    return regime, chosen


def write_summary(rows: list[dict]) -> None:
    fieldnames = ["family", "adapter", "target", "feature_group", "regime", "N", "positives", "base_rate", "AUROC", "AUPRC", "AUPRC_gain", "Recall@20"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def render(selected: list[dict], metric_key: str, metric_label: str, out_png: Path, out_pdf: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(ADAPTERS), 2, figsize=(10.5, 11.5), sharey="col", constrained_layout=True)
    target_order = ["bad_action_t1", "exploit_action_t1"]
    target_labels = {"bad_action_t1": "Bad action", "exploit_action_t1": "Exploit action"}

    selected_map = {(r["adapter"], r["target"]): [] for r in selected}
    meta = {}
    for r in selected:
        selected_map.setdefault((r["adapter"], r["target"]), []).append(r)
        meta[(r["adapter"], r["target"])] = (r["regime"], r["positives"], r["N"])

    for ridx, adapter in enumerate(ADAPTERS):
        for cidx, target in enumerate(target_order):
            ax = axes[ridx, cidx]
            sub = selected_map.get((adapter, target), [])
            if not sub:
                ax.axis("off")
                ax.text(0.5, 0.5, "insufficient positives", ha="center", va="center", fontsize=10, transform=ax.transAxes)
                continue
            x = np.arange(len(sub))
            vals = [fnum(r.get(metric_key)) or 0.0 for r in sub]
            bars = ax.bar(x, vals, color=[FEATURE_COLORS[r["feature_group"]] for r in sub], edgecolor="black", linewidth=0.5)
            if metric_key == "AUPRC_gain":
                ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
            elif metric_key == "AUROC":
                ax.axhline(0.5, color="0.35", linestyle="--", linewidth=1.0)
            ax.set_xticks(x)
            ax.set_xticklabels([FEATURE_LABELS[r["feature_group"]] for r in sub], rotation=0)
            if cidx == 0:
                ax.set_ylabel(f"{ADAPTER_LABELS[adapter]}\n{metric_label}")
            regime, pos, n = meta[(adapter, target)]
            ax.set_title(f"{target_labels[target]}\n{regime}  pos={int(float(pos))} / N={int(float(n))}", fontsize=10)
            for bar, val in zip(bars, vals):
                threshold = 0.0 if metric_key != "AUROC" else 0.5
                if val > threshold:
                    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=FEATURE_COLORS[f], label=FEATURE_LABELS[f]) for f in FEATURE_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, title="Feature group")
    fig.suptitle(f"Qwen per-adapter Gameable ALFWorld prediction ablations ({metric_label})", fontsize=14, y=1.01)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["auroc", "auprc_gain", "recall20"], default="auroc")
    return ap.parse_args()


def metric_config(name: str) -> tuple[str, str, Path, Path]:
    if name == "auroc":
        return "AUROC", "AUROC", OUT_DIR / "prediction_ablation_qwen_by_adapter_auroc.png", OUT_DIR / "prediction_ablation_qwen_by_adapter_auroc.pdf"
    if name == "recall20":
        return "Recall@20", "Recall@20", OUT_DIR / "prediction_ablation_qwen_by_adapter_recall20.png", OUT_DIR / "prediction_ablation_qwen_by_adapter_recall20.pdf"
    return "AUPRC_gain", "AUPRC gain", OUT_DIR / "prediction_ablation_qwen_by_adapter.png", OUT_DIR / "prediction_ablation_qwen_by_adapter.pdf"


def main() -> None:
    args = parse_args()
    bad_rows = read_csv(BAD_SRC)
    exploit_rows = read_csv(EXPLOIT_SRC)
    selected = []
    for adapter in ADAPTERS:
        regime, sub = select_rows(bad_rows, adapter, "bad_action_t1")
        for r in sub:
            selected.append({
                "family": "Qwen",
                "adapter": adapter,
                "target": "bad_action_t1",
                "feature_group": r["feature_group"],
                "regime": regime,
                "N": r["n"],
                "positives": r["positives"],
                "base_rate": r["base_rate"],
                "AUROC": r["auroc"],
                "AUPRC": r["ap"],
                "AUPRC_gain": r["ap_gain"],
                "Recall@20": r["recall_at_20pct"],
            })
        regime, sub = select_rows(exploit_rows, adapter, "exploit_action_t1")
        for r in sub:
            selected.append({
                "family": "Qwen",
                "adapter": adapter,
                "target": "exploit_action_t1",
                "feature_group": r["feature_group"],
                "regime": regime,
                "N": r["n"],
                "positives": r["positives"],
                "base_rate": r["base_rate"],
                "AUROC": r["auroc"],
                "AUPRC": r["auprc"],
                "AUPRC_gain": r["auprc_gain"],
                "Recall@20": r["recall_at_20pct"],
            })
    write_summary(selected)
    metric_key, metric_label, out_png, out_pdf = metric_config(args.metric)
    try:
        render(selected, metric_key, metric_label, out_png, out_pdf)
    except ModuleNotFoundError as exc:
        print(f"[WARN] Could not render figure: {exc}")
    print(f"[DONE] Wrote {OUT_CSV}")
    print(f"[DONE] Target figure paths: {out_png}, {out_pdf}")


if __name__ == "__main__":
    main()
