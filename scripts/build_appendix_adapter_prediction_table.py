#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path("/home/patrick/vllmPatrickMonitoring")
BAD_SRC = ROOT / "tables" / "alfworld_prediction_by_ttc_adapter_featuregroup.csv"
EXPLOIT_SRC = ROOT / "tables_rescue" / "qwen_exploit_prediction_by_ttc_adapter_featuregroup.csv"
OUT_TEX = ROOT / "paper_appendix_outputs" / "appendix_adapter_prediction_breakdown.tex"

ADAPTER_ORDER = {"control": 0, "mix05": 1, "mix10": 2, "mix50": 3, "hack": 4}
FEATURE_ORDER = {"combined": 0, "entropy-only": 1, "p_hack-only": 2, "temporal-p_hack-only": 3}


def read_csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(x):
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def best_rows_bad(rows):
    out = []
    grouped = {}
    for r in rows:
        if r.get("stable") != "True":
            continue
        fam = r.get("family")
        adapter = r.get("adapter")
        if fam is None or adapter is None:
            continue
        grouped.setdefault((fam, adapter), []).append(r)
    for key, sub in grouped.items():
        best = max(
            sub,
            key=lambda r: (
                f(r.get("ap_gain")) if f(r.get("ap_gain")) is not None else -999,
                f(r.get("auroc")) if f(r.get("auroc")) is not None else -999,
                -FEATURE_ORDER.get(r.get("feature_group", ""), 99),
            ),
        )
        out.append(("bad_action_t1", best))
    return out


def best_rows_exploit_qwen(rows):
    out = []
    grouped = {}
    for r in rows:
        if r.get("stable") != "True":
            continue
        if (f(r.get("positives")) or 0) < 5:
            continue
        adapter = r.get("adapter")
        grouped.setdefault(adapter, []).append(r)
    for adapter, sub in grouped.items():
        best = max(
            sub,
            key=lambda r: (
                f(r.get("auprc_gain")) if f(r.get("auprc_gain")) is not None else -999,
                f(r.get("auroc")) if f(r.get("auroc")) is not None else -999,
                -FEATURE_ORDER.get(r.get("feature_group", ""), 99),
            ),
        )
        out.append(("exploit_action_t1", best))
    return out


def fmt(x):
    if x is None:
        return "n/a"
    return f"{float(x):.3f}"


def tex_escape(text: str) -> str:
    return str(text).replace("_", r"\_")


def main():
    bad = read_csv(BAD_SRC)
    exploit = read_csv(EXPLOIT_SRC)

    rows = []
    for target, r in best_rows_bad(bad):
        rows.append(
            {
                "family": r["family"],
                "adapter": r["adapter"],
                "target": target,
                "best_group": r["feature_group"],
                "regime": r["regime"],
                "auroc": f(r.get("auroc")),
                "auprc": f(r.get("ap")),
                "gain": f(r.get("ap_gain")),
                "recall20": f(r.get("recall_at_20pct")),
                "positives": int(f(r.get("positives")) or 0),
            }
        )
    for target, r in best_rows_exploit_qwen(exploit):
        rows.append(
            {
                "family": "Qwen",
                "adapter": r["adapter"],
                "target": target,
                "best_group": r["feature_group"],
                "regime": r["regime"],
                "auroc": f(r.get("auroc")),
                "auprc": f(r.get("auprc")),
                "gain": f(r.get("auprc_gain")),
                "recall20": f(r.get("recall_at_20pct")),
                "positives": int(f(r.get("positives")) or 0),
            }
        )

    rows.sort(key=lambda r: ({"Qwen": 0, "Llama": 1, "Falcon": 2}.get(r["family"], 9), ADAPTER_ORDER.get(r["adapter"], 9), r["target"]))

    lines = [
        r"\begin{tabular}{lllllrrrrr}",
        r"\toprule",
        r"Family & Adapter & Target & Best group & Regime & AUROC & AUPRC & AUPRC gain & Recall@20 & Positives \\",
        r"\midrule",
    ]
    last_family = None
    for r in rows:
        if last_family is not None and r["family"] != last_family:
            lines.append(r"\midrule")
        last_family = r["family"]
        lines.append(
            f"{tex_escape(r['family'])} & {tex_escape(r['adapter'])} & {tex_escape(r['target'])} & {tex_escape(r['best_group'])} & {tex_escape(r['regime'])} & "
            f"{fmt(r['auroc'])} & {fmt(r['auprc'])} & {fmt(r['gain'])} & {fmt(r['recall20'])} & {r['positives']} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    OUT_TEX.write_text("\n".join(lines) + "\n")
    print(OUT_TEX)


if __name__ == "__main__":
    main()
