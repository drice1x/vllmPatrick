#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/home/patrick/vllmPatrickMonitoring")
OUTDIR = ROOT / "paper_analysis_outputs"
INVENTORY = OUTDIR / "webshop_prediction_file_inventory.md"
NORMALIZED = OUTDIR / "context_calibrated_prediction_ablation_webshop.csv"


def maybe_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def family_from_name(name: str) -> str:
    text = str(name)
    for fam in ["Qwen", "Llama", "Falcon"]:
        if text.startswith(fam):
            return fam
    return "pooled"


def family_from_path(path: Path) -> str:
    text = str(path).lower()
    if "llama" in text:
        return "Llama"
    if "falcon" in text:
        return "Falcon"
    return "Qwen"


def normalize_target(name: str) -> str:
    if name in {"bad_buy_action_t1", "bad_buy_action", "bad_buy_t1"}:
        return "bad_buy_t1"
    if name in {"low_reward_buy_action_t1", "low_reward_buy_action", "low_reward_buy_t1", "low_reward_buy"}:
        return "low_reward_buy_t1"
    if name in {"buy_action_t1", "buy_action", "buy_t1"}:
        return "buy_t1"
    return name


def add_row(rows: list[dict], family: str, target: str, block: dict, feature_group: str) -> None:
    n = block.get("n")
    positives = block.get("positives")
    base = block.get("base_rate")
    auroc = block.get("auroc")
    auprc = block.get("average_precision", block.get("AUPRC", block.get("AP")))
    gain = None
    if auprc is not None and base is not None:
        gain = float(auprc) - float(base)
    r20 = None
    if isinstance(block.get("recall_at_20pct_flag"), dict):
        r20 = block["recall_at_20pct_flag"].get("recall")
    elif block.get("recall_at_20") is not None:
        r20 = block.get("recall_at_20")
    elif block.get("recall_at_20pct") is not None:
        r20 = block.get("recall_at_20pct")
    rows.append(
        {
            "family": family,
            "target": normalize_target(target),
            "feature_group": feature_group,
            "N": n,
            "positives": positives,
            "base_rate": base,
            "AUROC": auroc,
            "AUPRC": auprc,
            "AUPRC_gain": gain,
            "Recall@20": r20,
        }
    )


def inspect_json_summary(path: Path) -> tuple[dict, list[dict]]:
    obj = maybe_json(path)
    report = {
        "path": str(path),
        "format": "json",
        "contains_prediction_metrics": False,
        "contains_step_pairs_only": False,
        "targets_found": [],
        "metric_names_found": [],
        "family_model_names_found": [],
        "usable_directly": False,
        "needs_conversion": False,
    }
    normalized_rows: list[dict] = []
    if not isinstance(obj, dict):
        return report, normalized_rows
    targets = list(obj.keys())
    report["targets_found"] = targets
    report["contains_prediction_metrics"] = any(isinstance(v, dict) and "pooled" in v for v in obj.values())
    report["usable_directly"] = report["contains_prediction_metrics"]
    report["needs_conversion"] = report["contains_prediction_metrics"]
    feature_group = "p_hack_plus_entropy"
    metric_names = set()
    family_names = set()
    file_family = family_from_path(path)
    for target, target_obj in obj.items():
        if not isinstance(target_obj, dict):
            continue
        pooled = target_obj.get("pooled")
        if isinstance(pooled, dict):
            metric_names.update(pooled.keys())
            # These files are already family-specific sweeps, so the pooled row is
            # the clean family-level result we want for paper tables.
            add_row(normalized_rows, file_family, target, pooled, feature_group)
        by_model = target_obj.get("by_model", {})
        if isinstance(by_model, dict):
            for model_name, block in by_model.items():
                family = family_from_name(model_name)
                family_names.add(family)
                if isinstance(block, dict):
                    metric_names.update(block.keys())
    report["metric_names_found"] = sorted(metric_names)
    report["family_model_names_found"] = sorted(family_names)
    return report, normalized_rows


def inspect_json_pairs(path: Path) -> dict:
    obj = maybe_json(path)
    report = {
        "path": str(path),
        "format": "json",
        "contains_prediction_metrics": False,
        "contains_step_pairs_only": False,
        "targets_found": [],
        "metric_names_found": [],
        "family_model_names_found": [],
        "usable_directly": False,
        "needs_conversion": False,
    }
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        keys = set(obj[0].keys())
        report["contains_step_pairs_only"] = True
        report["targets_found"] = sorted([k for k in keys if "buy_t1" in k or "bad_buy" in k or "low_reward_buy" in k])
        report["metric_names_found"] = sorted([k for k in keys if "p_hack" in k or "entropy" in k])
        report["family_model_names_found"] = sorted({family_from_name(x.get("model", "")) for x in obj if isinstance(x, dict)})
        report["needs_conversion"] = True
    return report


def inspect_md(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    lines = text.splitlines()[:50]
    lowered = text.lower()
    return {
        "path": str(path),
        "format": "markdown",
        "contains_prediction_metrics": any(k in lowered for k in ["auroc", "average_precision", "auprc", "base_rate", "positives"]),
        "contains_step_pairs_only": False,
        "targets_found": [t for t in ["buy_t1", "bad_buy_t1", "low_reward_buy_t1"] if t in text],
        "metric_names_found": [m for m in ["AUROC", "AUPRC", "AP", "base_rate", "positives", "Recall@20", "recall_at_20"] if m.lower() in lowered],
        "family_model_names_found": [f for f in ["Qwen", "Llama", "Falcon"] if f.lower() in lowered],
        "usable_directly": False,
        "needs_conversion": False,
        "preview": lines,
    }


def write_inventory(reports: list[dict]) -> None:
    lines = ["# WebShop Prediction File Inventory", ""]
    for rep in reports:
        lines += [
            f"## `{rep['path']}`",
            "",
            f"- format: `{rep['format']}`",
            f"- contains prediction metrics: `{rep['contains_prediction_metrics']}`",
            f"- contains step pairs only: `{rep['contains_step_pairs_only']}`",
            f"- targets found: `{', '.join(rep.get('targets_found', [])) or 'none'}`",
            f"- metric names found: `{', '.join(rep.get('metric_names_found', [])) or 'none'}`",
            f"- family/model names found: `{', '.join(rep.get('family_model_names_found', [])) or 'none'}`",
            f"- usable directly: `{rep['usable_directly']}`",
            f"- needs conversion: `{rep['needs_conversion']}`",
            "",
        ]
        if rep.get("preview"):
            lines.append("First 50 lines preview:")
            lines.append("```md")
            lines.extend(rep["preview"])
            lines.append("```")
            lines.append("")
    INVENTORY.write_text("\n".join(lines))


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    candidates = [
        ROOT / "BALROG/webshop_react_eval/runs/predictive_next_step_buycurated/predictive_summary.json",
        ROOT / "BALROG/webshop_react_eval/runs/predictive_next_step_buycurated/predictive_summary.md",
        ROOT / "BALROG/webshop_react_eval/runs/predictive_next_step_buycurated_llama/predictive_summary.json",
        ROOT / "BALROG/webshop_react_eval/runs/predictive_next_step_buycurated_llama/predictive_summary.md",
        ROOT / "BALROG/webshop_react_eval/runs/predictive_next_step_buycurated_falcon/predictive_summary.json",
        ROOT / "BALROG/webshop_react_eval/runs/predictive_next_step_buycurated_falcon/predictive_summary.md",
        ROOT / "BALROG/webshop_react_eval/runs/next_step_analysis_buycurated/next_step_pairs.json",
        ROOT / "BALROG/webshop_react_eval/runs/next_step_analysis_buycurated_llama/next_step_pairs.json",
        ROOT / "BALROG/webshop_react_eval/runs/next_step_analysis_buycurated_falcon/next_step_pairs.json",
    ]
    reports = []
    normalized_rows = []
    used_files = []
    for path in candidates:
        if not path.exists():
            continue
        if path.name == "predictive_summary.json":
            rep, rows = inspect_json_summary(path)
            reports.append(rep)
            if rows:
                normalized_rows.extend(rows)
                used_files.append(str(path))
        elif path.suffix == ".json":
            reports.append(inspect_json_pairs(path))
        elif path.suffix == ".md":
            reports.append(inspect_md(path))
    write_inventory(reports)

    wrote_csv = False
    if normalized_rows:
        fieldnames = ["family", "target", "feature_group", "N", "positives", "base_rate", "AUROC", "AUPRC", "AUPRC_gain", "Recall@20"]
        with NORMALIZED.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in normalized_rows:
                w.writerow(row)
        wrote_csv = True

    print("[INFO] Candidate files inspected:")
    for rep in reports:
        print(f"  {rep['path']}")
    print(f"[INFO] Inventory written: {INVENTORY}")
    if wrote_csv:
        targets = sorted(set(r["target"] for r in normalized_rows))
        families = sorted(set(r["family"] for r in normalized_rows))
        print("[INFO] WebShop prediction results existed: True")
        print(f"[INFO] Files used: {', '.join(used_files)}")
        print(f"[INFO] Targets found: {targets}")
        print(f"[INFO] Families found: {families}")
        print(f"[INFO] Normalized CSV written: {NORMALIZED}")
    else:
        print("[INFO] WebShop prediction results existed: False")
        print("[INFO] Normalized CSV written: False")
        print("[INFO] Regeneration command:")
        print("cd ~/vllmPatrickMonitoring/BALROG/webshop_react_eval")
        print("python predict_next_step_actions.py --pairs-json runs/next_step_analysis_buycurated/next_step_pairs.json --outdir runs/predictive_next_step_buycurated")


if __name__ == "__main__":
    main()
