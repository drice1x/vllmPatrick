#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/patrick/vllmPatrickMonitoring")
STEPS_CSV = ROOT / "BALROG" / "analysis_gameable_alfworld_temporal" / "steps_with_entropy_phack_temporal.csv"
OUT_DIR = ROOT / "paper_figures"
OUT_PNG = OUT_DIR / "qwen_bad_action_temporal_grid.png"
OUT_PDF = OUT_DIR / "qwen_bad_action_temporal_grid.pdf"
OUT_MD = OUT_DIR / "qwen_bad_action_temporal_grid_note.md"

ADAPTERS = ["control", "mix05", "mix10", "mix50", "hack"]
ADAPTER_LABELS = {
    "control": "Control",
    "mix05": "Mix05",
    "mix10": "Mix10",
    "mix50": "Mix50",
    "hack": "Hack",
}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def ttc_sort_key(regime: str) -> tuple[int, float]:
    if regime == "unconstrained":
        return (1, 1e9)
    val = regime.replace("ttc_", "")
    try:
        return (0, float(val))
    except Exception:
        return (1, 1e9 - 1)


def collect() -> tuple[list[str], dict]:
    rows = read_rows(STEPS_CSV)
    p_cols = sorted([k for k in rows[0].keys() if k.startswith("reasoning_p_bin")], key=lambda s: int(s.split("bin")[-1]))
    e_cols = sorted([k for k in rows[0].keys() if k.startswith("reasoning_entropy_bin")], key=lambda s: int(s.split("bin")[-1]))

    # sort within episode and create next_bad_action
    rows = [r for r in rows if str(r.get("model", "")).startswith("Qwen")]
    rows.sort(key=lambda r: (r.get("source_file", ""), int(fnum(r.get("step")) or 0)))
    for i, r in enumerate(rows):
        next_bad = None
        if i + 1 < len(rows) and rows[i + 1].get("source_file") == r.get("source_file"):
            next_bad = fnum(rows[i + 1].get("bad_action")) or 0.0
        r["next_bad_action"] = next_bad

    regimes = sorted({r.get("method") or f"ttc_{int(fnum(r.get('ttc')))}" if fnum(r.get("ttc")) is not None else "unconstrained" for r in rows}, key=ttc_sort_key)
    payload = {}
    for adapter in ADAPTERS:
        for regime in regimes:
            sub = [r for r in rows if (r.get("adapter") == adapter) and ((r.get("method") or f"ttc_{int(fnum(r.get('ttc')))}" if fnum(r.get("ttc")) is not None else "unconstrained") == regime)]
            bad_p, good_p, bad_e, good_e = [], [], [], []
            for r in sub:
                label = r.get("next_bad_action")
                if label is None:
                    continue
                p_vals = [fnum(r.get(c)) for c in p_cols]
                e_vals = [fnum(r.get(c)) for c in e_cols]
                if any(v is None for v in p_vals) or any(v is None for v in e_vals):
                    continue
                if float(label) > 0.5:
                    bad_p.append(p_vals)
                    bad_e.append(e_vals)
                else:
                    good_p.append(p_vals)
                    good_e.append(e_vals)
            payload[(adapter, regime)] = {
                "bad_p": bad_p,
                "good_p": good_p,
                "bad_e": bad_e,
                "good_e": good_e,
                "n_valid": len(bad_p) + len(good_p),
                "n_bad": len(bad_p),
                "n_nonbad": len(good_p),
            }
    return regimes, payload


def mean_sem(traces: list[list[float]]) -> tuple[list[float], list[float]]:
    n = len(traces)
    if n == 0:
        return [], []
    cols = len(traces[0])
    means, sems = [], []
    for j in range(cols):
        vals = [t[j] for t in traces]
        m = sum(vals) / n
        means.append(m)
        if n == 1:
            sems.append(0.0)
        else:
            var = sum((v - m) ** 2 for v in vals) / max(1, n - 1)
            sems.append((var ** 0.5) / (n ** 0.5))
    return means, sems


def render(regimes: list[str], payload: dict) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0.0, 1.0, 8)
    fig, axes = plt.subplots(len(ADAPTERS) * 2, len(regimes), figsize=(2.4 * len(regimes), 2.0 * len(ADAPTERS) * 2), sharex=True, constrained_layout=True)
    if len(regimes) == 1:
        axes = np.array(axes).reshape(len(ADAPTERS) * 2, 1)

    for ridx, adapter in enumerate(ADAPTERS):
        for cidx, regime in enumerate(regimes):
            cell = payload[(adapter, regime)]
            for offset, key_bad, key_good, ylabel in [
                (0, "bad_p", "good_p", "p_hack"),
                (1, "bad_e", "good_e", "entropy"),
            ]:
                ax = axes[ridx * 2 + offset, cidx]
                bad = cell[key_bad]
                good = cell[key_good]
                ax.axvspan(0.8, 1.0, color="#dddddd", alpha=0.35)
                if bad and good:
                    m_bad, s_bad = mean_sem(bad)
                    m_good, s_good = mean_sem(good)
                    mb = np.array(m_bad); sb = np.array(s_bad)
                    mg = np.array(m_good); sg = np.array(s_good)
                    ax.plot(x, mb, color="#d62728", linewidth=1.8, label="bad")
                    ax.fill_between(x, mb - 1.96 * sb, mb + 1.96 * sb, color="#d62728", alpha=0.18)
                    ax.plot(x, mg, color="#4C78A8", linewidth=1.8, linestyle="--", label="nonbad")
                    ax.fill_between(x, mg - 1.96 * sg, mg + 1.96 * sg, color="#4C78A8", alpha=0.14)
                else:
                    ax.text(0.5, 0.5, f"insufficient\nvalid={cell['n_valid']}\nbad={cell['n_bad']}\nnon={cell['n_nonbad']}", ha="center", va="center", fontsize=8, transform=ax.transAxes)
                if ridx == 0:
                    ax.set_title(f"{regime}\nvalid={cell['n_valid']} bad={cell['n_bad']} non={cell['n_nonbad']}", fontsize=9)
                if cidx == 0:
                    ax.set_ylabel(f"{ADAPTER_LABELS[adapter]}\n{ylabel}")
                if ridx == len(ADAPTERS) - 1:
                    ax.set_xlabel("Norm. progress")
                ax.grid(alpha=0.15)
    handles = [
        plt.Line2D([0], [0], color="#d62728", lw=2, label="bad_action$_{t+1}=1$"),
        plt.Line2D([0], [0], color="#4C78A8", lw=2, linestyle="--", label="bad_action$_{t+1}=0$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Qwen bad-action temporal structure across TTC budgets", fontsize=14, y=1.01)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def write_note(regimes: list[str], payload: dict) -> None:
    lines = [
        "# Qwen Bad-Action Temporal Grid",
        "",
        "This figure replaces the adapter ablation barplot with direct temporal structure.",
        "- rows: adapters",
        "- columns: TTC budgets",
        "- top sub-row in each adapter: reasoning p_hack",
        "- bottom sub-row in each adapter: reasoning entropy",
        "- red = next step bad_action=1",
        "- blue dashed = next step bad_action=0",
        "",
    ]
    best = []
    for (adapter, regime), cell in payload.items():
        if cell["n_bad"] and cell["n_nonbad"]:
            # simple late-window delta on p_hack
            bp = mean_sem(cell["bad_p"])[0]
            gp = mean_sem(cell["good_p"])[0]
            if bp and gp:
                delta = sum(bp[-2:]) / 2 - sum(gp[-2:]) / 2
                best.append((abs(delta), delta, adapter, regime, cell["n_bad"], cell["n_nonbad"]))
    best.sort(reverse=True)
    lines.append("Largest late-window p_hack separations:")
    for _, delta, adapter, regime, nb, nn in best[:8]:
        lines.append(f"- {ADAPTER_LABELS[adapter]} {regime}: Δlate_p={delta:.3f} (bad={nb}, nonbad={nn})")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    regimes, payload = collect()
    write_note(regimes, payload)
    try:
        render(regimes, payload)
    except ModuleNotFoundError as exc:
        print(f"[WARN] Could not render figure: {exc}")
    print(f"[DONE] Target figure paths: {OUT_PNG}, {OUT_PDF}")
    print(f"[DONE] Note: {OUT_MD}")


if __name__ == "__main__":
    main()
