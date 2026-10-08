"""The tables of the Block-8 regularization report, from data/runs/*/score.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parent / "karnataka_inputs" / "shared"))
sys.path.insert(0, str(HERE))
from style import table  # noqa: E402

RUNS = ROOT / "data" / "runs"


def score(name):
    f = RUNS / name / "score.json"
    return json.loads(f.read_text()) if f.exists() else None


def f2(v):
    return "–" if v is None else f"{v:.2f}"


def pct(v):
    return "–" if v is None else f"{100 * v:.0f} %"


def metres(v):
    return "–" if v is None else f"{v:,.0f}"


def gap_cell(s):
    """The held-out χ²/N, marked by how much worse than the fitted nodes it is."""
    g = s["chi2_holdout"] / max(s["chi2_fit_auto"], 1e-9)
    cls = "n warn" if g > 1.5 else ("n good" if g < 1.15 else "n")
    return (f2(s["chi2_holdout"]), cls)


def trial_table(names, label_of=None):
    rows = []
    for n in names:
        s = score(n)
        if s is None:
            continue
        rows.append([
            (label_of(n) if label_of else s["label"], ""),
            f2(s["chi2_fit_auto"]), gap_cell(s),
            f"{s['rms_fit']:,.0f}", f"{s['rms_holdout']:,.0f}", f"{s['rms_holdout_diag']:,.0f}",
            metres(s["depth_p50"]), metres(s["depth_p90"]), pct(s["share_0_200"]),
            pct(s["beside"]), pct(s["below"]), f"{s['chi_max']:.2f}", str(s["n_iterations"]),
            f"{s['seconds']}",
        ])
    head = ["Trial", "χ²/N fitted", "χ²/N left out", "RMS fitted (nT)", "RMS left out", "RMS 53 m off",
            "Depth 50 % (m)", "Depth 90 % (m)", "In top 200 m", "Beside", "Below core", "χ max (SI)",
            "Iterations", "Seconds"]
    return table(head, rows)


def synthetic_table(names, label_of=None):
    rows = []
    for n in names:
        s = score(n)
        if s is None:
            continue
        rows.append([
            (label_of(n) if label_of else s["label"], ""),
            f2(s["chi2_fit_auto"]), f2(s["chi2_holdout"]), f2(s["syn_corr"]), f2(s["syn_rel_l1"]),
            pct(s["syn_moment_in_bodies"]), f2(s["syn_moment_ratio"]),
            f2(s.get("syn_body1_recovered")), f2(s.get("syn_body2_recovered")), f2(s.get("syn_body3_recovered")),
            metres(s["depth_p50"]), pct(s["below"]),
        ])
    head = ["Trial", "χ²/N fitted", "χ²/N left out", "Correlation with the truth", "Relative L1 error",
            "Moment in the true bodies", "Total moment / true", "Band 1 recovered", "Band 2", "Deep block",
            "Depth 50 % (m)", "Below core"]
    return table(head, rows)
