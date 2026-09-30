"""Before and after the fixes of 30 September (evening): the open problems of Section 5.

    py examples/output/karnataka_joint/scripts/fixes_summary.py

Compares data/ec2_runs (the runs of the report) with data/ec2_runs_fixed (the same couplings
after the fixes; the group lasso as group_lasso_depth), data/lowres_runs with
data/lowres_runs_fixed (2 km), and the single magnetic inversion beta1 of karnataka_magnetic
with its magnetization-vector version beta1_mvi.  Writes figures/fixes_numbers.json; the
measures are those of make_figures.py (kmodel.Grid.shares, data_fit, coupling_measures).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from make_figures import BOX, FIGS, coupling_measures, load_joint  # noqa: E402
from kmodel import Grid, box_stats, data_fit, load  # noqa: E402

DATA = ROOT / "data"
MAGNETIC = ROOT.parent / "karnataka_magnetic" / "data" / "ec2_runs"
# (the report's run, the fixed run)
PAIRS = [("none", "none"), ("cross_gradient", "cross_gradient"),
         ("joint_total_variation", "joint_total_variation"),
         ("linear_correspondence", "linear_correspondence"), ("pgi", "pgi"),
         ("group_lasso", "group_lasso_depth"),
         ("group_lasso_uncoupled", "group_lasso_depth_uncoupled")]
UNDERFIT_NT = 150.0     # the stations an induced model underfits by more than this


def measures(path: Path) -> dict | None:
    if not (path / "result.zip").exists():
        return None
    r, gg, gm = load_joint(path)
    out = {"gravity": {**gg.shares(), **data_fit(r["_datas"]["gravity"]),
                       "min": float(gg.m.min()), "max": float(gg.m.max())},
           "magnetics": {**gm.shares(), **data_fit(r["_datas"]["magnetics"]),
                         "min": float(gm.m.min()), "max": float(gm.m.max())},
           "coupling": coupling_measures(gg, gm),
           "n_iterations": r.get("n_iterations"), "stopped": r.get("stopped_early")}
    gl = r.get("group_lasso")
    if gl:
        out["group_lasso"] = {k: gl.get(k) for k in ("criterion", "lambda1", "lambda1_max",
                                                     "data_weights", "weighting", "relaxation",
                                                     "admm_iterations")}
        out["group_lasso"]["balance"] = [
            {k: b.get(k) for k in ("round", "chi2_per_datum", "data_weights")}
            for b in gl.get("balance") or []]
        out["group_lasso"]["sweep_points"] = len(gl.get("sweep") or [])
    for name in ("run.json",):
        if (path / name).exists():
            out["run"] = json.loads((path / name).read_text())
    return out


def magnetic_pair() -> dict:
    """The induced single inversion (beta1) against the MVI one (beta1_mvi)."""
    if not (MAGNETIC / "beta1_mvi" / "result.zip").exists():
        return {}
    a, b = load(MAGNETIC / "beta1"), load(MAGNETIC / "beta1_mvi")
    da, db = a["_data"], b["_data"]
    ra, rb = da["observed"] - da["predicted"], db["observed"] - db["predicted"]
    bad = ra > UNDERFIT_NT
    locs = da["locations"]
    out = {"induced": {**data_fit(da), **Grid(a).shares(), "box_sandur": box_stats(Grid(a), BOX)},
           "mvi": {**data_fit(db), **Grid(b).shares(), "box_sandur": box_stats(Grid(b), BOX),
                   "magnetization": b.get("magnetization")},
           "underfit": {"threshold_nT": UNDERFIT_NT, "n": int(bad.sum()),
                        "induced_mean_nT": float(ra[bad].mean()) if bad.any() else None,
                        "mvi_mean_nT": float(rb[bad].mean()) if bad.any() else None,
                        "mvi_max_nT": float(rb[bad].max()) if bad.any() else None,
                        "extent_km": [float(locs[bad, 0].min() / 1e3), float(locs[bad, 0].max() / 1e3),
                                      float(locs[bad, 1].min() / 1e3), float(locs[bad, 1].max() / 1e3)]
                        if bad.any() else None}}
    for k, p in (("induced", MAGNETIC / "beta1"), ("mvi", MAGNETIC / "beta1_mvi")):
        if (p / "run.json").exists():
            out[k]["run"] = json.loads((p / "run.json").read_text())
    return out


def main():
    FIGS.mkdir(exist_ok=True)
    numbers = {"full": {}, "low": {}, "magnetic": magnetic_pair()}
    for old, new in PAIRS:
        numbers["full"][new] = {"before": measures(DATA / "ec2_runs" / old),
                                "after": measures(DATA / "ec2_runs_fixed" / new)}
        numbers["low"][new] = {"before": measures(DATA / "lowres_runs" / old),
                               "after": measures(DATA / "lowres_runs_fixed" / new)}
    path = FIGS / "fixes_numbers.json"
    path.write_text(json.dumps(numbers, indent=1, default=float), encoding="utf-8")
    for level in ("low", "full"):
        print(f"== {level}")
        for k, v in numbers[level].items():
            row = []
            for when in ("before", "after"):
                m = v[when]
                if m is None:
                    row.append(f"{when}: -")
                    continue
                row.append(f"{when}: chi2 {m['gravity']['chi2']:.2f}/{m['magnetics']['chi2']:.2f} "
                           f"core {m['gravity']['core']:.0%}/{m['magnetics']['core']:.0%} "
                           f"below {m['gravity']['below']:.0%}/{m['magnetics']['below']:.0%}"
                           + (f" stopped" if m.get("stopped") else ""))
            print(f"{k:28s} " + " | ".join(row))
    if numbers["magnetic"]:
        print("== magnetic", json.dumps(numbers["magnetic"]["underfit"]))
    print(path)


if __name__ == "__main__":
    main()
