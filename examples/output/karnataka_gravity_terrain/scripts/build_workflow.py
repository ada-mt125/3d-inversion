"""All runs of the gravity comparison with terrain (8 full-resolution EC2 runs, 16 runs of the
2 km study) in one DAG viewer workflow.

    py examples/output/karnataka_gravity_terrain/scripts/build_workflow.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[2]))
from geoinv3d.viz.result_workflow import build_workflow, load_result   # noqa: E402
from geoinv3d.viz.serve_dag import generate_viewer   # noqa: E402

FULL = {"original_sparse": "Original sparse (α_s=1e-4, sensitivity)", "l1l2_irls": "L1–L2 (IRLS)",
        "as1_beta0.5": "sparse α_s=1, β=0.5", "as1_beta1": "sparse α_s=1, β=1",
        "as1_beta1.5": "sparse α_s=1, β=1.5", "as0.1_beta1": "sparse α_s=0.1, β=1",
        "as1_beta1_no_tc": "sparse α_s=1, β=1, no terrain correction",
        "as1_beta1_flat": "sparse α_s=1, β=1, flat earth, no terrain correction"}


def lowres_name(run):
    s = run["settings"]
    w = f"β={s['depth_weighting_exponent']:g}" if s.get("depth_weighting") == "depth" else "sens"
    norms = ",".join(f"{n:g}" for n in s["norms"])
    return f"2 km: α_s={s['alpha_s']:g} [{norms}] {w}"


def main():
    runs = []
    for key, name in FULL.items():
        r = load_result(ROOT / "data" / "ec2_runs" / key)
        r["_name"], r["_backend"] = f"1 km: {name}", "ec2"
        runs.append(r)
    for d in sorted((ROOT / "data" / "lowres_runs").iterdir()):
        if (d / "result.zip").exists():
            r = load_result(d)
            r["_name"], r["_backend"] = lowres_name(r), "local"
            runs.append(r)
    dest = ROOT / "karnataka_gravity_terrain.geoinv3d.json"
    dest.write_text(json.dumps(build_workflow(runs), separators=(",", ":")), encoding="utf-8")
    print(len(runs), "runs ->", generate_viewer(str(dest)))


if __name__ == "__main__":
    main()
