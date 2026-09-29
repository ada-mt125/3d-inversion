"""All runs of the comparison (6 full-resolution EC2 runs, 16 runs of the 2 km study) in one DAG viewer workflow.

    py examples/output/karnataka_gravity/scripts/build_workflow.py

Writes depth_study/karnataka_depth_study.geoinv3d.json and the self-contained viewer next to it.
"""

from __future__ import annotations

import json

from common import FULL, LOWRES, ROOT, load_full, load_lowres
from geoinv3d.viz.result_workflow import build_workflow
from geoinv3d.viz.serve_dag import generate_viewer


def lowres_name(run):
    s = run["settings"]
    w = f"β={s['depth_weighting_exponent']:g}" if s.get("depth_weighting") == "depth" else "sens"
    norms = ",".join(f"{n:g}" for n in s["norms"])
    return f"2 km: α_s={s['alpha_s']:g} [{norms}] {w}"


def main():
    runs = []
    for key, _, en, *_ in FULL:
        r = load_full(key)
        r["_name"], r["_backend"] = f"1 km: {en}", "ec2"
        runs.append(r)
    for key in LOWRES:
        r = load_lowres(key)
        r["_name"], r["_backend"] = lowres_name(r), "local"
        runs.append(r)
    dest = ROOT / "depth_study" / "karnataka_depth_study.geoinv3d.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(build_workflow(runs), separators=(",", ":")), encoding="utf-8")
    print(len(runs), "runs ->", generate_viewer(str(dest)))


if __name__ == "__main__":
    main()
