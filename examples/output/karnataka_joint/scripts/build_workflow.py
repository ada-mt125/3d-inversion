"""The joint runs (full resolution on EC2, and the local 2 km study) in one DAG viewer workflow.

    py examples/output/karnataka_joint/scripts/build_workflow.py [--set v2]

--set v2: the second series (data/ec2_runs_fixed, data/lowres_runs_fixed) and the two magnetic
inversions of it (susceptibility beta1, magnetization vector beta1_mvi), into
karnataka_joint_v2.geoinv3d.json and its viewer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[2]))
from geoinv3d.methods.coupling import coupling_label   # noqa: E402
from geoinv3d.viz.result_workflow import build_workflow, load_result   # noqa: E402
from geoinv3d.viz.serve_dag import generate_viewer   # noqa: E402

ORDER = ["none", "cross_gradient", "joint_total_variation", "linear_correspondence", "pgi", "group_lasso",
         "group_lasso_uncoupled"]


ORDER_V2 = ["none", "cross_gradient", "joint_total_variation", "linear_correspondence", "pgi",
            "group_lasso_depth", "group_lasso_depth_uncoupled"]
MAGNETIC = ROOT.parent / "karnataka_magnetic" / "data" / "ec2_runs"


def _study_settings(r):
    """The settings the runs of this study were given (scripts/joint_params.py): each model's
    own regularization, which results written before the pipeline recorded it (30 September)
    carry only as the task's defaults."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from joint_params import GRAVITY_REG, MAGNETIC_REG
    s = r.setdefault("settings", {})
    shared = {k: v for k, v in GRAVITY_REG.items() if MAGNETIC_REG.get(k) == v
              and k in ("regularization_type", "alpha_s", "norms", "depth_weighting", "depth_weighting_exponent")}
    if s.get("regularization_type") == "group_lasso":
        if s.get("gl_weighting") == "depth":
            s["depth_weighting"], s["depth_weighting_exponent"] = "depth", shared.get("depth_weighting_exponent")
        else:
            s["depth_weighting"] = s.get("gl_weighting", "sensitivity")
    elif s.get("regularization_type") != "pgi":
        s.update(shared)


def main():
    v2 = "--set" in sys.argv and sys.argv[sys.argv.index("--set") + 1] == "v2"
    folders = ((("ec2_runs_fixed", "1 km", "ec2"), ("lowres_runs_fixed", "2 km", "local")) if v2
               else (("ec2_runs", "1 km", "ec2"), ("lowres_runs", "2 km", "local")))
    runs = []
    for folder, prefix, backend in folders:
        for key in (ORDER_V2 if v2 else ORDER):
            d = ROOT / "data" / folder / key
            if not (d / "result.zip").exists():
                continue
            r = load_result(d)
            if v2:
                _study_settings(r)
            kind = (r.get("coupling") or {}).get("kind") or key
            r["_name"], r["_backend"] = f"{prefix}: {coupling_label(kind)}", backend
            runs.append(r)
    if v2:   # the magnetic data alone: along the present field, and as a magnetization vector
        for key, label in (("beta1", "susceptibility"), ("beta1_mvi", "magnetization vector (MVI)")):
            d = MAGNETIC / key
            if (d / "result.zip").exists():
                r = load_result(d)
                r.setdefault("settings", {})["magnetization"] = "vector" if key.endswith("mvi") else "induced"
                r["_name"], r["_backend"] = f"1 km magnetics: {label}", "ec2"
                runs.append(r)
    dest = ROOT / ("karnataka_joint_v2.geoinv3d.json" if v2 else "karnataka_joint.geoinv3d.json")
    dest.write_text(json.dumps(build_workflow(runs), separators=(",", ":")), encoding="utf-8")
    print(len(runs), "runs ->", generate_viewer(str(dest)))


if __name__ == "__main__":
    main()
