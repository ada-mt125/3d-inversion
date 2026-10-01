"""The joint study (its three series of full-resolution runs) in one DAG viewer workflow, into
karnataka_joint.geoinv3d.json and its viewer.

    py examples/output/karnataka_joint/scripts/build_workflow.py [--with-2km]

Built by karnataka_inputs/build_all_workflow.py, which builds the gravity, magnetic and joint
viewers alike: the first series (data/ec2_runs), the second (data/ec2_runs_fixed, report v2)
and the runs with the rock-sample constraints (data/ec2_runs_bounds), each in trees of its own;
--with-2km adds the local 2 km runs of each.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    sys.path.insert(0, str(ROOT.parent / "karnataka_inputs"))
    import build_all_workflow
    build_all_workflow.build("joint", "--with-2km" in sys.argv)


if __name__ == "__main__":
    main()
