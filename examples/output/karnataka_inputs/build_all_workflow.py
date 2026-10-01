"""Every run of the three Karnataka studies with terrain (gravity, magnetics, joint) in one
DAG viewer page.

    py examples/output/karnataka_inputs/build_all_workflow.py [--full-only]

Writes karnataka_all_runs.geoinv3d.json and its self-contained viewer
karnataka_all_runs_viewer.html next to this script (--full-only leaves out the 2 km studies).
The joint study has three series, each in trees of its own: the first (data/ec2_runs,
lowres_runs), the second (data/ec2_runs_fixed, lowres_runs_fixed: report v2) and the runs with
the rock-sample constraints (data/ec2_runs_bounds, lowres_bounds: report v2, Section 5).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
sys.path.insert(0, str(OUT.parents[1]))
from geoinv3d.methods.coupling import coupling_label   # noqa: E402
from geoinv3d.viz.result_workflow import build_workflow, load_result   # noqa: E402
from geoinv3d.viz.serve_dag import generate_viewer   # noqa: E402

GRAVITY = {"original_sparse": "original sparse (α_s=1e-4, sensitivity)", "l1l2_irls": "L1–L2 (IRLS)",
           "as1_beta0.5": "sparse α_s=1, β=0.5", "as1_beta1": "sparse α_s=1, β=1",
           "as1_beta1.5": "sparse α_s=1, β=1.5", "as0.1_beta1": "sparse α_s=0.1, β=1",
           "as1_beta1_no_tc": "sparse β=1, no terrain correction",
           "as1_beta1_flat": "sparse β=1, flat earth, no terrain correction"}
MAGNETIC = {"sens": "sparse, sensitivity weighting", "beta0.5": "sparse, β=0.5", "beta1": "sparse, β=1",
            "beta1.5": "sparse, β=1.5", "beta2": "sparse, β=2", "beta3": "sparse, β=3", "l1l2_irls": "L1–L2 (IRLS)",
            "as0.1_beta1": "β=1, α_s=0.1", "beta1_ub0.3": "β=1, upper bound 0.3 SI",
            "beta1_err2": "β=1, error 2 % + 5 nT", "beta1_h500": "β=1, continued to 500 m",
            "beta1_raw80": "β=1, as flown (80 m)", "beta1_flat": "β=1, flat earth",
            "beta1_mvi": "β=1, magnetization vector (MVI)"}
JOINT = ["none", "cross_gradient", "joint_total_variation", "linear_correspondence", "pgi", "group_lasso",
         "group_lasso_uncoupled"]
JOINT_V2 = ["none", "cross_gradient", "joint_total_variation", "linear_correspondence", "pgi",
            "group_lasso_depth", "group_lasso_depth_uncoupled"]


def lowres_name(run):
    s = run["settings"]
    w = f"β={s['depth_weighting_exponent']:g}" if s.get("depth_weighting") == "depth" else "sens"
    norms = ",".join(f"{n:g}" for n in s["norms"])
    return f"α_s={s['alpha_s']:g} [{norms}] {w}"


def add(runs, path, name, backend, study):
    if (path / "result.zip").exists():
        r = load_result(path)
        r["_name"], r["_backend"], r["_study"] = name, backend, study
        if r.get("stopped_early"):
            r["_name"] += " (stopped early)"
        runs.append(r)


def main():
    full_only = "--full-only" in sys.argv
    runs = []
    g = OUT / "karnataka_gravity_terrain" / "data"
    for key, name in GRAVITY.items():
        add(runs, g / "ec2_runs" / key, f"Gravity 1 km: {name}", "ec2", "Gravity study")
    if not full_only:
        for d in sorted((g / "lowres_runs").iterdir()):
            if (d / "result.zip").exists():
                r = load_result(d)
                r["_name"], r["_backend"], r["_study"] = f"Gravity 2 km: {lowres_name(r)}", "local", "Gravity study"
                runs.append(r)
    m = OUT / "karnataka_magnetic" / "data"
    m = m / "ec2_runs" if (m / "ec2_runs").exists() else m / "ec2_trials"
    for key, name in MAGNETIC.items():
        add(runs, m / key, f"Magnetics 1 km: {name}", "ec2", "Magnetic study")
    j = OUT / "karnataka_joint" / "data"
    for folder, prefix, backend in (("ec2_runs", "Joint 1 km", "ec2"), ("lowres_runs", "Joint 2 km", "local")):
        if full_only and folder == "lowres_runs":
            continue
        for key in JOINT:
            d = j / folder / key
            if (d / "result.zip").exists():
                kind = (load_result(d).get("coupling") or {}).get("kind") or key
                add(runs, d, f"{prefix}: {coupling_label(kind)}", backend, "Joint study")
    # the second series (report v2), with each model's own regularization as joint_params gave it
    sys.path.insert(0, str(OUT / "karnataka_joint" / "scripts"))
    from build_workflow import _study_settings
    from joint_params import VARIANTS
    for folder, prefix, backend in (("ec2_runs_fixed", "Joint 1 km", "ec2"),
                                    ("lowres_runs_fixed", "Joint 2 km", "local")):
        if full_only and backend == "local":
            continue
        for key in JOINT_V2:
            d = j / folder / key
            if (d / "result.zip").exists():
                r = load_result(d)
                _study_settings(r)
                kind = (r.get("coupling") or {}).get("kind") or key
                r["_name"], r["_backend"], r["_study"] = (f"{prefix}, 2nd series: {coupling_label(kind)}", backend,
                                                          "Joint study, 2nd series")
                runs.append(r)

    # with the rock-sample constraints: density bounds and depth weighting of joint_params.VARIANTS
    def constrained(key):
        v = VARIANTS[key]
        lo, hi = v["density_bounds"]
        bg, bm = v.get("betas", (1.0, 1.0))
        return f"density {lo:+.2f} / {hi:+.2f} g/cc, β {bg:g} / {bm:g}".replace("-", "−")
    for key in VARIANTS:
        for coupling in ("none", "joint_total_variation"):
            label = coupling_label(coupling)
            add(runs, j / "ec2_runs_bounds" / f"{coupling}_{key}", f"Joint 1 km, constrained: {label}, "
                f"{constrained(key)}", "ec2", "Joint study, rock-sample constraints")
            if not full_only:
                add(runs, j / "lowres_bounds" / key / coupling, f"Joint 2 km, constrained: {label}, "
                    f"{constrained(key)}", "local", "Joint study, rock-sample constraints")
    # the reference runs of the two single studies (sparse, β = 1) overlay each other in 3D:
    # density with the susceptibility as a second layer, and the other way round (the two
    # models of a joint run get each other automatically)
    ref = {r["_name"]: r for r in runs}
    grav, mag = ref.get("Gravity 1 km: sparse α_s=1, β=1"), ref.get("Magnetics 1 km: sparse, β=1")
    if grav and mag and len(grav["_model"]) == len(mag["_model"]):
        grav["_overlay"] = {"model": mag["_model"], "label": "Susceptibility (magnetic study, β=1)", "unit": "SI"}
        mag["_overlay"] = {"model": grav["_model"], "label": "Density (gravity study, β=1)", "unit": "g/cc"}
    dest = HERE / "karnataka_all_runs.geoinv3d.json"
    dest.write_text(json.dumps(build_workflow(runs), separators=(",", ":")), encoding="utf-8")
    viewer = generate_viewer(str(dest), str(HERE / "karnataka_all_runs_viewer.html"))
    print(len(runs), "runs ->", viewer, f"{Path(viewer).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
