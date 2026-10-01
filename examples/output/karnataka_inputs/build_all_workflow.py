"""The DAG viewers of the three Karnataka studies with terrain: gravity, magnetics and the joint
inversion, each in its own folder.

    py examples/output/karnataka_inputs/build_all_workflow.py [--with-2km] [--only gravity,magnetic,joint]

Writes, for each study, its workflow and self-contained viewer into the study's folder:
karnataka_gravity_terrain/karnataka_gravity_terrain.geoinv3d.json (and _viewer.html),
karnataka_magnetic/karnataka_magnetic.geoinv3d.json, karnataka_joint/karnataka_joint.geoinv3d.json.
The runs on the 1 km mesh; --with-2km adds the local 2 km studies.  Each study's
scripts/build_workflow.py builds its own viewer the same way.  The reference gravity and
magnetic runs carry each other's model as a second layer of their 3D view.  The joint study
has three series, each in trees of its own: the first (data/ec2_runs, lowres_runs), the second
(data/ec2_runs_fixed, lowres_runs_fixed: report v2) and the runs with the rock-sample
constraints (data/ec2_runs_bounds, lowres_bounds: report v2, Section 5).
"""

from __future__ import annotations

import importlib.util
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
GRAV_DIR, MAG_DIR, JOINT_DIR = OUT / "karnataka_gravity_terrain", OUT / "karnataka_magnetic", OUT / "karnataka_joint"
# where each study's workflow goes; its viewer is <name>.geoinv3d_viewer.html beside it
DEST = {"gravity": GRAV_DIR / "karnataka_gravity_terrain.geoinv3d.json",
        "magnetic": MAG_DIR / "karnataka_magnetic.geoinv3d.json",
        "joint": JOINT_DIR / "karnataka_joint.geoinv3d.json"}
REF_GRAVITY, REF_MAGNETIC = "as1_beta1", "beta1"   # the reference runs (sparse, β = 1)


def _module(path, name):
    """A script of another study by its path (several are called build_workflow.py)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).parent))
    spec.loader.exec_module(mod)
    return mod


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
        return r


def _magnetic_data():
    m = MAG_DIR / "data"
    return m / "ec2_runs" if (m / "ec2_runs").exists() else m / "ec2_trials"


def _overlay(run, other, label, unit):
    """``other``'s model as a second layer of ``run``'s 3D view (same mesh and cells)."""
    if run is not None and other is not None and len(run["_model"]) == len(other["_model"]):
        run["_overlay"] = {"model": other["_model"], "label": label, "unit": unit}


def gravity_runs(with_2km=False):
    runs, ref = [], None
    g = GRAV_DIR / "data"
    for key, name in GRAVITY.items():
        r = add(runs, g / "ec2_runs" / key, f"Gravity 1 km: {name}", "ec2", "Gravity study")
        ref = r if key == REF_GRAVITY else ref
    if with_2km:
        for d in sorted((g / "lowres_runs").iterdir()):
            if (d / "result.zip").exists():
                r = load_result(d)
                r["_name"], r["_backend"], r["_study"] = f"Gravity 2 km: {lowres_name(r)}", "local", "Gravity study"
                runs.append(r)
    # the reference density with the reference susceptibility as a second layer
    d = _magnetic_data() / REF_MAGNETIC
    if ref is not None and (d / "result.zip").exists():
        _overlay(ref, load_result(d), "Susceptibility (magnetic study, β=1)", "SI")
    return runs


def magnetic_runs(with_2km=False):
    runs, ref = [], None
    for key, name in MAGNETIC.items():
        r = add(runs, _magnetic_data() / key, f"Magnetics 1 km: {name}", "ec2", "Magnetic study")
        ref = r if key == REF_MAGNETIC else ref
    d = GRAV_DIR / "data" / "ec2_runs" / REF_GRAVITY
    if ref is not None and (d / "result.zip").exists():
        _overlay(ref, load_result(d), "Density (gravity study, β=1)", "g/cc")
    return runs


def joint_runs(with_2km=False):
    """The three series of the joint study (the two models of a joint run overlay each other)."""
    runs = []
    j = JOINT_DIR / "data"
    for folder, prefix, backend in (("ec2_runs", "Joint 1 km", "ec2"), ("lowres_runs", "Joint 2 km", "local")):
        if backend == "local" and not with_2km:
            continue
        for key in JOINT:
            d = j / folder / key
            if (d / "result.zip").exists():
                kind = (load_result(d).get("coupling") or {}).get("kind") or key
                add(runs, d, f"{prefix}: {coupling_label(kind)}", backend, "Joint study")
    # the second series (report v2), with each model's own regularization as joint_params gave it
    scripts = JOINT_DIR / "scripts"
    study_settings = _module(scripts / "build_workflow.py", "karnataka_joint_build_workflow")._study_settings
    variants = _module(scripts / "joint_params.py", "joint_params").VARIANTS
    for folder, prefix, backend in (("ec2_runs_fixed", "Joint 1 km", "ec2"),
                                    ("lowres_runs_fixed", "Joint 2 km", "local")):
        if backend == "local" and not with_2km:
            continue
        for key in JOINT_V2:
            d = j / folder / key
            if (d / "result.zip").exists():
                r = load_result(d)
                study_settings(r)
                kind = (r.get("coupling") or {}).get("kind") or key
                r["_name"], r["_backend"], r["_study"] = (f"{prefix}, 2nd series: {coupling_label(kind)}", backend,
                                                          "Joint study, 2nd series")
                runs.append(r)

    # with the rock-sample constraints: density bounds and depth weighting of joint_params.VARIANTS
    def constrained(key):
        v = variants[key]
        lo, hi = v["density_bounds"]
        bg, bm = v.get("betas", (1.0, 1.0))
        return f"density {lo:+.2f} / {hi:+.2f} g/cc, β {bg:g} / {bm:g}".replace("-", "−")
    for key in variants:
        for coupling in ("none", "joint_total_variation"):
            label = coupling_label(coupling)
            add(runs, j / "ec2_runs_bounds" / f"{coupling}_{key}", f"Joint 1 km, constrained: {label}, "
                f"{constrained(key)}", "ec2", "Joint study, rock-sample constraints")
            if with_2km:
                add(runs, j / "lowres_bounds" / key / coupling, f"Joint 2 km, constrained: {label}, "
                    f"{constrained(key)}", "local", "Joint study, rock-sample constraints")
    return runs


STUDIES = {"gravity": gravity_runs, "magnetic": magnetic_runs, "joint": joint_runs}


def build(study, with_2km=False):
    """The workflow and viewer of one study, in its folder; returns the viewer's path."""
    runs = STUDIES[study](with_2km)
    dest = DEST[study]
    dest.write_text(json.dumps(build_workflow(runs), separators=(",", ":")), encoding="utf-8")
    viewer = generate_viewer(str(dest))
    print(f"{study}: {len(runs)} runs ->", viewer, f"{Path(viewer).stat().st_size / 1e6:.0f} MB")
    return viewer


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    only = argv[argv.index("--only") + 1].split(",") if "--only" in argv else list(STUDIES)
    for study in only:
        build(study, "--with-2km" in argv)


if __name__ == "__main__":
    main()
