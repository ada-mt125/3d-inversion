"""The six couplings of a joint gravity–magnetic inversion on one synthetic.

Three bodies, all dense (+0.3 g/cc); their susceptibilities are high (0.05 SI), low
(0.01 SI) and none.  Every model is regularized with the L1–L2 elastic net (by IRLS);
PGI and the group lasso bring their own regularization (they replace it).

    py examples/output/coupling_comparison/run_coupling_comparison.py [blocks|dipping] [viewer]

"viewer" rebuilds only the workflow and its viewer from the saved result_*.zip;
a comma-separated list of couplings reruns those only and keeps the others' results.

blocks:  three 200 x 300 x 200 m boxes, 100-300 m deep.
dipping: three tabular intrusions, 120 m wide (horizontally), dipping 55° to the east
         from 75 m to 375 m depth.

Writes, in ./<shape>/: data/ (the CSVs), result_<coupling>.zip, comparison.json,
comparison.md, coupling_comparison_<shape>.geoinv3d.json and its viewer
coupling_comparison_<shape>_viewer.html.  Everything runs locally.
"""
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")   # ρ and χ in the progress lines, also into a file
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from geoinv3d.cloud.worker import pack_result, run_data_pipeline   # noqa: E402
from geoinv3d.datamodel.mesh import Mesh3D   # noqa: E402
from geoinv3d.datamodel.survey import SurveyData   # noqa: E402
from geoinv3d.methods.coupling import coupling_label   # noqa: E402
from geoinv3d.methods.gravity import GravityMethod   # noqa: E402
from geoinv3d.methods.magnetics import MagneticsMethod   # noqa: E402
from geoinv3d.viz.result_workflow import build_workflow, load_result, result_mesh   # noqa: E402
from geoinv3d.viz.serve_dag import generate_viewer   # noqa: E402

SHAPE = sys.argv[1] if len(sys.argv) > 1 else "blocks"
if SHAPE not in ("blocks", "dipping"):
    raise SystemExit("shape: blocks or dipping")
OUT = HERE / SHAPE
DATA = OUT / "data"
DATA.mkdir(parents=True, exist_ok=True)

FIELD = (50000.0, 90.0, 0.0)          # vertical inducing field: each anomaly over its body
RHO, NOISE = 0.3, 0.02                # density contrast (g/cc); noise, share of the peak
BODIES = [("A", "dense, high χ", 300.0, 0.05),
          ("B", "dense, low χ", 750.0, 0.01),
          ("C", "dense, no χ", 1200.0, 0.0)]
Y0, HALF_Y = 600.0, 150.0


def body_masks(cc):
    """Which of the points (n x 3, ground at z = 0) lie in each body."""
    x, y, z = np.asarray(cc, dtype=float).T
    depth = -z
    in_y = np.abs(y - Y0) < HALF_Y
    out = []
    for _, _, xc, _ in BODIES:
        if SHAPE == "blocks":
            m = (np.abs(x - xc) < 100) & in_y & (depth > 100) & (depth < 300)
        else:   # the centre line moves east with depth: (depth - top) / tan(dip)
            top, bottom, dip = 75.0, 375.0, np.radians(55.0)
            x_mid = xc - 100 + (depth - top) / np.tan(dip)
            m = (np.abs(x - x_mid) < 60) & in_y & (depth > top) & (depth < bottom)
        out.append(m)
    return out


def true_models(cc):
    masks = body_masks(cc)
    rho = np.zeros(len(cc))
    chi = np.zeros(len(cc))
    for m, (_, _, _, sus) in zip(masks, BODIES):
        rho[m] = RHO
        chi[m] = sus
    return rho, chi


def make_data():
    """gz (mGal, positive down as field data) and TMI (nT) over the bodies, modelled on a
    25 m mesh of the bodies' region (not the inversion mesh), with 2 % noise."""
    xs, ys = np.arange(0.0, 1501.0, 75.0), np.arange(0.0, 1201.0, 75.0)
    xx, yy = np.meshgrid(xs, ys)
    locs = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, 1.0)])
    fine = Mesh3D.uniform(76, 16, 18, 25.0, 25.0, 25.0, origin=(-50.0, 400.0, -450.0))
    rho, chi = true_models(fine.to_discretize().cell_centers)
    survey = SurveyData(locations=locs, observed=np.zeros(len(locs)), std=np.ones(len(locs)))
    gz = -GravityMethod().make_simulation_full(fine, survey).dpred(rho)
    tmi = MagneticsMethod(inducing_field=FIELD).make_simulation_full(fine, survey).dpred(chi)
    rng = np.random.default_rng(1)
    out = {}
    for name, d in (("gravity", gz), ("magnetics", tmi)):
        sigma = NOISE * float(np.abs(d).max())
        noisy = d + rng.normal(scale=sigma, size=d.size)
        rows = "\n".join(f"{x:.2f},{y:.2f},{z:.2f},{v:.8g}" for (x, y, z), v in zip(locs, noisy))
        (DATA / f"{name}.csv").write_text("x,y,z,value\n" + rows + "\n")
        out[name] = sigma
    return out, len(locs)


SIGMA, N_STATIONS = make_data()
BASE = {
    "method_type": "joint", "inversion_mode": "joint",
    "datasets": [
        {"type": "gravity", "method": "gravity", "files": ["gravity.csv"], "component": "gz",
         "noise_pct": 0.0, "noise_floor": SIGMA["gravity"]},
        {"type": "magnetic", "method": "magnetic", "files": ["magnetics.csv"], "component": "tmi",
         "noise_pct": 0.0, "noise_floor": SIGMA["magnetics"],
         "method_kwargs": {"inducing_field": list(FIELD)}}],
    "joint_methods": ["gravity", "magnetic"],
    "joint_kwargs_list": [{}, {"inducing_field": list(FIELD)}],
    "joint_weights": [1.0, 1.0],
    "topography": {"flat_elevation": 0.0},
    "mesh_type": "tensor", "core_cell_m": 50.0, "core_cell_z_m": 50.0,
    "depth_core_m": 600.0, "pad_distance_m": 600.0,
    "param_mode": "manual", "regularization_type": "l1l2", "l1_ratio": 0.8, "l1l2_solver": "irls",
    "alpha_s": 1.0, "alpha_x": 1.0, "alpha_y": 1.0, "alpha_z": 1.0,
    # 40 IRLS cycles (and room for them): with 20 every L1–L2 run stopped at the cycle limit
    "max_iter": 60, "max_irls_iterations": 40,
}
share = 1 / 50   # each body's part of the core volume, roughly
PGI_UNITS = [{"name": name, "means": {"gravity": RHO, "magnetics": sus}, "proportion": share}
             for name, _, _, sus in BODIES]
COUPLINGS = [
    ("none", {"coupling": "none"}),
    ("cross_gradient", {"coupling": "cross_gradient", "coupling_weight": 1.0}),
    ("joint_total_variation", {"coupling": "joint_total_variation", "coupling_weight": 1.0}),
    # one linear relation for all cells, from the magnetic body's ratio: body C breaks it
    ("linear_correspondence", {"coupling": "linear_correspondence", "coupling_weight": 1.0,
                               "coupling_options": {"slope": RHO / 0.05, "intercept": 0.0}}),
    # the true rock units (full petrophysical information): PGI's best case
    ("pgi", {"coupling": "pgi", "coupling_options": {"units": PGI_UNITS}}),
    ("group_lasso", {"coupling": "group_lasso", "gl_lambda1_selection": "lcurve",
                     "gl_lambda2": 0.3, "gl_data_scaling": "max_ratio"}),
    # the control: the group lasso's solver, L2 and lambda1 rule, each model on its own
    ("group_lasso_uncoupled", {"coupling": "group_lasso", "gl_coupling": "none",
                               "gl_lambda1_selection": "lcurve", "gl_lambda2": 0.3,
                               "gl_data_scaling": "max_ratio"}),
]
NOTES = {
    "none": "one β for both models, no coupling: the reference",
    "linear_correspondence": f"ρ = {RHO / 0.05:g} χ (the high-χ body's ratio) imposed everywhere",
    "pgi": "the true rock units given (full petrophysical information); PGI replaces L1–L2",
    "group_lasso": "Utsugi (2025): L2 (λ2 = 0.3) + group lasso, λ1 at the L-curve corner; replaces L1–L2",
    "group_lasso_uncoupled": "control: L1 + L2 by ADMM with the group lasso run's settings, no pairing, no coupling",
}


def metrics(result, rho_true, chi_true, seconds):
    mesh = result_mesh(result)
    act = result.get("active_cells")    # none: every cell is below the ground
    act = np.ones(mesh.n_cells, dtype=bool) if act is None else np.asarray(act, dtype=bool)
    cc = mesh.cell_centers[act]
    models = result["recovered_models"]
    rho, chi = np.asarray(models["gravity"]), np.asarray(models["magnetics"])
    masks = body_masks(cc)
    core = (cc[:, 0] > -50) & (cc[:, 0] < 1550) & (cc[:, 1] > -50) & (cc[:, 1] < 1250) & (cc[:, 2] > -600)
    outside = core & ~np.any(masks, axis=0)
    # chi^2 of each dataset from its observed and predicted data (every coupling has them)
    chi2 = {k: float(np.sum(((np.asarray(v["observed"]) - np.asarray(v["predicted"]))
                             / np.asarray(v["std"])) ** 2)) for k, v in result["joint_data"].items()}
    out = {"seconds": round(seconds), "chi2": {k: round(v, 1) for k, v in chi2.items()},
           "n_data": N_STATIONS, "bodies": {}}
    for (name, desc, _, sus), m in zip(BODIES, masks):
        out["bodies"][name] = {"true": [RHO, sus], "rho_mean": round(float(rho[m].mean()), 4),
                               "chi_mean": round(float(chi[m].mean()), 5)}
    out["rho_outside_mean_abs"] = round(float(np.abs(rho[outside]).mean()), 5)
    out["chi_outside_mean_abs"] = round(float(np.abs(chi[outside]).mean()), 6)
    rt, ct = rho_true[act], chi_true[act]
    out["rho_error_rms"] = round(float(np.sqrt(np.mean((rho - rt)[core] ** 2))), 5)
    out["chi_error_rms"] = round(float(np.sqrt(np.mean((chi - ct)[core] ** 2))), 6)
    out["corr_rho_chi"] = round(float(np.corrcoef(rho[core], chi[core])[0, 1]), 3)
    if "coupling" in result:
        c = result["coupling"]
        out["coupling"] = {k: v for k, v in c.items() if k in ("kind", "label", "weight", "multipliers")}
    if "pgi" in result:   # which unit each cell ended in, against the truth
        mem = np.asarray(result["pgi"]["membership"])
        names = [u["name"] for u in result["pgi"]["units"]]
        truth = np.zeros(len(cc), dtype=int)
        for (name, *_), m in zip(BODIES, masks):
            truth[m] = names.index(name)
        out["pgi_cells_right"] = {name: f"{int(np.sum((mem == names.index(name)) & m))}/{int(m.sum())}"
                                  for (name, *_), m in zip(BODIES, masks)}
        out["pgi_false_body_cells"] = int(np.sum((mem > 0) & (truth == 0) & core))
    return out


VIEWER_ONLY = len(sys.argv) > 2 and sys.argv[2] == "viewer"
ONLY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 and not VIEWER_ONLY else None
SAVED = json.loads((OUT / "comparison.json").read_text(encoding="utf-8")) if ONLY else {}
runs, table, truth = [], {}, None
for key, extra in (COUPLINGS if not VIEWER_ONLY else []):
    if ONLY is not None and key not in ONLY:   # keep the saved result
        run = load_result(OUT / f"result_{key}.zip")
        run["_name"] = f"{SHAPE} · {coupling_label(key if key != 'group_lasso_uncoupled' else 'group_lasso_uncoupled')}"
        run["_backend"] = "local"
        runs.append(run)
        table[key] = SAVED[key]
        continue
    t = time.time()
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        result = run_data_pipeline({**BASE, "task_id": f"coupling-{SHAPE}-{key}", **extra}, str(DATA))
    (OUT / f"log_{key}.txt").write_text(log.getvalue(), encoding="utf-8")
    path = pack_result(result, str(OUT / f"result_{key}.zip"))
    if truth is None:   # the true models on the inversion mesh (all runs share it)
        truth = true_models(result_mesh(result).cell_centers)
    entry = metrics(result, *truth, time.time() - t)
    entry["note"] = NOTES.get(key, "")
    table[key] = entry
    run = load_result(path)
    run["_name"] = f"{SHAPE} · {coupling_label(key if key != 'group_lasso_uncoupled' else 'group_lasso_uncoupled')}"
    run["_backend"] = "local"
    runs.append(run)
    print(f"{key:24s} {entry['seconds']:4d}s chi2={entry['chi2']} "
          + " ".join(f"{b}: ρ {v['rho_mean']:.3f} χ {v['chi_mean']:.4f}" for b, v in entry["bodies"].items()),
          flush=True)

if truth is None and runs:
    truth = true_models(result_mesh(runs[0]).cell_centers)
if VIEWER_ONLY:   # the saved results, as they are
    for key, _ in COUPLINGS:
        run = load_result(OUT / f"result_{key}.zip")
        run["_name"] = f"{SHAPE} · {coupling_label(key if key != 'group_lasso_uncoupled' else 'group_lasso_uncoupled')}"
        run["_backend"] = "local"
        runs.append(run)
    truth = true_models(result_mesh(runs[0]).cell_centers)
    table = json.loads((OUT / "comparison.json").read_text(encoding="utf-8"))
(OUT / "comparison.json").write_text(json.dumps(table, indent=1, ensure_ascii=False), encoding="utf-8")

# a table to read next to the viewer
lines = [f"# Six couplings, {SHAPE}: three dense bodies (+{RHO} g/cc), χ = 0.05 / 0.01 / 0 SI",
         "", f"{N_STATIONS} stations of gz and TMI (vertical field), {NOISE:.0%} noise; L1–L2 (IRLS, "
         "L1 share 0.8) per model; unit-free coupling weight 1.", "",
         "| coupling | A ρ / χ (0.3 / 0.05) | B ρ / χ (0.3 / 0.01) | C ρ / χ (0.3 / 0) | "
         "ρ rms error | χ rms error | χ² grav / mag | s | note |", "|---|---|---|---|---|---|---|---|---|"]
for key, e in table.items():
    b = e["bodies"]
    cells = " | ".join(f"{b[n]['rho_mean']:.3f} / {b[n]['chi_mean']:.4f}" for n in ("A", "B", "C"))
    chi2 = " / ".join(f"{v:.0f}" for v in e["chi2"].values())
    lines.append(f"| {coupling_label(key)} | {cells} | {e['rho_error_rms']:.4f} | {e['chi_error_rms']:.5f} "
                 f"| {chi2} (N = {N_STATIONS}) | {e['seconds']} | {e.get('note', '')} |")
if "pgi" in table:
    lines += ["", f"PGI cells in the right unit: {table['pgi']['pgi_cells_right']}; cells called a body "
              f"outside the bodies: {table['pgi']['pgi_false_body_cells']}."]
(OUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

wf = build_workflow(runs, true_model={"gravity": truth[0], "magnetics": truth[1]}, true_label="True model")
name = f"coupling_comparison_{SHAPE}"
(OUT / f"{name}.geoinv3d.json").write_text(json.dumps(wf))
print(generate_viewer(str(OUT / f"{name}.geoinv3d.json"), str(OUT / f"{name}_viewer.html")))
