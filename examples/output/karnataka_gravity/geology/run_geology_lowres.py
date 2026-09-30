"""Karnataka gravity at 2 km with the terrain from the NGPM station elevations, without and
with the geology constraints of karnataka_geology.json (rock samples + iron boreholes).

    py examples/output/karnataka_gravity/geology/run_geology_lowres.py [DATA_DIR]

DATA_DIR is the karnataka_ap_gravity folder (default: ~/Desktop/karnataka_ap_gravity); the
files are copied into ./data next to this script.  Writes result_<name>.zip, a viewer with
both runs (karnataka_geology_viewer.html) and comparison.json.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from geoinv3d.cloud.worker import pack_result, run_data_pipeline   # noqa: E402
from geoinv3d.viz.result_workflow import build_workflow, load_result, result_mesh   # noqa: E402

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "OneDrive - Imperial College London" \
    / "Desktop" / "karnataka_ap_gravity"
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
for rel in ("GEOTIFF/NGPM_BA.tiff", "ASCII/combined_NGPM_gravity.csv",
            "Physical_properties_rock_samples.csv"):
    if not (DATA / Path(rel).name).exists():
        shutil.copy(SRC / rel, DATA / Path(rel).name)
for f in (SRC / "Drill").glob("DRILLING_BOREHOLE_DETAILS_STATE_KARNATAKA.*"):
    if not (DATA / f.name).exists():
        shutil.copy(f, DATA / f.name)

BASE = {
    "method_type": "gravity", "inversion_mode": "single",
    "datasets": [{"type": "gravity", "method": "gravity", "files": ["NGPM_BA.tiff"],
                  "component": "gz", "gz_convention": "positive_down", "noise_pct": 0.0,
                  "noise_floor": 0.5, "regional": {"method": "polynomial", "order": 2},
                  "decimate_spacing_m": 2000}],
    "data_file": "NGPM_BA.tiff", "aoi": [641000, 711000, 1634000, 1704000],
    "topography": {"file": "combined_NGPM_gravity.csv"},
    "mesh_type": "tensor", "core_cell_m": 2000.0, "core_cell_z_m": 1000.0,
    "depth_core_m": 10000.0, "pad_distance_m": 20000.0,
    "param_mode": "manual", "regularization_type": "sparse", "norms": [0, 2, 2, 2],
    "alpha_s": 1.0, "alpha_x": 1, "alpha_y": 1, "alpha_z": 1,
    "depth_weighting": "depth", "depth_weighting_exponent": 1.0,
    "bounds_lower": -0.3, "bounds_upper": 0.8, "max_iter": 20, "max_irls_iterations": 10,
}
GEOLOGY = json.loads((HERE / "karnataka_geology.json").read_text())

runs, table = [], {}
for name, extra in (("unconstrained", {}), ("geology", {"geology": GEOLOGY})):
    t = time.time()
    result = run_data_pipeline({**BASE, "task_id": f"karnataka-2km-{name}", **extra}, str(DATA))
    path = pack_result(result, str(HERE / f"result_{name}.zip"))
    run = load_result(path)
    run["_name"] = f"2 km · {name}"
    run["_backend"] = "local"
    runs.append(run)
    cc = result_mesh(result).cell_centers
    act = result["active_cells"]
    ground_top = result["topography"]["elevation_max"]
    m = np.zeros(len(act))
    m[act] = result["recovered_model"]
    d = result["data"]
    res = d["observed"] - d["predicted"]
    # density in the top kilometre below the ground, where the constraints put units
    from geoinv3d.cloud.worker import _load_topography
    surface = _load_topography(BASE, str(DATA), result["crs"])[0]
    depth = surface(cc[:, 0], cc[:, 1]) - cc[:, 2]
    top1km = act & (depth > 0) & (depth < 1000)
    entry = {"seconds": round(time.time() - t), "chi2_per_datum": float(np.mean((res / d["std"]) ** 2)),
             "rms_mgal": float(np.sqrt(np.mean(res ** 2))),
             "top_1km_mean": float(m[top1km].mean()), "top_1km_max": float(m[top1km].max())}
    if "geology" in result:
        ref = np.zeros(len(act))
        ref[act] = result["reference_model"]
        con = act & (ref != 0)
        entry.update(constrained_cells=int(result["geology"]["n_constrained"]),
                     units=result["geology"]["units"],
                     constrained_reference_mean=float(ref[con].mean()),
                     constrained_model_mean=float(m[con].mean()))
        runs_free = table.get("unconstrained")
        if runs_free is not None:   # the same cells in the run without constraints
            free = np.zeros(len(act))
            free[act] = load_result(HERE / "result_unconstrained.zip")["_model"]
            entry["same_cells_unconstrained_mean"] = float(free[con].mean())
    table[name] = entry
    print(name, json.dumps({k: v for k, v in entry.items() if k != "units"}))

(HERE / "comparison.json").write_text(json.dumps(table, indent=1))
wf = build_workflow(runs)
(HERE / "karnataka_geology.geoinv3d.json").write_text(json.dumps(wf))
from geoinv3d.viz.serve_dag import generate_viewer   # noqa: E402
print(generate_viewer(str(HERE / "karnataka_geology.geoinv3d.json"),
                      str(HERE / "karnataka_geology_viewer.html")))
