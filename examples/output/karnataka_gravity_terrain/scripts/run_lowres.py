"""The 16-run 2 km study of the flat-earth report, repeated with terrain (local, about a minute each).

    py examples/output/karnataka_gravity_terrain/scripts/run_lowres.py [name ...]

2 km x 2 km x 500 m cells, every second node of the 1 km data (1,296 points), the ground from
the DEM, the Bouguer anomaly with the terrain correction.  Results go to data/lowres_runs/<name>/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))
from geoinv3d.cloud.worker import pack_result, result_metadata_json, run_data_pipeline   # noqa: E402

INPUTS = ROOT.parent / "karnataka_inputs"
BASE = {
    "method_type": "gravity", "inversion_mode": "single", "crs": "EPSG:32643",
    "datasets": [{"method": "gravity", "files": ["gravity_complete_1km.csv"], "component": "gz",
                  "noise_pct": 0.0, "noise_floor": 0.5, "regional": {"method": "polynomial", "order": 2},
                  "decimate_spacing_m": 2000}],
    "topography": {"file": "dem_utm43n_450m.tif"}, "mesh_type": "tensor",
    "core_cell_m": 2000.0, "core_cell_z_m": 500.0, "depth_core_m": 10000.0, "pad_distance_m": 20000.0,
    "param_mode": "manual", "max_iter": 30, "max_irls_iterations": 30, "beta0_ratio": 1.0,
    "cooling_factor": 2.0, "use_preconditioner": True, "bounds_lower": -0.2, "bounds_upper": 0.5,
    "regularization_type": "sparse", "alpha_x": 1, "alpha_y": 1, "alpha_z": 1,
}


def v(alpha_s, norms, dw=None):
    out = {"alpha_s": alpha_s, "norms": norms}
    if dw is not None:
        out.update(depth_weighting="depth", depth_weighting_exponent=dw)
    return out


VARIANTS = {
    "base": v(1e-4, [0, 2, 2, 1]), "as1e-2": v(1e-2, [0, 2, 2, 1]), "as0.1": v(0.1, [0, 2, 2, 1]),
    "as1": v(1.0, [0, 2, 2, 1]), "as1_p0111": v(1.0, [0, 1, 1, 1]), "as1_p0222": v(1.0, [0, 2, 2, 2]),
    "as1_p0220": v(1.0, [0, 2, 2, 0]), "as1e-4_p0222": v(1e-4, [0, 2, 2, 2]),
    "as1_p0222_dw0.5": v(1.0, [0, 2, 2, 2], 0.5), "as1_p0222_dw1": v(1.0, [0, 2, 2, 2], 1.0),
    "as1_p0222_dw1.5": v(1.0, [0, 2, 2, 2], 1.5), "as1_p0222_dw2": v(1.0, [0, 2, 2, 2], 2.0),
    "as1_p0221_dw1": v(1.0, [0, 2, 2, 1], 1.0), "as1_p0111_dw1": v(1.0, [0, 1, 1, 1], 1.0),
    "as0.1_p0222_dw1": v(0.1, [0, 2, 2, 2], 1.0), "base_dw1": v(1e-4, [0, 2, 2, 1], 1.0),
}


def main():
    names = sys.argv[1:] or list(VARIANTS)
    for name in names:
        dest = ROOT / "data" / "lowres_runs" / name
        if (dest / "result.zip").exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        t = time.time()
        result = run_data_pipeline({**BASE, **VARIANTS[name], "task_id": f"karnataka-terrain-2km-{name}"},
                                   str(INPUTS))
        pack_result(result, str(dest / "result.zip"))
        (dest / "result.json").write_text(result_metadata_json(result), encoding="utf-8")
        (dest / "run.json").write_text(json.dumps({"seconds": round(time.time() - t)}))
        print(name, f"{time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
