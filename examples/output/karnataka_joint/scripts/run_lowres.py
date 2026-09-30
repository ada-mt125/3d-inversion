"""The couplings of the joint inversion on the 2 km mesh, locally (a few minutes each).

    py examples/output/karnataka_joint/scripts/run_lowres.py [--out DIR] [--bounds KEY] [coupling ...]

2 km x 2 km x 500 m cells, every second node (1,296 gravity + 1,296 magnetic data), the
ground from the DEM.  Results go to data/lowres_runs/<coupling>/ (or DIR/<coupling>/, e.g.
data/lowres_runs_fixed for the runs with the fixes of 30 September evening).  --bounds rho35
(a key of joint_params.DENSITY_BOUNDS) sets the density bounds from the rock samples; its runs
go to data/lowres_bounds/<key>/<coupling>/ unless --out is given.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parents[2]))
sys.path.insert(0, str(HERE))
from geoinv3d.cloud.worker import pack_result, result_metadata_json, run_data_pipeline   # noqa: E402
from joint_params import COUPLINGS, VARIANTS, base   # noqa: E402

INPUTS = ROOT.parent / "karnataka_inputs"


def main():
    args = sys.argv[1:]
    out, bounds = None, None
    while args[:1] in (["--out"], ["--bounds"]):
        if args[0] == "--out":
            out = Path(args[1])
        else:
            bounds = args[1]
        args = args[2:]
    if out is None:
        out = ROOT / "data" / ("lowres_runs" if bounds is None else f"lowres_bounds/{bounds}")
    variant = {} if bounds is None else VARIANTS[bounds]
    tag = "" if bounds is None else f"-{bounds}"
    for name in args or list(COUPLINGS):
        dest = out / name
        if (dest / "result.zip").exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        t = time.time()
        params = {**base(2000.0, 500.0, decimate=2000, **variant),
                  **COUPLINGS[name], "task_id": f"karnataka-joint-2km{tag}-{name}"}
        result = run_data_pipeline(params, str(INPUTS))
        pack_result(result, str(dest / "result.zip"))
        (dest / "result.json").write_text(result_metadata_json(result), encoding="utf-8")
        (dest / "run.json").write_text(json.dumps({"seconds": round(time.time() - t)}))
        print(name, f"{time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
