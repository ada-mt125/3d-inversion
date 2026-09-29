"""Turn cloud job results into a DAG workflow and a viewer page (command line).

    python deploy/result_to_viewer.py RESULT [RESULT ...] [--data g.csv]
        [--true-block WIDTH] [--out DIR] [--name NAME]

Each RESULT is a result directory (``~/.geoinv3d/results/<task>/``) or a
result.zip; see geoinv3d.viz.result_workflow.  Results from workers older
than data.npz need ``--data`` (the observed x,y,value csv): the predicted
data are then forward-modelled here.  ``--true-block WIDTH`` adds the
synthetic block of deploy/ec2_smoke_test.py.  The Jobs page of the upload
wizard does the same through the API ("View in DAG").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "deploy"))

from geoinv3d.viz.result_workflow import build_workflow, full_model, load_result, result_mesh  # noqa: E402


def add_data_from_csv(runs, csv_path: Path) -> None:
    """Old results: observed data from the csv, predicted by forward modelling."""
    from geoinv3d.datamodel.mesh import Mesh3D
    from geoinv3d.datamodel.survey import SurveyData
    from geoinv3d.methods.gravity import GravityMethod
    table = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    locs = np.column_stack([table[:, 0], table[:, 1], np.zeros(len(table))])
    observed = table[:, 2]
    ds = runs[0]["datasets"][0]
    std = ds["noise_pct"] * np.abs(observed) + ds["noise_floor"]
    mesh = result_mesh(runs[0])
    sim = None
    for run in runs:
        if "_data" in run:
            continue
        if sim is None:
            print("Forward-modelling the predicted data locally ...", flush=True)
            survey = SurveyData(locations=locs, observed=observed, std=std)
            sim = GravityMethod().make_simulation(Mesh3D.from_discretize(mesh), survey)
        run["_data"] = {"locations": locs, "observed": observed, "std": std,
                        "predicted": sim.dpred(full_model(run, mesh))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--true-block", type=float, default=None, metavar="WIDTH")
    ap.add_argument("--out", type=Path, default=REPO / "examples" / "output" / "ec2_runs")
    ap.add_argument("--name", default="ec2_run")
    args = ap.parse_args()

    from geoinv3d.viz.serve_dag import generate_viewer
    runs = [load_result(r) for r in args.results]
    if any("_data" not in r for r in runs):
        if args.data is None:
            sys.exit("These results have no data.npz: pass --data OBSERVED.csv")
        add_data_from_csv(runs, args.data)
    true_model = None
    if args.true_block:
        from ec2_smoke_test import DENSITY, block_mask
        true_model = block_mask(result_mesh(runs[0]).cell_centers, args.true_block) * DENSITY
    workflow = build_workflow(runs, true_model, true_label="True block (0.3 g/cc)")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.name}.geoinv3d.json"
    path.write_text(json.dumps(workflow, separators=(",", ":")), encoding="utf-8")
    print(f"workflow: {path}\nviewer:   {generate_viewer(str(path))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
