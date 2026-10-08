"""The regularization trials on the Block-8 window in one DAG viewer: the trials on the real data
and those on the synthetic data, each in a tree of its own; every synthetic trial carries the
true model as a second layer of its 3D view.

    py examples/output/block8_regularization/scripts/build_workflow.py

Writes block8_regularization.geoinv3d.json and block8_regularization.geoinv3d_viewer.html into
the study's folder (the trials finished so far; run it again as more finish).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parents[2]))
sys.path.insert(0, str(HERE))
from geoinv3d.viz.result_workflow import build_workflow, load_result   # noqa: E402
from geoinv3d.viz.serve_dag import generate_viewer                     # noqa: E402
import trials                                                           # noqa: E402

DEST = ROOT / "block8_regularization.geoinv3d.json"


def runs():
    out = []
    true = trials.INPUTS / "synthetic_model.npy"
    true = np.load(true) if true.exists() else None
    for synthetic in (False, True):
        for key, (label, _) in trials.TRIALS.items():
            name = f"syn-{key}" if synthetic else key
            d = trials.RUNS / name
            if not (d / "result.zip").exists():
                continue
            r = load_result(d)
            r["_backend"] = "local"
            if synthetic:
                r["_name"], r["_study"] = f"Synthetic: {label}", "Block-8 window, synthetic test"
                if true is not None and len(true) == len(r["_model"]):
                    r["_overlay"] = {"model": true, "label": "True model (synthetic)", "unit": "SI"}
            else:
                r["_name"], r["_study"] = f"Block-8 5 km: {label}", "Block-8 window, real data"
            out.append(r)
    return out


def map_layers():
    """The boreholes of data/inputs/boreholes.geojson (the GSI drilling collars near the window) as a layer: collars on
    the maps, traces in the 3D view (dag_interactive.html's map layers); and the window."""
    layers = []
    f = trials.INPUTS / "boreholes.geojson"
    if f.exists():
        items = []
        for ft in json.loads(f.read_text(encoding="utf-8"))["features"]:
            p = ft["properties"]
            x, y = ft["geometry"]["coordinates"][:2]
            item = {"x": x, "y": y, "label": p["name"]}
            if p.get("length_m"):
                item.update(depth=p["length_m"], dip=p.get("cl_inclina") or 90.0,
                            azimuth=p.get("bearing") or 0.0)
                if p.get("rl_collar_") is not None:
                    item["z"] = p["rl_collar_"]
            items.append(item)
        layers.append({"name": "Boreholes (GSI): Ramanadurga South, iron ore", "kind": "points",
                       "color": "#00b4d8", "items": items})
    w = trials.WINDOW
    layers.append({"name": "The 5 km window", "kind": "lines", "color": "#adb5bd",
                   "items": [{"xy": [[w[0], w[2]], [w[1], w[2]], [w[1], w[3]], [w[0], w[3]]],
                              "label": "window", "closed": True}]})
    return layers


def main():
    rs = runs()
    workflow = build_workflow(rs)
    workflow["map_layers"] = map_layers()
    DEST.write_text(json.dumps(workflow, separators=(",", ":")), encoding="utf-8")
    viewer = generate_viewer(str(DEST))
    print(f"{len(rs)} runs ->", viewer, f"{Path(viewer).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
