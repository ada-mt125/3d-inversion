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


def main():
    rs = runs()
    workflow = build_workflow(rs)
    DEST.write_text(json.dumps(workflow, separators=(",", ":")), encoding="utf-8")
    viewer = generate_viewer(str(DEST))
    print(f"{len(rs)} runs ->", viewer, f"{Path(viewer).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
