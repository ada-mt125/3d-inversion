"""A synthetic test on the trials' own mesh and stations: a known model like the window's geology,
its data with the same errors, so that the trials can be scored against the truth.

    py examples/output/block8_regularization/scripts/synthetic.py [FROM_TRIAL]

The model (susceptibility, SI) on the active cells of FROM_TRIAL's mesh (default sp0111):
  1  an iron-formation band striking NW-SE (135°), dipping 70° to the NE, 150 m thick, from the
     ground to 600 m below it, 0.4 SI, across the whole window and beyond its edges
  2  a thinner band, parallel, 1.4 km to the NE: 100 m, to 400 m, 0.2 SI, 3 km long (ends in
     the window)
  3  a deeper block south-west of the bands: 800 m x 800 m, 700-1,200 m below the ground, 0.05 SI

Data: the TMI of that model at the trials' stations (80 m above the ground, every second node)
and at the nodes left out, plus Gaussian noise of the automatic errors (2 % + 1.5 % of the
5-95 % spread), seed 0.  Writes to data/inputs/: synthetic.csv (x, y, z, tmi, the stations the
trials invert), synthetic_holdout.npy (the left-out nodes), synthetic_model.npy and
synthetic_labels.npy (per active cell), synthetic.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trials  # noqa: E402

STRIKE, DIP = 135.0, 70.0          # degrees; dip towards strike + 90 (NE)
BANDS = [  # offset from the window centre across strike (m, + to the NE), thickness, depth, chi, half-length
    {"label": 1, "offset": -600.0, "thick": 150.0, "depth": 600.0, "chi": 0.4, "half_length": None},
    {"label": 2, "offset": 800.0, "thick": 100.0, "depth": 400.0, "chi": 0.2, "half_length": 1500.0},
]
BLOCK = {"label": 3, "centre": (-1300.0, -1300.0), "size": 800.0, "top": 700.0, "bottom": 1200.0, "chi": 0.05}


def true_model(cc, depth):
    """(model, labels) at cell centres ``cc`` (n, 3) with ``depth`` below the ground."""
    x0, y0 = np.mean(trials.WINDOW[:2]), np.mean(trials.WINDOW[2:])
    dx, dy = cc[:, 0] - x0, cc[:, 1] - y0
    s = np.radians(STRIKE)
    along = dx * np.sin(s) + dy * np.cos(s)                  # along strike (to the SE)
    across = dx * np.sin(s - np.pi / 2) + dy * np.cos(s - np.pi / 2)   # towards the dip (NE)
    m = np.zeros(len(cc))
    lab = np.zeros(len(cc), dtype=np.int8)
    for b in BANDS:
        u = across - b["offset"] - depth / np.tan(np.radians(DIP))
        sel = (np.abs(u) <= b["thick"] / 2) & (depth >= 0) & (depth <= b["depth"])
        if b["half_length"]:
            sel &= np.abs(along) <= b["half_length"]
        m[sel], lab[sel] = b["chi"], b["label"]
    bx, by = BLOCK["centre"]
    sel = ((np.abs(dx - bx) <= BLOCK["size"] / 2) & (np.abs(dy - by) <= BLOCK["size"] / 2)
           & (depth >= BLOCK["top"]) & (depth <= BLOCK["bottom"]))
    m[sel], lab[sel] = BLOCK["chi"], BLOCK["label"]
    return m, lab


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "default"
    r = trials.load(src)
    dmesh = trials.mesh_of(r["meta"])
    active = r["active"]
    surf = trials.surface()
    cc = dmesh.cell_centers[active]
    depth = surf(cc[:, 0], cc[:, 1]) - cc[:, 2]
    m, lab = true_model(cc, depth)
    field = trials.inducing_field()
    locs = r["data"]["locations"]
    d = trials.sensitivity(dmesh, active, locs, field, cache=True).astype(float) @ m
    h, G = trials.holdout(r, synthetic=False)
    dh = trials.dot(G, m)
    p5, p95 = np.percentile(d, [5, 95])
    floor = 0.015 * float(p95 - p5)
    rng = np.random.default_rng(0)
    d_noisy = d + rng.normal(size=d.size) * (0.02 * np.abs(d) + floor)
    dh_noisy = dh + rng.normal(size=dh.size) * (0.02 * np.abs(dh) + floor)
    out = trials.INPUTS
    np.savetxt(out / "synthetic.csv", np.column_stack([locs, d_noisy]), delimiter=",",
               header="x,y,z,tmi", comments="", fmt="%.3f")
    np.save(out / "synthetic_holdout.npy", dh_noisy)
    np.save(out / "synthetic_model.npy", m)
    np.save(out / "synthetic_labels.npy", lab)
    vol = dmesh.cell_volumes[active]
    info = {"from_trial": src, "strike": STRIKE, "dip": DIP, "bands": BANDS, "block": BLOCK,
            "n_cells": {int(k): int((lab == k).sum()) for k in (1, 2, 3)},
            "moment": {int(k): float((vol * m)[lab == k].sum()) for k in (1, 2, 3)},
            "data_min": float(d.min()), "data_max": float(d.max()), "spread": float(p95 - p5),
            "noise_floor": floor, "n_data": int(d.size), "n_holdout": int(dh.size)}
    (out / "synthetic.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
