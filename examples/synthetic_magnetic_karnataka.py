"""Synthetic magnetic test in the Karnataka area: two blocks and a dipping dyke.

    python examples/synthetic_magnetic_karnataka.py

Writes ``examples/output/synthetic_magnetic/karnataka_mag_synthetic.csv`` (x, y, z, tmi):
TMI on a 1 km grid over the 70 x 70 km area of the gravity study, 80 m above flat ground (the
survey's flight height), with Gaussian noise of 2 % + 1 nT.  The inducing field is IGRF 2020 at
the centre of the area (15.09 N, 76.64 E): 42,100 nT, inclination 19.3 deg, declination -0.9 deg
from true north, i.e. -1.4 deg from UTM grid north (grid convergence +0.4 deg).

The model (susceptibility, SI):
    A  shallow block  E 650-660 km, N 1680-1690 km, 1-3 km deep       0.03
    B  deep block     E 685-697 km, N 1642-1654 km, 4-7 km deep       0.04
    C  dyke           N 1650-1695 km, 0.5-6 km deep, 2 km thick,      0.05
                      top centred at E 668 km, dipping 45 deg to the east
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

AOI = (641000.0, 711000.0, 1634000.0, 1704000.0)      # W, E, S, N (UTM 43N)
FIELD = (42100.0, 19.3, -1.4)                          # nT, inclination, declination (grid north)
HEIGHT = 80.0                                          # stations above flat ground at z = 0
SPACING = 1000.0
NOISE_PCT, NOISE_FLOOR = 0.02, 1.0
OUT = REPO / "examples" / "output" / "synthetic_magnetic"

BODIES = [
    {"name": "A shallow block", "kind": "block", "x": (650e3, 660e3), "y": (1680e3, 1690e3),
     "depth": (1000.0, 3000.0), "chi": 0.03},
    {"name": "B deep block", "kind": "block", "x": (685e3, 697e3), "y": (1642e3, 1654e3),
     "depth": (4000.0, 7000.0), "chi": 0.04},
    {"name": "C dipping dyke", "kind": "dyke", "x_top": 668e3, "y": (1650e3, 1695e3),
     "depth": (500.0, 6000.0), "dip_deg": 45.0, "thickness": 2000.0, "chi": 0.05},
]


def body_masks(cc: np.ndarray) -> dict[str, np.ndarray]:
    """Which points (x, y, z; z up, ground at 0) lie in each body."""
    x, y, depth = cc[:, 0], cc[:, 1], -cc[:, 2]
    masks = {}
    for b in BODIES:
        in_y = (y >= b["y"][0]) & (y <= b["y"][1])
        in_z = (depth >= b["depth"][0]) & (depth <= b["depth"][1])
        if b["kind"] == "block":
            in_x = (x >= b["x"][0]) & (x <= b["x"][1])
        else:   # centre line moves east by depth / tan(dip)
            centre = b["x_top"] + (depth - b["depth"][0]) / np.tan(np.radians(b["dip_deg"]))
            in_x = np.abs(x - centre) <= b["thickness"] / 2
        masks[b["name"]] = in_x & in_y & in_z
    return masks


def true_susceptibility(cc: np.ndarray) -> np.ndarray:
    """The model at the points ``cc`` (e.g. the cell centres of any mesh)."""
    chi = np.zeros(len(cc))
    for b in BODIES:
        chi[body_masks(cc)[b["name"]]] = b["chi"]
    return chi


def stations() -> np.ndarray:
    xs = np.arange(AOI[0], AOI[1] + 1, SPACING)
    ys = np.arange(AOI[2], AOI[3] + 1, SPACING)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, HEIGHT)])


def forward(locs: np.ndarray) -> np.ndarray:
    """TMI of the model on a 500 m x 250 m mesh (only the magnetic cells are active)."""
    from geoinv3d.datamodel.mesh import Mesh3D
    from geoinv3d.datamodel.survey import SurveyData
    from geoinv3d.methods.magnetics import MagneticsMethod
    h, dz, depth = 500.0, 250.0, 8000.0
    nx, ny, nz = int((AOI[1] - AOI[0]) / h), int((AOI[3] - AOI[2]) / h), int(depth / dz)
    mesh = Mesh3D.uniform(nx, ny, nz, h, h, dz, origin=(AOI[0], AOI[2], -depth))
    chi = true_susceptibility(mesh.to_discretize().cell_centers)
    active = chi > 0
    survey = SurveyData(locations=locs, observed=np.zeros(len(locs)), std=np.ones(len(locs)))
    sim = MagneticsMethod(inducing_field=FIELD).make_simulation_active(mesh, survey, active)
    return np.asarray(sim.dpred(chi[active]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    locs = stations()
    tmi = forward(locs)
    rng = np.random.default_rng(2026)
    noisy = tmi + rng.normal(scale=NOISE_PCT * np.abs(tmi) + NOISE_FLOOR)
    path = OUT / "karnataka_mag_synthetic.csv"
    rows = "\n".join(f"{x:.1f},{y:.1f},{z:.1f},{v:.4f}" for (x, y, z), v in zip(locs, noisy))
    path.write_text("x,y,z,tmi\n" + rows + "\n", encoding="utf-8")
    (OUT / "model.json").write_text(json.dumps({
        "aoi": AOI, "inducing_field": FIELD, "height_m": HEIGHT, "spacing_m": SPACING,
        "noise": {"pct": NOISE_PCT, "floor_nT": NOISE_FLOOR, "seed": 2026}, "bodies": BODIES},
        indent=1), encoding="utf-8")
    print(f"{len(locs)} stations -> {path}")
    print(f"TMI {tmi.min():.1f} .. {tmi.max():.1f} nT (noise-free), rms noise "
          f"{np.sqrt(np.mean((noisy - tmi) ** 2)):.2f} nT")


if __name__ == "__main__":
    main()
