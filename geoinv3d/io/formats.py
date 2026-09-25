"""Model I/O in various formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..datamodel.mesh import Mesh3D
from ..datamodel.model import PhysicalModel, PhysicalProperty


def save_model_npy(model: PhysicalModel, path: str) -> None:
    """Save a model as .npy (values) + .json (metadata)."""
    p = Path(path)
    np.save(p.with_suffix(".npy"), model.values)
    meta = {
        "prop": model.prop.value,
        "name": model.name,
        "mesh": {
            "hx": model.mesh.hx.tolist(),
            "hy": model.mesh.hy.tolist(),
            "hz": model.mesh.hz.tolist(),
            "origin": list(model.mesh.origin),
        },
    }
    with open(p.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def load_model_npy(path: str) -> PhysicalModel:
    """Load a model from .npy (values) + .json (metadata)."""
    p = Path(path)
    values = np.load(p.with_suffix(".npy"))
    with open(p.with_suffix(".json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    mesh = Mesh3D(
        hx=np.array(meta["mesh"]["hx"]),
        hy=np.array(meta["mesh"]["hy"]),
        hz=np.array(meta["mesh"]["hz"]),
        origin=tuple(meta["mesh"]["origin"]),
    )
    return PhysicalModel(
        mesh=mesh,
        values=values,
        prop=PhysicalProperty(meta["prop"]),
        name=meta.get("name", ""),
    )
