"""Inversion task serialization for cloud execution.

Packs all data needed to run an inversion into a self-contained archive:
mesh, model values, survey data, and inversion parameters.  The archive
can be shipped to any machine with SimPEG installed.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class InversionTask:
    """Self-contained description of an inversion job."""

    task_id: str

    # Mesh — uniform (nx/ny/nz + dx/dy/dz) or non-uniform (hx/hy/hz)
    nx: int = 0
    ny: int = 0
    nz: int = 0
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Method
    method_type: str = "gravity"
    method_kwargs: dict = field(default_factory=dict)

    # Regularization: "sparse" (lp-norm IRLS), "l1l2" (elastic net, Utsugi 2019)
    # or anything else for smooth L2
    regularization_type: str = "sparse"

    # Shared inversion parameters
    max_iter: int = 30
    beta0_ratio: float = 1.0
    cooling_factor: float = 2.0
    alpha_s: float = 1e-4
    alpha_x: float = 1.0
    alpha_y: float = 1.0
    alpha_z: float = 1.0

    # Sparse IRLS parameters (from deep-mesh production settings)
    norms: tuple[float, float, float, float] = (0.0, 2.0, 2.0, 1.0)
    irls_cooling_factor: float = 1.1
    max_irls_iterations: int = 30
    use_preconditioner: bool = True

    # L1–L2 (elastic net) mixing: 1 = pure L1, 0 = pure L2 smallness
    l1_ratio: float = 0.5
    # L1–L2 solver: "cda" (Utsugi 2019: coordinate descent along a lambda path,
    # lambda from the L-curve; potential fields) or "irls" (SimPEG IRLS).
    # Weighting "S2" (1/||k_j||, recommended by Utsugi) or "S1" (||k_j||^-1/2).
    l1l2_solver: str = "cda"
    l1l2_weighting: str = "S2"
    lambda_decades: float = 4.0
    lambda_step: float = 0.1

    # Choice of beta: "auto" (L-curve for the L1–L2 CDA path, as in Utsugi
    # 2019; discrepancy otherwise), "discrepancy" (chi^2 = N), "lcurve" or
    # "gcv".  IRLS/smooth sweeps use beta_sweep if given, else the discrepancy
    # beta times beta_sweep_factors.
    beta_selection: str = "auto"
    beta_sweep: Optional[list[float]] = None
    beta_sweep_factors: tuple[float, ...] = tuple(np.logspace(-2, 2, 13))

    # Bounds for ProjectedGNCG (None = unbounded)
    bounds_lower: Optional[float] = None
    bounds_upper: Optional[float] = None

    # Noise model
    noise_pct: float = 0.05
    noise_floor: float = 0.5

    # Joint inversion (multiple methods)
    joint_methods: Optional[list[str]] = None
    joint_kwargs_list: Optional[list[dict]] = None
    joint_weights: Optional[list[float]] = None
    cross_gradient_weight: float = 0.0

    # Arrays stored separately in the archive
    initial_model: Optional[NDArray] = None
    observed_data: Optional[NDArray] = None
    data_std: Optional[NDArray] = None
    station_locations: Optional[NDArray] = None
    active_cells: Optional[NDArray] = None
    hx: Optional[NDArray] = None
    hy: Optional[NDArray] = None
    hz: Optional[NDArray] = None

    # For joint: list of (observed, std, locations) per method
    joint_surveys: Optional[list[dict]] = None
    joint_initial_models: Optional[list[NDArray]] = None

    def to_meta(self) -> dict:
        """Serializable metadata (no large arrays)."""
        d = {
            "task_id": self.task_id,
            "nx": self.nx, "ny": self.ny, "nz": self.nz,
            "dx": self.dx, "dy": self.dy, "dz": self.dz,
            "origin": list(self.origin),
            "method_type": self.method_type,
            "method_kwargs": self.method_kwargs,
            "regularization_type": self.regularization_type,
            "max_iter": self.max_iter,
            "beta0_ratio": self.beta0_ratio,
            "cooling_factor": self.cooling_factor,
            "alpha_s": self.alpha_s,
            "alpha_x": self.alpha_x,
            "alpha_y": self.alpha_y,
            "alpha_z": self.alpha_z,
            "norms": list(self.norms),
            "irls_cooling_factor": self.irls_cooling_factor,
            "max_irls_iterations": self.max_irls_iterations,
            "use_preconditioner": self.use_preconditioner,
            "l1_ratio": self.l1_ratio,
            "l1l2_solver": self.l1l2_solver,
            "l1l2_weighting": self.l1l2_weighting,
            "lambda_decades": self.lambda_decades,
            "lambda_step": self.lambda_step,
            "beta_selection": self.beta_selection,
            "beta_sweep_factors": [float(f) for f in self.beta_sweep_factors],
            "noise_pct": self.noise_pct,
            "noise_floor": self.noise_floor,
        }
        if self.beta_sweep is not None:
            d["beta_sweep"] = [float(b) for b in self.beta_sweep]
        if self.bounds_lower is not None:
            d["bounds_lower"] = self.bounds_lower
        if self.bounds_upper is not None:
            d["bounds_upper"] = self.bounds_upper
        if self.joint_methods:
            d["joint_methods"] = self.joint_methods
            d["joint_kwargs_list"] = self.joint_kwargs_list or []
            d["joint_weights"] = self.joint_weights or []
            d["cross_gradient_weight"] = self.cross_gradient_weight
        return d


def pack_task(task: InversionTask, output_path: str) -> str:
    """Pack an InversionTask into a .zip archive.

    Archive contents:
        meta.json          — task parameters (no arrays)
        initial_model.npy  — starting model values
        observed.npy       — observed data
        std.npy            — data standard deviations
        locations.npy      — station locations (N x 3)
        joint_obs_0.npy, joint_std_0.npy, ... — per-method data (joint)
        joint_model_0.npy, ...                — per-method initial models
    """
    path = Path(output_path)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps(task.to_meta(), indent=2))

        if task.initial_model is not None:
            _write_npy(zf, "initial_model.npy", task.initial_model)
        if task.observed_data is not None:
            _write_npy(zf, "observed.npy", task.observed_data)
        if task.data_std is not None:
            _write_npy(zf, "std.npy", task.data_std)
        if task.station_locations is not None:
            _write_npy(zf, "locations.npy", task.station_locations)
        if task.active_cells is not None:
            _write_npy(zf, "active_cells.npy", task.active_cells)
        if task.hx is not None:
            _write_npy(zf, "hx.npy", task.hx)
        if task.hy is not None:
            _write_npy(zf, "hy.npy", task.hy)
        if task.hz is not None:
            _write_npy(zf, "hz.npy", task.hz)

        if task.joint_surveys:
            for i, survey in enumerate(task.joint_surveys):
                _write_npy(zf, f"joint_obs_{i}.npy", survey["observed"])
                _write_npy(zf, f"joint_std_{i}.npy", survey["std"])
                _write_npy(zf, f"joint_locs_{i}.npy", survey["locations"])

        if task.joint_initial_models:
            for i, m in enumerate(task.joint_initial_models):
                _write_npy(zf, f"joint_model_{i}.npy", m)

    return str(path)


def unpack_task(archive_path: str) -> InversionTask:
    """Unpack a .zip archive into an InversionTask."""
    with zipfile.ZipFile(archive_path, "r") as zf:
        meta = json.loads(zf.read("meta.json"))

        task = InversionTask(
            task_id=meta["task_id"],
            nx=meta.get("nx", 0), ny=meta.get("ny", 0), nz=meta.get("nz", 0),
            dx=meta.get("dx", 0.0), dy=meta.get("dy", 0.0), dz=meta.get("dz", 0.0),
            origin=tuple(meta.get("origin", [0, 0, 0])),
            method_type=meta.get("method_type", "gravity"),
            method_kwargs=meta.get("method_kwargs", {}),
            regularization_type=meta.get("regularization_type", "sparse"),
            max_iter=meta.get("max_iter", 30),
            beta0_ratio=meta.get("beta0_ratio", 1.0),
            cooling_factor=meta.get("cooling_factor", 2.0),
            alpha_s=meta.get("alpha_s", 1e-4),
            alpha_x=meta.get("alpha_x", 1.0),
            alpha_y=meta.get("alpha_y", 1.0),
            alpha_z=meta.get("alpha_z", 1.0),
            norms=tuple(meta.get("norms", [0.0, 2.0, 2.0, 1.0])),
            irls_cooling_factor=meta.get("irls_cooling_factor", 1.1),
            max_irls_iterations=meta.get("max_irls_iterations", 30),
            use_preconditioner=meta.get("use_preconditioner", True),
            l1_ratio=meta.get("l1_ratio", 0.5),
            l1l2_solver=meta.get("l1l2_solver", "cda"),
            l1l2_weighting=meta.get("l1l2_weighting", "S2"),
            lambda_decades=meta.get("lambda_decades", 4.0),
            lambda_step=meta.get("lambda_step", 0.1),
            beta_selection=meta.get("beta_selection", "auto"),
            beta_sweep=meta.get("beta_sweep"),
            beta_sweep_factors=tuple(meta.get("beta_sweep_factors",
                                              InversionTask.beta_sweep_factors)),
            bounds_lower=meta.get("bounds_lower"),
            bounds_upper=meta.get("bounds_upper"),
            noise_pct=meta.get("noise_pct", 0.05),
            noise_floor=meta.get("noise_floor", 0.5),
            joint_methods=meta.get("joint_methods"),
            joint_kwargs_list=meta.get("joint_kwargs_list"),
            joint_weights=meta.get("joint_weights"),
            cross_gradient_weight=meta.get("cross_gradient_weight", 0.0),
        )

        names = zf.namelist()
        if "initial_model.npy" in names:
            task.initial_model = _read_npy(zf, "initial_model.npy")
        if "observed.npy" in names:
            task.observed_data = _read_npy(zf, "observed.npy")
        if "std.npy" in names:
            task.data_std = _read_npy(zf, "std.npy")
        if "locations.npy" in names:
            task.station_locations = _read_npy(zf, "locations.npy")
        if "active_cells.npy" in names:
            task.active_cells = _read_npy(zf, "active_cells.npy")
        if "hx.npy" in names:
            task.hx = _read_npy(zf, "hx.npy")
        if "hy.npy" in names:
            task.hy = _read_npy(zf, "hy.npy")
        if "hz.npy" in names:
            task.hz = _read_npy(zf, "hz.npy")

        # Joint surveys
        joint_surveys = []
        i = 0
        while f"joint_obs_{i}.npy" in zf.namelist():
            joint_surveys.append({
                "observed": _read_npy(zf, f"joint_obs_{i}.npy"),
                "std": _read_npy(zf, f"joint_std_{i}.npy"),
                "locations": _read_npy(zf, f"joint_locs_{i}.npy"),
            })
            i += 1
        if joint_surveys:
            task.joint_surveys = joint_surveys

        joint_models = []
        i = 0
        while f"joint_model_{i}.npy" in zf.namelist():
            joint_models.append(_read_npy(zf, f"joint_model_{i}.npy"))
            i += 1
        if joint_models:
            task.joint_initial_models = joint_models

    return task


def _write_npy(zf: zipfile.ZipFile, name: str, arr: NDArray) -> None:
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr))
    zf.writestr(name, buf.getvalue())


def _read_npy(zf: zipfile.ZipFile, name: str) -> NDArray:
    return np.load(io.BytesIO(zf.read(name)))
