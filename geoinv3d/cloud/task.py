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

    # Regularization: "sparse" (lp-norm IRLS), "l1l2" (elastic net, Utsugi 2019),
    # "mgs" (minimum gradient support), "tv" (total variation), "l2" (smooth L2
    # with depth weighting), "group_lasso" (joint gravity-magnetic only, below)
    # or anything else for the legacy smooth L2 path
    regularization_type: str = "sparse"

    # Shared inversion parameters
    max_iter: int = 30
    beta0_ratio: float = 1.0
    cooling_factor: float = 2.0
    # None: the default for the regularization (see effective_alpha_s): 1 where
    # alpha_x/y/z are length scales (l2, sparse, mgs, tv, joint), 1e-4 for the
    # legacy smooth path with raw SimPEG alphas.  (1e-4 with length scales
    # makes the smallness term ~(length scale x cell)^2 times weaker than the
    # gradient terms, and sparse models turn into columns down to the mesh
    # bottom; see LOGBOOK 2026-09-27.)
    alpha_s: Optional[float] = None
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
    # regularization_type "bayes": the linear-Gaussian posterior (worker.run_bayesian_inversion)
    bayes_samples: int = 30
    bayes_seed: int = 0
    bayes_threshold: Optional[float] = None
    bayes_beta: str = "discrepancy"     # chi^2 = N, or "evidence" (beta and the errors' scale)
    bayes_prior: str = "compact"        # or "smooth" (the smooth L2 Gaussian)
    lambda_decades: float = 4.0
    lambda_step: float = 0.1

    # MGS / TV focusing parameter e: focusing_scale (None: 1 for MGS, 0.1 for
    # TV) times this percentile of the L2 model's gradient magnitude, fixed
    # once IRLS starts
    focusing_percentile: float = 95.0
    focusing_scale: Optional[float] = None

    # Depth weighting of the model norm for l2 / sparse / mgs / tv:
    # "sensitivity" (SimPEG's rms sensitivities, recomputed by the directive)
    # or "depth" (Li & Oldenburg: norm weight (z + z0)^(-beta/2) with
    # beta = depth_weighting_exponent; ~2 for gravity, ~3 for magnetics;
    # smaller beta keeps the model shallower)
    depth_weighting: str = "sensitivity"
    depth_weighting_exponent: float = 2.0

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

    # Joint inversion (multiple datasets)
    joint_methods: Optional[list[str]] = None
    joint_kwargs_list: Optional[list[dict]] = None
    joint_weights: Optional[list[float]] = None
    # How the models are coupled (geoinv3d/methods/coupling.py): "cross_gradient",
    # "joint_total_variation", "linear_correspondence", "pgi", "group_lasso" or
    # "none".  None: as before couplings were a choice of their own —
    # regularization_type "group_lasso" meant the group lasso, a positive
    # cross_gradient_weight the cross-gradient (see the ``coupling`` property).
    joint_coupling: Optional[str] = None
    # Unit-free weight of the coupling term(s): 1 weighs them like the models'
    # regularization (CouplingScale); None: cross_gradient_weight, the former name
    coupling_weight: Optional[float] = None
    # The coupling's own settings: slope / intercept of the linear correspondence,
    # units / learn of PGI, eps of the joint total variation, scale ("raw": the weight
    # as a plain multiplier)
    coupling_options: Optional[dict] = None
    cross_gradient_weight: float = 0.0
    # Model label of each dataset: datasets with the same label share one model
    # (e.g. gz and gzz data of one density model, MT and DC data of one
    # conductivity model); None or a missing entry: a model of its own
    joint_models: Optional[list[Optional[str]]] = None
    # Per-dataset overrides of the regularization settings above for its model
    # (the first dataset of a model decides): any of JOINT_REG_KEYS, e.g.
    # {"regularization_type": "mgs", "bounds_lower": 0.0}.  The joint
    # inversion (not the group lasso) regularizes each model with
    # regularization_type l2 / sparse / l1l2 (by IRLS) / mgs / tv, or the
    # legacy WeightedLeastSquares for any other type.
    joint_regularizations: Optional[list[dict]] = None
    # Balance the models' regularizations against their data (see
    # JointRegularizationBalance); not used by the legacy path
    joint_balance: bool = True

    # Joint group lasso (regularization_type "group_lasso"; Utsugi 2025, extended
    # to any number of models and datasets, nonlinear methods and a
    # cross-gradient; see geoinv3d/methods/group_lasso.py).  lambda1 (group
    # sparsity) is chosen on a sweep of gl_n_lambda1 values over
    # gl_lambda1_decades decades below lambda1_max, by gl_lambda1_selection
    # "lcurve" (log misfit vs log group penalty, falling back to chi^2 = N) or
    # "discrepancy", or fixed ("fixed": gl_lambda1, else gl_lambda1_ratio x
    # lambda1_max), or "search": no sweep, a secant search for chi^2 = N from
    # gl_lambda1_ratio x lambda1_max (e.g. the ratio of a 2 km run), warm-started from a
    # short descent (lambda1_max / 10, then halfway in log to the start).  gl_lambda2 is plain L2 damping of the unit-column
    # variables (X^T X has a unit diagonal); gl_mu the ADMM penalty (None:
    # mean eigenvalue of X^T X).  On the 16 x 16 x 8 synthetic, lambda2 = 0.3
    # gave the true amplitudes; 0.01 inflated them ~6x, 1 spread the body and
    # kept chi^2 above N for small errors (LOGBOOK 2026-09-29).  gl_data_scaling
    # "auto" is the paper's "max_ratio" for potential fields and "std" when a
    # dataset is nonlinear (MT, DC).  gl_cross_gradient (lambda3, dimensionless:
    # 1 weighs the structural coupling like the data) adds a cross-gradient
    # between every pair of models.  Nonlinear datasets are solved by
    # Levenberg–Marquardt Gauss–Newton: at most gl_gn_max_iter linearizations
    # per lambda1, until the objective changes by less than gl_gn_tol.
    gl_lambda1_selection: str = "lcurve"
    gl_lambda1: Optional[float] = None
    gl_lambda1_ratio: float = 0.02
    gl_lambda2: float = 0.3
    gl_mu: Optional[float] = None
    gl_data_scaling: str = "auto"
    gl_gamma: float = 2.0
    gl_n_lambda1: int = 13
    gl_lambda1_decades: float = 3.0
    gl_max_iter: int = 3000
    gl_tol: float = 1e-4
    gl_cross_gradient: float = 0.0
    # "group" (the group lasso) or "none": the same solver, L2 and lambda1 choice, but each
    # model soft-thresholded on its own (lambda1 (|beta| + |rho|)): a control that separates
    # what the coupling does from what the solver and its sparsity do
    gl_coupling: str = "group"
    gl_gn_max_iter: int = 20
    gl_gn_tol: float = 1e-5
    # How the group lasso weighs its cells: "sensitivity" (the paper: w = ||column||^(-gamma/2),
    # gl_gamma = 2 makes every cell equally cheap, so on a mesh with wide padding the model
    # leaves the core) or "depth": each cell weighed in the penalty by its volume x the Li &
    # Oldenburg depth weight of its model's regularization (depth_weighting_exponent), as the
    # SimPEG regularizations weigh it (see GroupLassoProblem cell_weights)
    gl_weighting: str = "sensitivity"
    # ADMM over-relaxation (1: plain ADMM; 1.6 took 31 % fewer iterations for the same
    # minimizer on the Karnataka 2 km mesh)
    gl_relaxation: float = 1.6
    # Balance the datasets: one lambda1 serves all models, so chi^2 = N in total can hide
    # an overfitted gravity model and an underfitted magnetic one (0.4 and 2.6 x N on the
    # Karnataka 2 km mesh).  True: after the sweep, reweigh each model's data by
    # sqrt(chi^2 / N) and find lambda1 for chi^2 = N again (warm-started), at most
    # gl_balance_rounds times, until each model's chi^2 / N is within 1/1.2 .. 1.2
    gl_balance: bool = False
    gl_balance_rounds: int = 6
    # Data weights of the models to start from (one per model, in model order, or
    # {model: weight}), e.g. those a 2 km run balanced to: the balance then starts near its end
    gl_data_weights: Optional[object] = None

    # Arrays stored separately in the archive
    initial_model: Optional[NDArray] = None
    observed_data: Optional[NDArray] = None
    data_std: Optional[NDArray] = None
    station_locations: Optional[NDArray] = None
    active_cells: Optional[NDArray] = None
    # Per active cell (e.g. from geology constraints, methods/geology.py): the reference
    # model of the smallness term, bounds overriding bounds_lower/upper, and a multiplier
    # of the smallness term's weights
    reference_model: Optional[NDArray] = None
    cell_lower: Optional[NDArray] = None
    cell_upper: Optional[NDArray] = None
    smallness_weights: Optional[NDArray] = None
    # Sharp boundaries (geology units marked "sharp"; like ModEM's covariance "tears"): a
    # label per active cell; the smoothness terms weigh the faces between cells of different
    # labels by smoothness_break_factor
    smoothness_labels: Optional[NDArray] = None
    smoothness_break_factor: float = 0.01
    hx: Optional[NDArray] = None
    hy: Optional[NDArray] = None
    hz: Optional[NDArray] = None

    # For joint: list of (observed, std, locations) per method
    joint_surveys: Optional[list[dict]] = None
    joint_initial_models: Optional[list[NDArray]] = None

    @property
    def coupling(self) -> str:
        """The coupling of a joint task (see :func:`geoinv3d.methods.coupling.resolve`)."""
        from ..methods.coupling import resolve
        weight = self.coupling_weight if self.coupling_weight is not None else self.cross_gradient_weight
        return resolve(self.joint_coupling, self.regularization_type, weight)

    @property
    def effective_coupling_weight(self) -> float:
        if self.coupling_weight is not None:
            return float(self.coupling_weight)
        if self.cross_gradient_weight:
            return float(self.cross_gradient_weight)
        return 1.0 if self.joint_coupling else 0.0

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
            "bayes_samples": self.bayes_samples,
            "bayes_seed": self.bayes_seed,
            "bayes_threshold": self.bayes_threshold,
            "bayes_beta": self.bayes_beta,
            "bayes_prior": self.bayes_prior,
            "lambda_decades": self.lambda_decades,
            "lambda_step": self.lambda_step,
            "beta_selection": self.beta_selection,
            "focusing_percentile": self.focusing_percentile,
            "focusing_scale": self.focusing_scale,
            "depth_weighting": self.depth_weighting,
            "depth_weighting_exponent": self.depth_weighting_exponent,
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
            d["joint_coupling"] = self.coupling
            if self.coupling_weight is not None:
                d["coupling_weight"] = self.coupling_weight
            if self.coupling_options:
                d["coupling_options"] = self.coupling_options
            d["joint_balance"] = self.joint_balance
            if self.joint_models is not None:
                d["joint_models"] = list(self.joint_models)
            if self.joint_regularizations is not None:
                d["joint_regularizations"] = [dict(r or {}) for r in self.joint_regularizations]
        if self.regularization_type == "group_lasso" or self.joint_coupling == "group_lasso":
            d.update({k: getattr(self, k) for k in GROUP_LASSO_KEYS})
        if self.smoothness_labels is not None:
            d["smoothness_break_factor"] = self.smoothness_break_factor
        return d


# Regularization settings a joint dataset may override for its model
# (InversionTask.joint_regularizations)
JOINT_REG_KEYS = (
    "regularization_type", "alpha_s", "alpha_x", "alpha_y", "alpha_z", "norms", "l1_ratio",
    "focusing_percentile", "focusing_scale", "depth_weighting", "depth_weighting_exponent",
    "bounds_lower", "bounds_upper",
)


CELL_ARRAYS = ("reference_model", "cell_lower", "cell_upper", "smallness_weights",
               "smoothness_labels")

GROUP_LASSO_KEYS = (
    "gl_lambda1_selection", "gl_lambda1", "gl_lambda1_ratio", "gl_lambda2", "gl_mu",
    "gl_data_scaling", "gl_gamma", "gl_n_lambda1", "gl_lambda1_decades", "gl_max_iter",
    "gl_tol", "gl_cross_gradient", "gl_gn_max_iter", "gl_gn_tol", "gl_coupling",
    "gl_weighting", "gl_relaxation", "gl_balance", "gl_balance_rounds", "gl_data_weights",
)


LEGACY_SMOOTH_ALPHA_S = 1e-4
LENGTH_SCALE_ALPHA_S = 1.0


def effective_alpha_s(task: InversionTask, length_scales: bool = True) -> float:
    """``task.alpha_s``, or its default for the regularization when None."""
    if task.alpha_s is not None:
        return float(task.alpha_s)
    return LENGTH_SCALE_ALPHA_S if length_scales else LEGACY_SMOOTH_ALPHA_S


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
        for key in CELL_ARRAYS:
            if getattr(task, key) is not None:
                _write_npy(zf, f"{key}.npy", getattr(task, key))
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
            bayes_samples=int(meta.get("bayes_samples", 30)),
            bayes_seed=int(meta.get("bayes_seed", 0)),
            bayes_threshold=meta.get("bayes_threshold"),
            bayes_beta=meta.get("bayes_beta", "discrepancy"),
            bayes_prior=meta.get("bayes_prior", "compact"),
            lambda_decades=meta.get("lambda_decades", 4.0),
            lambda_step=meta.get("lambda_step", 0.1),
            beta_selection=meta.get("beta_selection", "auto"),
            focusing_percentile=meta.get("focusing_percentile", 95.0),
            focusing_scale=meta.get("focusing_scale"),
            depth_weighting=meta.get("depth_weighting", "sensitivity"),
            depth_weighting_exponent=meta.get("depth_weighting_exponent", 2.0),
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
            joint_coupling=meta.get("joint_coupling"),
            coupling_weight=meta.get("coupling_weight"),
            coupling_options=meta.get("coupling_options"),
            joint_models=meta.get("joint_models"),
            joint_regularizations=meta.get("joint_regularizations"),
            joint_balance=meta.get("joint_balance", True),
            **{k: meta[k] for k in GROUP_LASSO_KEYS if k in meta},
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
        for key in CELL_ARRAYS:
            if f"{key}.npy" in names:
                setattr(task, key, _read_npy(zf, f"{key}.npy"))
        task.smoothness_break_factor = float(meta.get("smoothness_break_factor", 0.01))
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
