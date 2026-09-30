"""How the models of a joint inversion are coupled: one layer of its own.

A joint inversion minimizes the data misfits, a regularization of each model and a
coupling between the models.  The coupling is a choice separate from each model's
regularization (Colombo & Rovetta 2018; Haber & Holtzman Gazit 2013), and each kind
rests on its own assumption about how the properties relate:

====================== ============= =================================================
coupling               family        assumption / term
====================== ============= =================================================
cross_gradient         structural    boundaries coincide: ∇m_i × ∇m_j = 0
                                     (Gallardo & Meju 2003, 2004; SimPEG CrossGradient)
joint_total_variation  structural    the models change in the same places, sparsely:
                                     Σ sqrt(Σ_i |∇m_i|² + ε)  (Haber & Holtzman Gazit
                                     2013; SimPEG JointTotalVariation); convex
linear_correspondence  petrophysical a linear relation λ1 m1 + λ2 m2 + λ3 = 0 holds
                                     cell by cell (SimPEG LinearCorrespondence); two models
pgi                    petrophysical each cell belongs to one of a few rock units whose
                                     property pairs form a Gaussian mixture
                                     (Astic & Oldenburg 2019; Astic et al. 2021; SimPEG PGI)
group_lasso            sparsity      anomalies share their cells (support), any sign or
                                     ratio: Σ_k ‖(m_1k, …, m_Pk)‖₂ with L2 damping
                                     (Utsugi 2025); its own ADMM solver
none                   —             the models are only inverted together (shared beta)
====================== ============= =================================================

The first three are terms added to the per-model regularizations of
:class:`~geoinv3d.methods.joint.JointInversion` (weighted by ``coupling_weight``, times
beta like them).  PGI replaces them with SimPEG's PGI regularization (a mixture-model
smallness and the smoothness of every model) and its directives.  The group lasso has
its own solver (:mod:`geoinv3d.methods.group_lasso`); adding its optional cross-gradient
(``gl_cross_gradient``) gives a hybrid that is not Utsugi's method, reported as
``group_lasso+cross_gradient``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Coupling:
    key: str
    label: str
    family: str          # structural / petrophysical / sparsity / none
    solver: str          # "gauss_newton" (a term of JointInversion), "pgi" or "admm"
    reference: str
    max_models: Optional[int] = None


COUPLINGS = {c.key: c for c in (
    Coupling("cross_gradient", "cross-gradient", "structural", "gauss_newton",
             "Gallardo & Meju (2003, 2004)"),
    Coupling("joint_total_variation", "joint total variation", "structural", "gauss_newton",
             "Haber & Holtzman Gazit (2013)"),
    Coupling("linear_correspondence", "linear correspondence", "petrophysical", "gauss_newton",
             "SimPEG LinearCorrespondence", max_models=2),
    Coupling("pgi", "petrophysically guided (PGI)", "petrophysical", "pgi",
             "Astic & Oldenburg (2019); Astic et al. (2021)"),
    Coupling("group_lasso", "group lasso (joint sparsity)", "sparsity", "admm", "Utsugi (2025)"),
    Coupling("none", "none (inverted together, uncoupled)", "none", "gauss_newton", ""),
)}
HYBRID_LABELS = {"group_lasso+cross_gradient": "group lasso + cross-gradient (hybrid)",
                 "group_lasso_uncoupled": "none — L1 + L2 by ADMM (control)"}


def coupling_label(key: str) -> str:
    """A display name of a coupling key (hybrids included)."""
    if key in HYBRID_LABELS:
        return HYBRID_LABELS[key]
    return COUPLINGS[key].label if key in COUPLINGS else str(key)


def resolve(coupling: Optional[str], regularization_type: Optional[str] = None,
            cross_gradient_weight: float = 0.0) -> str:
    """The coupling of a joint inversion, also from the settings that stood for it before
    it was a layer of its own: ``regularization_type="group_lasso"`` meant the group
    lasso, a positive ``cross_gradient_weight`` the cross-gradient."""
    if coupling:
        key = str(coupling).lower().replace("-", "_").replace(" ", "_")
        key = {"jtv": "joint_total_variation", "xgrad": "cross_gradient",
               "cross_gradients": "cross_gradient", "petrophysical": "pgi"}.get(key, key)
        if key not in COUPLINGS:
            raise ValueError(f"Unknown coupling '{coupling}' (expected one of {sorted(COUPLINGS)})")
        if regularization_type == "group_lasso" and key != "group_lasso":
            raise ValueError(f"regularization_type='group_lasso' is the group lasso coupling, "
                             f"not '{key}'")
        return key
    if regularization_type == "group_lasso":
        return "group_lasso"
    return "cross_gradient" if cross_gradient_weight and cross_gradient_weight > 0 else "none"


def check(key: str, n_models: int) -> None:
    c = COUPLINGS[key]
    if key != "none" and n_models < 2:
        raise ValueError(f"The {c.label} coupling needs at least two models; there is {n_models}")
    if c.max_models is not None and n_models > c.max_models:
        raise ValueError(f"The {c.label} coupling relates exactly {c.max_models} models; "
                         f"there are {n_models}")


def linear_coefficients(options: dict) -> np.ndarray:
    """λ1, λ2, λ3 of λ1 m1 + λ2 m2 + λ3 = 0: given as ``coefficients``, or as the line
    m1 = ``slope`` m2 + ``intercept`` (model units: density contrast, SI, log conductivity)."""
    if options.get("coefficients") is not None:
        c = np.asarray(options["coefficients"], dtype=float)
        if c.shape != (3,) or not np.all(np.isfinite(c)) or (c[0] == 0 and c[1] == 0):
            raise ValueError(f"coefficients must be three numbers, λ1 and λ2 not both 0: {c}")
        return c
    if options.get("slope") is None:
        raise ValueError("The linear correspondence needs its relation: 'slope' and 'intercept' "
                         "(model 1 = slope × model 2 + intercept) or 'coefficients'")
    return np.array([1.0, -float(options["slope"]), -float(options.get("intercept", 0.0))])


def _pair_term(cls, mesh, projection, **kwargs):
    """A two-model SimPEG similarity measure (CrossGradient, LinearCorrespondence, …) between
    two slices of a longer joint model vector: ``projection`` (2n x n_total) extracts
    [m_i | m_j]; value, gradient and Hessian map back with its transpose."""
    from simpeg import maps

    class PairTerm(cls):
        def __init__(self, mesh, projection, **kwargs):
            n = projection.shape[0] // 2
            super().__init__(mesh, wire_map=maps.Wires(("m1", n), ("m2", n)), **kwargs)
            self._projection = projection

        @property
        def nP(self):
            return self._projection.shape[1]

        def __call__(self, model):
            return super().__call__(self._projection @ model)

        def deriv(self, model):
            return self._projection.T @ super().deriv(self._projection @ model)

        def deriv2(self, model, v=None):
            P = self._projection
            if v is None:
                return P.T @ super().deriv2(P @ model) @ P
            return P.T @ super().deriv2(P @ model, P @ v)

    PairTerm.__name__ = f"Pair{cls.__name__}"
    return PairTerm(mesh, projection, **kwargs)


def _scaled_joint_total_variation(dmesh, wires, eps: float, eps_relative: float = 1e-2, **kwargs):
    """SimPEG's JointTotalVariation of the models scaled to comparable gradients.

    JTV adds the models' squared gradients, so the model with the larger gradients in its
    own units (density in g/cc against susceptibility in SI) would decide alone where the
    structure is; Haber & Holtzman Gazit (2013) scale the models first.  Here model i is
    multiplied by s_i = 1 / rms(∇m_i), set by :meth:`rescale` at a model (by
    :class:`~geoinv3d.methods.directives.CouplingScale`; all 1 until then)."""
    import scipy.sparse as sp
    from simpeg.regularization import JointTotalVariation

    class ScaledJointTotalVariation(JointTotalVariation):
        def __init__(self, dmesh, wires, eps_relative=1e-2, **kwargs):
            super().__init__(dmesh, wire_map=wires, **kwargs)
            self.model_scales = np.ones(len(wires.maps))
            self.eps_relative = float(eps_relative)
            self.scale_by = "value"   # CouplingScale: weigh it by its value, not its curvature

        def _d(self):
            return np.concatenate([np.full(w.shape[0], s) for (_, w), s
                                   in zip(self.wire_map.maps, self.model_scales)])

        def rescale(self, model):
            """s_i = 1 / rms of the gradient of model i (kept at 1 where it is 0), and ε at
            ``eps_relative`` of the scaled models' mean squared gradient.

            A total variation's curvature is 1/√ε where the models are flat: with a tiny
            absolute ε the flat cells of a sparse (L1) model made JTV's largest curvature
            enormous, so the weight matched to it was ~0 and JTV did nothing (the L1–L2
            coupling comparison).  ε relative to the gradients rounds the corner there."""
            G = self._G
            rms = np.array([np.sqrt(np.mean((G @ (w * model)) ** 2)) for _, w in self.wire_map.maps])
            self.model_scales = np.where(rms > 0, 1.0 / np.where(rms > 0, rms, 1.0), 1.0)
            d = self._d()
            g2 = sum((G @ (w * (d * model))) ** 2 for _, w in self.wire_map.maps)
            v2 = self.regularization_mesh.vol ** 2
            level = float(np.mean((self.W @ g2) / v2))
            if level > 0:
                self.eps = self.eps_relative * level

        def __call__(self, model):
            return super().__call__(self._d() * model)

        def deriv(self, model):
            d = self._d()
            return d * super().deriv(d * model)

        def deriv2(self, model, v=None):
            d = self._d()
            if v is None:
                D = sp.diags(d)
                return D @ super().deriv2(d * model) @ D
            return d * super().deriv2(d * model, d * v)

    return ScaledJointTotalVariation(dmesh, wires, eps_relative=eps_relative, eps=eps, **kwargs)


def coupling_terms(key: str, dmesh, wires, n_params: list[int], active_cells=None,
                   options: Optional[dict] = None) -> list:
    """The SimPEG objective functions of a Gauss–Newton coupling between the models of the
    joint vector that ``wires`` splits (all on one mesh and set of active cells)."""
    from itertools import combinations

    import scipy.sparse as sp
    from simpeg import regularization

    options = dict(options or {})
    n_models = len(n_params)
    check(key, n_models)
    if key == "none":
        return []
    if len(set(n_params)) > 1:
        raise ValueError(f"The {COUPLINGS[key].label} compares models cell by cell: all models "
                         "need the same mesh and active cells")
    act = {} if active_cells is None else {"active_cells": active_cells}
    if key == "joint_total_variation":      # one term over all models at once
        return [_scaled_joint_total_variation(dmesh, wires, float(options.get("eps", 1e-8)),
                                              float(options.get("eps_relative", 1e-2)), **act)]
    if key == "cross_gradient":
        cls, kwargs = regularization.CrossGradient, {"approx_hessian": True}
    elif key == "linear_correspondence":
        cls, kwargs = regularization.LinearCorrespondence, {"coefficients": linear_coefficients(options)}
    else:
        raise ValueError(f"The {COUPLINGS[key].label} is not a term of the Gauss–Newton joint "
                         "inversion")
    if n_models == 2:
        return [cls(dmesh, wire_map=wires, **kwargs, **act)]
    terms = []
    for i, j in combinations(range(n_models), 2):   # every pair, projected out of the full vector
        projection = sp.vstack([wires.maps[i][1].P, wires.maps[j][1].P]).tocsr()
        terms.append(_pair_term(cls, dmesh, projection, **kwargs, **act))
    return terms


# ── PGI: the petrophysical units ───────────────────────────────────────


@dataclass
class PetroUnit:
    """A rock unit of PGI: the mean and spread of each model's property (model units),
    and its share of the volume."""

    name: str
    means: list
    stds: list
    proportion: float = 0.1


@dataclass
class PGISettings:
    units: list = field(default_factory=list)       # PetroUnit, the background first
    learn: bool = False          # update the units' means from the models (kappa = 0)
    alpha_smooth_ratio: float = 1e-2
    beta0_ratio: float = 1e-2
    chi_small: float = 1.0       # the petrophysical target (MultiTargetMisfits' chiSmall)

    def to_dict(self) -> dict:
        return {"units": [{"name": u.name, "means": list(u.means), "stds": list(u.stds),
                           "proportion": u.proportion} for u in self.units],
                "learn": self.learn, "alpha_smooth_ratio": self.alpha_smooth_ratio,
                "beta0_ratio": self.beta0_ratio, "chi_small": self.chi_small}


def pgi_settings(options: dict, model_names: list[str], references: list[float]) -> PGISettings:
    """PGI settings from ``coupling_options``: ``units`` [{name, means, stds, proportion}]
    with means and stds either lists in model order or dicts keyed by model name (in
    model units).  A background unit at the models' references is added first unless one
    is named "background"; its spread defaults to the other units' median spread."""
    raw = options.get("units") or []
    if not raw:
        raise ValueError("PGI needs rock units: coupling_options['units'] = [{name, means, stds, "
                         "proportion}, …] (the means of each property per unit)")
    n = len(model_names)

    def per_model(v, what, name):
        if isinstance(v, dict):
            missing = [m for m in model_names if m not in v]
            if missing:
                raise ValueError(f"PGI unit '{name}' has no {what} for {missing}")
            return [float(v[m]) for m in model_names]
        v = [float(x) for x in v]
        if len(v) != n:
            raise ValueError(f"PGI unit '{name}': {len(v)} {what} for {n} models {model_names}")
        return v

    units = []
    for k, u in enumerate(raw):
        name = str(u.get("name") or f"unit {k + 1}")
        means = per_model(u["means"], "means", name)
        stds = per_model(u.get("stds") or [np.nan] * n, "stds", name)
        units.append(PetroUnit(name, means, stds, float(u.get("proportion", 0.1))))
    # spreads not given: a tenth of the largest |mean - reference| of that property, else 1e-3
    for i in range(n):
        spread = [abs(u.means[i] - references[i]) for u in units]
        default = 0.1 * max(spread) if max(spread) > 0 else 1e-3
        for u in units:
            if not np.isfinite(u.stds[i]) or u.stds[i] <= 0:
                u.stds[i] = default
    if not any(u.name.lower() == "background" for u in units):
        # a tight background: 2.5 % of the largest unit contrast of each property (SimPEG's
        # joint tutorial ~3 %).  One as wide as the units lets a smeared model count as
        # background, and PGI then never assigns a cell to a unit (the block synthetic:
        # 0 of 48 body cells at 10 %, 47 of 48 at 1.7 %)
        stds = []
        for i in range(n):
            contrast = max(abs(u.means[i] - references[i]) for u in units)
            stds.append(0.025 * contrast if contrast > 0 else float(np.median([u.stds[i] for u in units])))
        share = max(0.5, 1.0 - sum(u.proportion for u in units))
        units.insert(0, PetroUnit("background", list(references), stds, share))
    else:   # the background first
        units.sort(key=lambda u: u.name.lower() != "background")
    total = sum(u.proportion for u in units)
    if not total > 0:
        raise ValueError("PGI unit proportions must be positive")
    for u in units:
        u.proportion /= total
    return PGISettings(units=units, learn=bool(options.get("learn", False)),
                       alpha_smooth_ratio=float(options.get("alpha_smooth_ratio", 1e-2)),
                       beta0_ratio=float(options.get("beta0_ratio", 1e-2)),
                       chi_small=float(options.get("chi_small", 1.0)))


def gaussian_mixture(settings: PGISettings, dmesh, active_cells, n_active: int):
    """SimPEG's WeightedGaussianMixture of the units (diagonal covariances)."""
    from simpeg import utils

    # SimPEG builds it on scikit-learn's GaussianMixture and, without scikit-learn,
    # quietly defines a stand-in that has no fit()
    try:
        import sklearn  # noqa: F401
    except ImportError:
        raise ImportError("The PGI coupling needs scikit-learn (SimPEG's Gaussian mixture is "
                          "built on it): pip install scikit-learn") from None

    n_models = len(settings.units[0].means)
    gmm = utils.WeightedGaussianMixture(n_components=len(settings.units), mesh=dmesh,
                                        actv=active_cells, covariance_type="diag")
    # fitted to noise only to create its arrays (as SimPEG's tutorials do; one row per
    # active cell, whose volumes weigh the rows), then set
    gmm.fit(np.random.default_rng(0).standard_normal((n_active, n_models)))
    gmm.means_ = np.array([u.means for u in settings.units], dtype=float)
    gmm.covariances_ = np.array([np.square(u.stds) for u in settings.units], dtype=float)
    gmm.compute_clusters_precisions()
    gmm.weights_ = np.array([u.proportion for u in settings.units], dtype=float)
    return gmm
