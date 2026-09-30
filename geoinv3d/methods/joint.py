"""Joint inversion orchestrator using SimPEG's ComboObjectiveFunction.

Uses the jif3d approach: each model is a slice of a concatenated model
vector, extracted by projection maps (SimPEG Wires).  Each model has its
own regularization; an optional cross-gradient term couples them
structurally.

Datasets and models
-------------------
Every :class:`MethodSetup` is one dataset.  By default each dataset has its
own model (labelled by its method name: "gravity", "magnetics", a second
gravity dataset "gravity_2", ...).  Datasets with the same ``model`` label
share one model: gz and gzz data of one density model, or MT and DC data of
one log-conductivity model.

Regularization
--------------
Each model's regularization is a :class:`ModelRegularization` (from its
first setup, else the inversion's default):

* ``"smooth"``: SimPEG's WeightedLeastSquares with raw alphas and no depth
  weighting (the original joint path);
* ``"l2"``: smooth L2 with depth weighting;
* ``"sparse"``: lp norms by IRLS;
* ``"l1l2"``: the L1–L2 elastic net of Utsugi (2019) by IRLS (its own
  ``||g_j||`` depth weighting; the coordinate-descent path is single-method);
* ``"mgs"`` / ``"tv"``: minimum gradient support / total variation focusing.

Kinds may differ between models, and each model may have bounds.  Without
any ModelRegularization the inversion runs exactly as before this option
existed (WeightedLeastSquares with ``reg_kwargs``, beta cooled by
BetaSchedule to chi^2 = N).  With them, it runs like a single-method
inversion of :func:`geoinv3d.cloud.worker.run_single_inversion`: depth
weighting, an L2 stage to 3 chi^2 = N, then IRLS for the models that need
it, with beta steered onto chi^2 = N by :class:`DampedUpdateIRLS` and
bounds by ProjectedGNCG.

Two things a single beta needs in a joint inversion (see
:mod:`geoinv3d.methods.directives`): sensitivity weights normalized model by
model (:class:`JointSensitivityWeights`; SimPEG's directive normalizes all
models together, which leaves the less sensitive one almost unregularized),
and regularizations balanced against each model's data
(:class:`JointRegularizationBalance`; otherwise the models' penalties add up
in their own units, g/cc^2 against SI^2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from ..datamodel.mesh import Mesh3D
from ..datamodel.result import JointInversionResult, IterationSnapshot
from ..datamodel.survey import SurveyData
from .base import MethodBase
from .directives import IterationCollector

JOINT_REGULARIZATIONS = ("smooth", "l2", "sparse", "l1l2", "mgs", "tv")
DEPTH_WEIGHTINGS = ("sensitivity", "depth", "none")


@dataclass(frozen=True)
class ModelRegularization:
    """Regularization of one model of a joint inversion.

    Args:
        kind: one of :data:`JOINT_REGULARIZATIONS` (see the module docstring).
        alpha_s: smallness weight.
        length_scale_x/y/z: smoothness length scales (for "smooth": the raw
            SimPEG ``length_scale`` values as well).
        norms: lp norms of the smallness and the x/y/z gradients ("sparse").
        l1_ratio: elastic-net mixing a in [0, 1] ("l1l2").
        focusing_percentile, focusing_scale: the focusing parameter of
            "mgs" / "tv" (see :class:`~geoinv3d.methods.regularization.Focusing`).
        depth_weighting: "sensitivity" (rms sensitivities of this model's
            data), "depth" (Li & Oldenburg, below this model's stations) or
            "none".  "l1l2" always uses its ``||g_j||`` weighting and "smooth"
            none.
        depth_weighting_exponent: beta of (z + z0)^(-beta/2) for "depth".
        lower, upper: bounds of the model values (None: unbounded).
    """

    kind: str = "l2"
    alpha_s: float = 1.0
    length_scale_x: float = 1.0
    length_scale_y: float = 1.0
    length_scale_z: float = 1.0
    norms: tuple[float, ...] = (0.0, 2.0, 2.0, 1.0)
    l1_ratio: float = 0.5
    focusing_percentile: float = 95.0
    focusing_scale: Optional[float] = None
    depth_weighting: str = "sensitivity"
    depth_weighting_exponent: float = 2.0
    lower: Optional[float] = None
    upper: Optional[float] = None

    def __post_init__(self) -> None:
        if self.kind not in JOINT_REGULARIZATIONS:
            raise ValueError(f"Unknown regularization kind '{self.kind}' "
                             f"(expected one of {JOINT_REGULARIZATIONS})")
        if self.depth_weighting not in DEPTH_WEIGHTINGS:
            raise ValueError(f"Unknown depth_weighting '{self.depth_weighting}' "
                             f"(expected one of {DEPTH_WEIGHTINGS})")
        object.__setattr__(self, "norms", tuple(float(p) for p in self.norms))
        if (self.lower is not None and self.upper is not None
                and not self.lower < self.upper):
            raise ValueError(f"lower ({self.lower}) must be below upper ({self.upper})")

    @property
    def label(self) -> str:
        """The label the worker uses for this kind (e.g. "sparse_IRLS")."""
        return REGULARIZATION_LABELS[self.kind]

    @property
    def bounded(self) -> bool:
        return self.lower is not None or self.upper is not None

    @property
    def sensitivity_mode(self) -> Optional[str]:
        """How :class:`JointSensitivityWeights` weights this model (None: not at all)."""
        if self.kind == "l1l2":
            return "column_norm"
        if self.kind == "smooth" or self.depth_weighting != "sensitivity":
            return None
        return "rms"

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["norms"] = list(self.norms)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelRegularization":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def build(self, dmesh, mapping, reference_model, active_cells=None):
        """The SimPEG regularization of this model.

        ``reference_model`` is in the space of the mapping's input (the whole
        joint model vector, like the model itself).
        """
        from simpeg import regularization
        from .regularization import ElasticNet, Focusing

        act = {} if active_cells is None else {"active_cells": active_cells}
        scales = dict(length_scale_x=self.length_scale_x, length_scale_y=self.length_scale_y,
                      length_scale_z=self.length_scale_z)
        if self.kind in ("smooth", "l2"):   # they differ in the depth weighting only
            return regularization.WeightedLeastSquares(
                dmesh, mapping=mapping, alpha_s=self.alpha_s, reference_model=reference_model,
                **scales, **act)
        if self.kind == "l1l2":
            return ElasticNet(dmesh, l1_ratio=self.l1_ratio, mapping=mapping,
                              reference_model=reference_model, **act)
        if self.kind == "sparse":
            return regularization.Sparse(
                dmesh, mapping=mapping, alpha_s=self.alpha_s, norms=list(self.norms),
                reference_model=reference_model, **scales, **act)
        return Focusing(dmesh, stabilizer=self.kind, mapping=mapping, alpha_s=self.alpha_s,
                        threshold_percentile=self.focusing_percentile,
                        threshold_scale=self.focusing_scale, reference_model=reference_model,
                        **scales, **act)


REGULARIZATION_LABELS = {"smooth": "smooth_L2", "l2": "smooth_L2", "sparse": "sparse_IRLS",
                         "l1l2": "elastic_net_IRLS", "mgs": "focusing_MGS",
                         "tv": "total_variation"}


@dataclass
class MethodSetup:
    """Configuration for one dataset (method + survey) in a joint inversion.

    Args:
        weight: multiplier of this dataset's data misfit.
        ref_model: reference model of this dataset's model (default: the
            method's ``default_model_value``, e.g. 0 for density or
            log(sigma_background) for MT).
        model: label of the model this dataset constrains; datasets with the
            same label share one model (default: a model of its own).
        regularization: the model's regularization (the first setup of a
            model decides; None: the inversion's default).
    """
    method: MethodBase
    survey: SurveyData
    mesh: Mesh3D
    initial_model: NDArray
    weight: float = 1.0
    ref_model: Optional[NDArray] = None
    active_cells: Optional[NDArray] = None
    model: Optional[str] = None
    regularization: Optional[ModelRegularization] = None

    @property
    def n_params(self) -> int:
        """Length of this setup's model slice."""
        if self.active_cells is not None:
            return int(np.count_nonzero(self.active_cells))
        return self.mesh.n_cells


@dataclass
class JointModel:
    """One model of a joint inversion and the datasets (setup indices) it explains."""

    name: str
    setups: list[int]
    n_params: int
    regularization: Optional[ModelRegularization]

    @property
    def kind(self) -> str:
        return "smooth" if self.regularization is None else self.regularization.kind


def assign_models(setups: list[MethodSetup]) -> tuple[list[str], list[str]]:
    """Model label and dataset label of every setup.

    Unlabelled setups get their method name, then "<method>_2", "<method>_3"
    for later datasets of the same method (skipping labels already given).
    Dataset labels are the survey names, else the method names, made unique
    the same way.
    """
    given = {s.model for s in setups if s.model}

    def unique(base, taken, counts):
        counts[base] = counts.get(base, 0) + 1
        k = counts[base]
        label = base if k == 1 else f"{base}_{k}"
        while label in taken:
            counts[base] += 1
            label = f"{base}_{counts[base]}"
        taken.add(label)
        return label

    models, taken, counts = [], set(given), {}
    for s in setups:
        models.append(s.model if s.model else unique(s.method.method_name, taken, counts))
    datasets, taken, counts = [], set(), {}
    for s in setups:
        datasets.append(unique(s.survey.name or s.method.method_name, taken, counts))
    return models, datasets


def _pair_cross_gradient(mesh, projection, **kwargs):
    """CrossGradient between two slices of a longer joint model vector (see
    :func:`geoinv3d.methods.coupling._pair_term`)."""
    from simpeg.regularization import CrossGradient

    from .coupling import _pair_term
    return _pair_term(CrossGradient, mesh, projection, approx_hessian=True, **kwargs)


def depth_weights(dmesh, locations, active_cells, exponent: float) -> NDArray:
    """Li & Oldenburg depth weights as SimPEG cell weights (see worker._depth_weights)."""
    from simpeg.utils import depth_weighting

    beta = float(exponent)
    if not beta >= 0:
        raise ValueError(f"depth_weighting_exponent must be >= 0, got {beta}")
    return depth_weighting(dmesh, np.asarray(locations, dtype=float)[:, :3],
                           active_cells=active_cells, exponent=2.0 * beta,
                           threshold=0.5 * float(dmesh.h_gridded.min()))


class JointInversion:
    """Orchestrate a joint inversion of multiple geophysical datasets.

    Each model is one physical property (density, susceptibility,
    log-conductivity, ...); the combined model vector is
    [model_0 | model_1 | ...], and SimPEG Wires projections extract each
    slice so the right values reach each simulation.

    Args:
        setups: the datasets (see :class:`MethodSetup`).
        coupling: how the models are coupled (see :mod:`geoinv3d.methods.coupling`):
            "cross_gradient", "joint_total_variation", "linear_correspondence",
            "pgi" or "none".  None: "cross_gradient" when ``cross_gradient_weight``
            > 0, else "none" (as before couplings were a choice of their own).
        coupling_weight: weight of the coupling term(s) (times beta, like the
            regularizations); None: ``cross_gradient_weight``.  Not used by PGI.
        coupling_options: the coupling's own settings: ``slope`` / ``intercept``
            (or ``coefficients``) of the linear correspondence, ``eps`` of the
            joint total variation, ``units`` / ``learn`` of PGI.
        cross_gradient_weight: the coupling weight under its former name.
        reg_kwargs: keyword arguments of the WeightedLeastSquares of models
            without a ModelRegularization (e.g. ``alpha_s``,
            ``length_scale_x``).  None keeps SimPEG's defaults.
        regularization: default ModelRegularization of models whose setups
            give none.  None (and none in the setups): the original joint path.
        max_irls_iterations, irls_cooling_factor, use_preconditioner: as for
            single-method inversions (not used by the original path).
        balance: scale each model's regularization by the eigenvalue ratio of
            its data and regularization terms (see
            :class:`~geoinv3d.methods.directives.JointRegularizationBalance`);
            not used by the original path.
    """

    def __init__(
        self,
        setups: list[MethodSetup],
        max_iter: int = 30,
        beta0_ratio: float = 1.0,
        cooling_factor: float = 2.0,
        cross_gradient_weight: float = 0.0,
        reg_kwargs: Optional[dict] = None,
        regularization: Optional[ModelRegularization] = None,
        max_irls_iterations: int = 30,
        irls_cooling_factor: float = 1.1,
        use_preconditioner: bool = True,
        balance: bool = True,
        coupling: Optional[str] = None,
        coupling_weight: Optional[float] = None,
        coupling_options: Optional[dict] = None,
    ) -> None:
        from .coupling import COUPLINGS, resolve

        if not setups:
            raise ValueError("A joint inversion needs at least one dataset")
        self.setups = setups
        self.max_iter = max_iter
        self.beta0_ratio = beta0_ratio
        self.cooling_factor = cooling_factor
        if coupling_weight is not None:
            weight = float(coupling_weight)
        elif cross_gradient_weight:
            weight = float(cross_gradient_weight)
        else:
            weight = 1.0 if coupling is not None else 0.0   # a coupling asked for: weight 1
        self.coupling = resolve(coupling, None, weight)
        if COUPLINGS[self.coupling].solver == "admm":
            raise ValueError("The group lasso has its own solver: "
                             "geoinv3d.methods.group_lasso.GroupLassoProblem")
        self.coupling_weight = 0.0 if self.coupling == "none" else weight
        self.coupling_options = dict(coupling_options or {})
        # the former name, for callers that read it
        self.cross_gradient_weight = self.coupling_weight if self.coupling == "cross_gradient" else 0.0
        self.reg_kwargs = dict(reg_kwargs or {})
        self.regularization = regularization
        self.max_irls_iterations = max_irls_iterations
        self.irls_cooling_factor = irls_cooling_factor
        self.use_preconditioner = use_preconditioner
        self.balance = balance
        self.model_labels, self.dataset_labels = assign_models(setups)
        self.models = self._group_models()

    # -- models ------------------------------------------------------------

    def _group_models(self) -> list[JointModel]:
        models: dict[str, JointModel] = {}
        for i, (setup, label) in enumerate(zip(self.setups, self.model_labels)):
            if label not in models:
                models[label] = JointModel(label, [i], setup.n_params,
                                           setup.regularization or self.regularization)
                continue
            first = self.setups[models[label].setups[0]]
            if setup.n_params != first.n_params or setup.mesh.n_cells != first.mesh.n_cells:
                raise ValueError(f"The datasets of model '{label}' have different meshes or "
                                 "active cells; datasets that share a model share both")
            if (setup.active_cells is None) != (first.active_cells is None) or (
                    setup.active_cells is not None
                    and not np.array_equal(setup.active_cells, first.active_cells)):
                raise ValueError(f"The datasets of model '{label}' have different active cells")
            models[label].setups.append(i)
        return list(models.values())

    @property
    def legacy(self) -> bool:
        """The original joint path: no model has a ModelRegularization."""
        return all(m.regularization is None for m in self.models)

    def _first(self, model: JointModel) -> MethodSetup:
        return self.setups[model.setups[0]]

    def reference_model(self, model: JointModel) -> NDArray:
        """The model's reference (its first setup's, else the method's background)."""
        setup = self._first(model)
        if setup.ref_model is not None:
            ref = np.asarray(setup.ref_model, dtype=float)
            if ref.shape != (model.n_params,):
                raise ValueError(f"The reference model of '{model.name}' has shape {ref.shape}, "
                                 f"expected ({model.n_params},)")
            return ref
        return np.full(model.n_params, float(getattr(setup.method, "default_model_value", 0.0)))

    def joint_reference_model(self) -> NDArray:
        """The reference models of all models as one joint vector."""
        return np.concatenate([self.reference_model(m) for m in self.models])

    def bounds(self) -> tuple[Optional[NDArray], Optional[NDArray]]:
        """Lower and upper bounds of the joint model vector (None: unbounded)."""
        if not any(m.regularization is not None and m.regularization.bounded
                   for m in self.models):
            return None, None
        lo, hi = [], []
        for m in self.models:
            r = m.regularization
            lo.append(np.full(m.n_params, -np.inf if r is None or r.lower is None else r.lower))
            hi.append(np.full(m.n_params, np.inf if r is None or r.upper is None else r.upper))
        return np.concatenate(lo), np.concatenate(hi)

    def starting_model(self) -> NDArray:
        """The first setup's initial model of every model, nudged inside any bounds."""
        m0 = np.concatenate([np.asarray(self._first(m).initial_model, dtype=float)
                             for m in self.models])
        lo, hi = self.bounds()
        if lo is not None:
            # A start exactly on a bound (e.g. m0 = 0 with lower = 0) leaves every
            # cell in ProjectedGNCG's active set and the model never moves
            span = hi - lo
            nudge = np.where(np.isfinite(span), 1e-4 * span, 1e-4)
            m0 = np.clip(m0, lo + nudge, hi - nudge)
        return m0

    # -- SimPEG problem ----------------------------------------------------

    def build(self) -> dict[str, Any]:
        """Build all SimPEG components and return them as a dict."""
        from simpeg import (
            maps, optimization, inverse_problem, inversion, directives, objective_function,
        )
        from .directives import (
            DampedUpdateIRLS, JointRegularizationBalance, JointSensitivityWeights,
            JointUpdatePreconditioner,
        )

        wires = maps.Wires(*[(m.name, m.n_params) for m in self.models])
        wire_of = {m.name: getattr(wires, m.name) for m in self.models}

        dmis_list, simulations = [], []
        for setup, label in zip(self.setups, self.model_labels):
            act_kwargs = {} if setup.active_cells is None else {"active_cells": setup.active_cells}
            sim = setup.method.make_simulation_mapped(
                setup.mesh, setup.survey, wire_of[label], **act_kwargs)
            simulations.append(sim)
            dmis_list.append(setup.method.make_dmis(setup.survey, sim))

        # SimPEG regularizations take the reference in the mapping's input space
        ref_full = self.joint_reference_model()
        reg_list = []
        for m in self.models:
            setup = self._first(m)
            dmesh = setup.mesh.to_discretize()
            act_kwargs = {} if setup.active_cells is None else {"active_cells": setup.active_cells}
            if m.regularization is None:
                from simpeg import regularization
                reg = regularization.WeightedLeastSquares(
                    dmesh, mapping=wire_of[m.name], **act_kwargs, **self.reg_kwargs)
                if setup.ref_model is not None:
                    reg.reference_model = ref_full
            else:
                reg = m.regularization.build(dmesh, wire_of[m.name], ref_full,
                                             setup.active_cells)
                if m.regularization.kind != "smooth" and m.regularization.depth_weighting == "depth":
                    locs = np.vstack([np.asarray(self.setups[i].survey.locations)[:, :3]
                                      for i in m.setups])
                    reg.set_weights(depth=depth_weights(
                        dmesh, locs, setup.active_cells,
                        m.regularization.depth_weighting_exponent))
            reg_list.append(reg)

        combo_dmis = objective_function.ComboObjectiveFunction(
            objfcts=dmis_list, multipliers=[float(s.weight) for s in self.setups])
        if self.coupling == "pgi":
            return self._build_pgi(wires, wire_of, simulations, dmis_list, combo_dmis)

        # The coupling term(s) (cross-gradient between each pair of models, one joint total
        # variation over all, or the linear correspondence of two).  They compare models
        # cell by cell, so all models share the first model's mesh and active cells.
        from .coupling import check, coupling_terms
        n_models = len(self.models)
        coupling_list = []
        if self.coupling != "none" and self.coupling_weight > 0:
            first = self._first(self.models[0])
            coupling_list = coupling_terms(
                self.coupling, first.mesh.to_discretize(), wires,
                [m.n_params for m in self.models], first.active_cells, self.coupling_options)
        elif self.coupling != "none":
            check(self.coupling, n_models)
        combo_reg = objective_function.ComboObjectiveFunction(
            objfcts=reg_list + coupling_list,
            multipliers=[1.0] * len(reg_list) + [self.coupling_weight] * len(coupling_list))

        lo, hi = self.bounds()
        collector = IterationCollector()
        balance = None
        # the coupling weight: unit-free (against the regularization's curvature) unless
        # coupling_options["scale"] == "raw" (the multiplier as given, as before)
        from .directives import CouplingScale
        scale = (CouplingScale(len(reg_list), self.coupling_weight)
                 if coupling_list and self.coupling_options.get("scale", "curvature") != "raw"
                 else None)
        if self.legacy:
            opt = optimization.InexactGaussNewton(maxIter=self.max_iter, cg_maxiter=20)
            directive_list = [
                collector,
                *([scale] if scale else []),
                directives.BetaEstimate_ByEig(beta0_ratio=self.beta0_ratio),
                directives.BetaSchedule(coolingFactor=self.cooling_factor, coolingRate=1),
                directives.TargetMisfit(chifact=1.0),
            ]
        else:
            if lo is not None:
                # the CG tolerances of the single-method path (see worker._single_problem)
                opt = optimization.ProjectedGNCG(
                    maxIter=self.max_iter, lower=lo, upper=hi,
                    maxIterLS=20, cg_maxiter=30, cg_atol=1e-4, cg_rtol=0.0)
            else:
                opt = optimization.InexactGaussNewton(
                    maxIter=self.max_iter, maxIterLS=20, cg_maxiter=30, cg_rtol=1e-4)
            model_index = {m.name: k for k, m in enumerate(self.models)}
            model_of_dmis = [model_index[lbl] for lbl in self.model_labels]
            directive_list = []
            # clipped at 1e-12 of the maximum for potential fields, 1e-2 for MT / DC
            targets = [(reg, m.regularization.sensitivity_mode,
                        1e-12 if all(getattr(self.setups[i].method, "linear", True)
                                     for i in m.setups) else 1e-2)
                       for reg, m in zip(reg_list, self.models)
                       if m.regularization is not None and m.regularization.sensitivity_mode]
            if targets:
                directive_list.append(JointSensitivityWeights(targets))
            if self.balance and n_models > 1:
                balance = JointRegularizationBalance(
                    slices=[wire_of[m.name] for m in self.models],
                    model_of_dmis=model_of_dmis, names=[m.name for m in self.models])
                directive_list.append(balance)
            if scale:
                directive_list.append(scale)
            directive_list += [
                directives.BetaEstimate_ByEig(beta0_ratio=self.beta0_ratio, random_seed=42),
                directives.TargetMisfit(chifact=1.0),
                # Also only steers beta onto chi^2 = N when no model needs IRLS
                DampedUpdateIRLS(
                    f_min_change=1e-4,
                    max_irls_iterations=self.max_irls_iterations,
                    chifact_start=3.0,
                    irls_cooling_factor=self.irls_cooling_factor,
                    cooling_factor=self.cooling_factor,
                ),
                collector,
            ]
            if self.use_preconditioner or lo is not None:
                directive_list.append(JointUpdatePreconditioner())

        inv_prob = inverse_problem.BaseInvProblem(combo_dmis, combo_reg, opt)
        inv = inversion.BaseInversion(inv_prob, directiveList=directive_list)

        return {
            "simulations": simulations,
            "dmis_list": dmis_list,
            "reg_list": reg_list,
            "coupling_list": coupling_list,
            "coupling_scale": scale,
            "cross_grad_list": coupling_list if self.coupling == "cross_gradient" else [],
            "combo_dmis": combo_dmis,
            "combo_reg": combo_reg,
            "opt": opt,
            "inv_prob": inv_prob,
            "inv": inv,
            "collector": collector,
            "wires": wires,
            "balance": balance,
            "bounds": (lo, hi),
        }

    def _build_pgi(self, wires, wire_of, simulations, dmis_list, combo_dmis) -> dict[str, Any]:
        """SimPEG's petrophysically guided inversion of all models together: a Gaussian-mixture
        smallness over the rock units' property pairs plus each model's smoothness (the PGI
        regularization) with the directives of SimPEG's joint PGI tutorials.  The per-model
        regularizations and the coupling weight are not used: PGI is both."""
        from simpeg import (directives, inverse_problem, inversion, maps, optimization,
                            regularization, utils)

        from .coupling import check, gaussian_mixture, pgi_settings

        check("pgi", len(self.models))
        if len({m.n_params for m in self.models}) > 1:
            raise ValueError("PGI classifies every cell by all its properties: all models need "
                             "the same mesh and active cells")
        first = self._first(self.models[0])
        dmesh = first.mesh.to_discretize()
        active = first.active_cells if first.active_cells is not None \
            else np.ones(dmesh.n_cells, dtype=bool)
        n = self.models[0].n_params
        refs = [float(np.median(self.reference_model(m))) for m in self.models]
        settings = pgi_settings(self.coupling_options, [m.name for m in self.models], refs)
        gmm = gaussian_mixture(settings, dmesh, active, n)
        m0 = self.starting_model()
        cells = np.column_stack([wire_of[m.name] * m0 for m in self.models])   # cell x property
        reg = regularization.PGI(
            mesh=dmesh, gmmref=gmm, wiresmap=wires,
            maplist=[maps.IdentityMap(nP=n) for _ in self.models], active_cells=active,
            alpha_pgi=1.0, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
            alpha_xx=0.0, alpha_yy=0.0, alpha_zz=0.0,
            reference_model=utils.mkvc(gmm.means_[gmm.predict(cells)]),
            weights_list=self._pgi_weights(simulations, dmis_list, m0, dmesh, active, wire_of))
        lo, hi = self.bounds()
        opt = optimization.ProjectedGNCG(
            maxIter=self.max_iter, lower=-np.inf if lo is None else lo,
            upper=np.inf if hi is None else hi, maxIterLS=20, cg_maxiter=100, cg_rtol=1e-3)
        n_units = len(settings.units)
        # learn: the units' means move with the models (kappa 0), the background's stays
        kappa = np.vstack([np.full(len(self.models), 1e10)] +
                          [np.zeros(len(self.models))] * (n_units - 1)) if settings.learn else 1e10
        several = len(dmis_list) > 1
        collector = IterationCollector()
        directive_list = [directives.AlphasSmoothEstimate_ByEig(
            alpha0_ratio=settings.alpha_smooth_ratio, random_seed=42)]
        if several:
            directive_list.append(directives.ScalingMultipleDataMisfits_ByEig(
                chi0_ratio=[float(s.weight) for s in self.setups], random_seed=42))
        directive_list += [
            directives.BetaEstimate_ByEig(beta0_ratio=settings.beta0_ratio, random_seed=42),
            directives.PGI_UpdateParameters(update_gmm=settings.learn, kappa=kappa),
            directives.MultiTargetMisfits(chiSmall=settings.chi_small, verbose=False),
        ]
        if several:
            directive_list.append(directives.JointScalingSchedule())
        directive_list += [
            directives.PGI_BetaAlphaSchedule(coolingFactor=self.cooling_factor, tolerance=0.2,
                                             progress=0.2),
            directives.PGI_AddMrefInSmooth(wait_till_stable=True),
            directives.UpdatePreconditioner(),
            collector,
        ]
        inv_prob = inverse_problem.BaseInvProblem(combo_dmis, reg, opt)
        inv = inversion.BaseInversion(inv_prob, directiveList=directive_list)
        return {"simulations": simulations, "dmis_list": dmis_list, "reg_list": [reg] * len(self.models),
                "coupling_list": [reg], "cross_grad_list": [], "combo_dmis": combo_dmis,
                "combo_reg": reg, "opt": opt, "inv_prob": inv_prob, "inv": inv,
                "collector": collector, "wires": wires, "balance": None, "bounds": (lo, hi),
                "pgi": {"regularization": reg, "settings": settings}}

    def _pgi_weights(self, simulations, dmis_list, m0, dmesh, active, wire_of) -> list:
        """Each model's cell weights: its regularization's Li & Oldenburg depth weights
        when it asks for depth weighting (as the other couplings weigh it; the
        sensitivity weighting spread PGI's models below the core of the Karnataka mesh),
        else its rms sensitivity per unit volume from its own datasets, normalized to a
        maximum of 1 (as JointSensitivityWeights), clipped at 1e-12 (potential fields)
        or 1e-2 (MT / DC) of it."""
        vol = dmesh.cell_volumes[active]
        out = []
        for m in self.models:
            r = m.regularization
            if r is not None and r.depth_weighting == "depth":
                locs = np.vstack([np.asarray(self.setups[i].survey.locations)[:, :3]
                                  for i in m.setups])
                out.append(depth_weights(dmesh, locs, active, r.depth_weighting_exponent))
                continue
            jtj = sum(simulations[i].getJtJdiag(m0, W=dmis_list[i].W) for i in m.setups)
            w = np.sqrt(np.asarray(wire_of[m.name] * jtj, dtype=float) / vol)
            linear = all(getattr(self.setups[i].method, "linear", True) for i in m.setups)
            w = np.maximum(w / w.max(), 1e-12 if linear else 1e-2) if w.max() > 0 else np.ones_like(w)
            out.append(w)
        return out

    def run(self, starting_model: NDArray | None = None) -> JointInversionResult:
        """Build and run the joint inversion, returning full results.

        ``result.recovered_models`` is keyed by model label; ``result.extras``
        holds, per model, its datasets, regularization, balance factor and
        bounds, and per dataset its final phi_d and predicted data.
        """
        if starting_model is None:
            starting_model = self.starting_model()

        components = self.build()
        inv = components["inv"]
        collector: IterationCollector = components["collector"]

        m_recovered = inv.run(starting_model)

        method_names = [s.method.method_name for s in self.setups]
        result = JointInversionResult(
            methods=method_names,
            weights=[s.weight for s in self.setups],
            converged=collector.stopped_at is None,   # not converged: stopped by the user
        )

        for snap in collector.snapshots:
            result.add_iteration(snap)

        inv_prob = components["inv_prob"]
        if not result.iterations:
            result.add_iteration(IterationSnapshot(
                iteration=0,
                model_values=np.array(m_recovered, copy=True),
                phi_d=float(inv_prob.phi_d),
                phi_m=float(inv_prob.phi_m),
                phi_total=float(inv_prob.phi_d + inv_prob.beta * inv_prob.phi_m),
                beta=float(inv_prob.beta),
            ))

        wires = components["wires"]
        balance = components["balance"]
        pgi = components.get("pgi")
        models_info = {}
        for k, (m, reg) in enumerate(zip(self.models, components["reg_list"])):
            result.recovered_models[m.name] = np.array(
                getattr(wires, m.name) * m_recovered, copy=True)
            info = {"datasets": [self.dataset_labels[i] for i in m.setups],
                    "methods": [method_names[i] for i in m.setups],
                    "regularization": "pgi" if pgi else "joint_L2" if m.regularization is None
                    else m.regularization.label}
            if m.regularization is not None and not pgi:
                r = m.regularization
                info["kind"] = r.kind
                if r.bounded:
                    info["bounds"] = [r.lower, r.upper]
                if r.kind in ("l2", "sparse", "mgs", "tv"):
                    info["depth_weighting"] = r.depth_weighting
                if r.kind == "sparse":
                    info["norms"] = list(r.norms)
                elif r.kind == "l1l2":
                    info["l1_ratio"] = r.l1_ratio
                elif r.kind in ("mgs", "tv"):
                    info["focusing_threshold"] = reg.focusing_threshold
            if pgi:
                r = m.regularization
                info["weighting"] = (f"depth (beta = {r.depth_weighting_exponent:g})"
                                     if r is not None and r.depth_weighting == "depth"
                                     else "sensitivity")
            if balance is not None and balance.multipliers is not None:
                info["balance"] = float(balance.multipliers[k])
            models_info[m.name] = info

        datasets_info = {}
        for i, (setup, dmis) in enumerate(zip(self.setups, components["dmis_list"])):
            label = self.dataset_labels[i]
            entry = {"method": method_names[i], "model": self.model_labels[i],
                     "n_data": int(setup.survey.n_data)}
            try:
                pred = np.asarray(dmis.simulation.dpred(m_recovered), dtype=float)
                entry["predicted"] = pred
                r = (pred - np.asarray(setup.survey.observed)) / np.asarray(setup.survey.std)
                entry["chi2"] = float(r @ r)
                result.per_method_phi_d[label] = [entry["chi2"]]
            except Exception as e:   # the result is still useful without it
                print(f"[JointInversion] Could not compute the predicted data of {label}: {e}")
            datasets_info[label] = entry
        result.extras = {"models": models_info, "datasets": datasets_info,
                         "coupling": self.coupling_info()}
        scale = components.get("coupling_scale")
        if scale is not None and scale.scales is not None:
            result.extras["coupling"]["scaling"] = "curvature"
            result.extras["coupling"]["multipliers"] = [float(scale.weight * v) for v in scale.scales]
        if pgi:   # which unit each cell ended in, and the units (their means may have moved)
            reg, settings = pgi["regularization"], pgi["settings"]
            gmm = reg.objfcts[0].gmm
            # SimPEG reorders the mixture's components (by weight; ties in any order), so
            # name each by the unit whose given means are nearest (in its spreads), not by
            # position: by position, units of equal share came back under each other's names
            given = np.array([u.means for u in settings.units], dtype=float)
            spread = np.array([u.stds for u in settings.units], dtype=float)
            names = [settings.units[int(np.argmin(np.sum(((mu - given) / spread) ** 2, axis=1)))].name
                     for mu in np.asarray(gmm.means_, dtype=float)]
            result.extras["pgi"] = {
                "membership": np.asarray(reg.objfcts[0].membership(m_recovered), dtype=int),
                "units": [{"name": names[k], "means": [float(v) for v in gmm.means_[k]],
                           "stds": [float(np.sqrt(v)) for v in np.asarray(gmm.covariances_[k]).ravel()],
                           "proportion": float(np.ravel(gmm.weights_)[k])}
                          for k in range(len(names))],
                "learn": settings.learn}
        return result

    def coupling_info(self) -> dict:
        """The coupling as recorded with the results."""
        from .coupling import COUPLINGS
        c = COUPLINGS[self.coupling]
        info = {"kind": self.coupling, "label": c.label, "family": c.family,
                "reference": c.reference, "models": [m.name for m in self.models]}
        if self.coupling not in ("none", "pgi"):
            info["weight"] = self.coupling_weight
        if self.coupling_options:
            info["options"] = {k: v for k, v in self.coupling_options.items() if k != "units"}
        return info
