"""Inversion nodes: single-method and joint inversion as DAG nodes."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.node import Node, register_node
from ..datamodel.model import PhysicalModel, PhysicalProperty
from ..datamodel.survey import SurveyData
from ..datamodel.result import (
    InversionResult, JointInversionResult, IterationSnapshot,
)
from .regularization_nodes import RegularizationConfig

_METHOD_REGISTRY: dict[str, type] = {}


def _get_method(method_type: str, kwargs: dict):
    """Lazy-import and instantiate a method by name."""
    if not _METHOD_REGISTRY:
        from ..methods.gravity import GravityMethod
        from ..methods.magnetics import MagneticsMethod
        from ..methods.dc_resistivity import DCResistivityMethod
        from ..methods.mt import MTMethod
        _METHOD_REGISTRY.update({
            "gravity": GravityMethod,
            "magnetics": MagneticsMethod,
            "dc_resistivity": DCResistivityMethod,
            "mt": MTMethod,
        })
    return _METHOD_REGISTRY[method_type](**kwargs)


@register_node
class SingleInversionNode(Node[InversionResult]):
    """Run a single-method inversion.

    Inputs: [model_node, survey_node, reg_node]
    Output: InversionResult with full iteration history.

    Each parameter change invalidates downstream, so the full
    history of tuning is preserved in the graph.
    """

    evictable = False

    def __init__(
        self,
        model_node: Node[PhysicalModel],
        survey_node: Node[SurveyData],
        reg_node: Node[RegularizationConfig],
        method_type: str = "gravity",
        method_kwargs: dict | None = None,
        max_iter: int = 30,
        beta0_ratio: float = 1.0,
        cooling_factor: float = 2.0,
        name: str = "Inversion",
    ) -> None:
        super().__init__(name, inputs=[model_node, survey_node, reg_node])
        self.method_type = method_type
        self.method_kwargs = method_kwargs or {}
        self.max_iter = max_iter
        self.beta0_ratio = beta0_ratio
        self.cooling_factor = cooling_factor

    def _compute(self, inputs: list[Any]) -> InversionResult:
        model: PhysicalModel = inputs[0]
        survey: SurveyData = inputs[1]
        reg_config: RegularizationConfig = inputs[2]

        method = _get_method(self.method_type, self.method_kwargs)

        from simpeg import (
            optimization, inverse_problem, inversion,
            directives, regularization,
        )
        from ..methods.directives import IterationCollector

        sim = method.make_simulation_full(model.mesh, survey)
        dmis = method.make_dmis(survey, sim)

        dmesh = model.mesh.to_discretize()
        reg = regularization.WeightedLeastSquares(
            dmesh,
            alpha_s=reg_config.alpha_s,
            alpha_x=reg_config.alpha_x,
            alpha_y=reg_config.alpha_y,
            alpha_z=reg_config.alpha_z,
        )

        opt = optimization.InexactGaussNewton(
            maxIter=self.max_iter, cg_maxiter=20,
        )

        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

        collector = IterationCollector()
        directive_list = [
            collector,
            directives.BetaEstimate_ByEig(beta0_ratio=self.beta0_ratio),
            directives.BetaSchedule(
                coolingFactor=self.cooling_factor, coolingRate=1,
            ),
            directives.TargetMisfit(),
        ]

        inv = inversion.BaseInversion(inv_prob, directiveList=directive_list)
        m_recovered = inv.run(model.values)

        result = InversionResult(method=self.method_type, converged=True)

        # Collect all iteration snapshots
        for snap in collector.snapshots:
            result.add_iteration(snap)

        # Ensure at least the final result is recorded
        if not result.iterations:
            result.add_iteration(IterationSnapshot(
                iteration=0,
                model_values=np.array(m_recovered, copy=True),
                phi_d=float(inv_prob.phi_d),
                phi_m=float(inv_prob.phi_m),
                phi_total=float(inv_prob.phi_d + inv_prob.beta * inv_prob.phi_m),
                beta=float(inv_prob.beta),
            ))

        # Store final model as a PhysicalModel
        result.final_model = PhysicalModel(
            mesh=model.mesh,
            values=m_recovered,
            prop=model.prop,
            name=f"{model.name} (recovered)",
        )

        return result

    def set_max_iter(self, max_iter: int) -> None:
        if max_iter != self.max_iter:
            self.max_iter = max_iter
            self.invalidate()

    def set_beta(self, beta0_ratio: float) -> None:
        if beta0_ratio != self.beta0_ratio:
            self.beta0_ratio = beta0_ratio
            self.invalidate()

    def set_cooling(self, cooling_factor: float) -> None:
        if cooling_factor != self.cooling_factor:
            self.cooling_factor = cooling_factor
            self.invalidate()

    def params(self) -> dict:
        return {
            "method_type": self.method_type,
            "method_kwargs": self.method_kwargs,
            "max_iter": self.max_iter,
            "beta0_ratio": self.beta0_ratio,
            "cooling_factor": self.cooling_factor,
            "name": self.name,
        }

    @classmethod
    def from_params(cls, params: dict, inputs: list[Node]) -> SingleInversionNode:
        return cls(
            model_node=inputs[0],
            survey_node=inputs[1],
            reg_node=inputs[2],
            method_type=params["method_type"],
            method_kwargs=params.get("method_kwargs", {}),
            max_iter=params.get("max_iter", 30),
            beta0_ratio=params.get("beta0_ratio", 1.0),
            cooling_factor=params.get("cooling_factor", 2.0),
            name=params.get("name", "Inversion"),
        )


@register_node
class SparseInversionNode(Node[InversionResult]):
    """Run a sparse IRLS inversion with deep-mesh production settings.

    Uses regularization.Sparse with configurable norms (default 0,2,2,1),
    UpdateSensitivityWeights, and optional UpdatePreconditioner.

    Inputs: [model_node, survey_node, reg_node]
    Output: InversionResult with full iteration history.
    """

    evictable = False

    def __init__(
        self,
        model_node: Node[PhysicalModel],
        survey_node: Node[SurveyData],
        reg_node: Node[RegularizationConfig],
        method_type: str = "gravity",
        method_kwargs: dict | None = None,
        max_iter: int = 30,
        beta0_ratio: float = 1.0,
        norms: tuple[float, ...] = (0.0, 2.0, 2.0, 1.0),
        irls_cooling_factor: float = 1.1,
        max_irls_iterations: int = 30,
        use_preconditioner: bool = True,
        name: str = "SparseInversion",
    ) -> None:
        super().__init__(name, inputs=[model_node, survey_node, reg_node])
        self.method_type = method_type
        self.method_kwargs = method_kwargs or {}
        self.max_iter = max_iter
        self.beta0_ratio = beta0_ratio
        self.norms = norms
        self.irls_cooling_factor = irls_cooling_factor
        self.max_irls_iterations = max_irls_iterations
        self.use_preconditioner = use_preconditioner

    def _compute(self, inputs: list[Any]) -> InversionResult:
        model: PhysicalModel = inputs[0]
        survey: SurveyData = inputs[1]
        reg_config: RegularizationConfig = inputs[2]

        method = _get_method(self.method_type, self.method_kwargs)

        from simpeg import (
            optimization, inverse_problem, inversion,
            directives, regularization,
        )
        from ..methods.directives import DampedUpdateIRLS, IterationCollector

        sim = method.make_simulation_full(model.mesh, survey)
        dmis = method.make_dmis(survey, sim)

        dmesh = model.mesh.to_discretize()
        reg = regularization.Sparse(
            dmesh,
            alpha_s=reg_config.alpha_s,
            length_scale_x=reg_config.alpha_x,
            length_scale_y=reg_config.alpha_y,
            length_scale_z=reg_config.alpha_z,
            norms=list(self.norms),
            reference_model=np.zeros(dmesh.nC),
        )

        opt = optimization.InexactGaussNewton(
            maxIter=self.max_iter, maxIterLS=20, maxIterCG=30, tolCG=1e-4,
        )

        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

        collector = IterationCollector()
        directive_list = [
            directives.UpdateSensitivityWeights(),
            directives.BetaEstimate_ByEig(beta0_ratio=self.beta0_ratio, random_seed=42),
            directives.TargetMisfit(chifact=1.0),
            DampedUpdateIRLS(
                f_min_change=1e-4,
                max_irls_iterations=self.max_irls_iterations,
                chifact_start=3.0,
                irls_cooling_factor=self.irls_cooling_factor,
            ),
            collector,
        ]
        if self.use_preconditioner:
            directive_list.append(directives.UpdatePreconditioner())

        inv = inversion.BaseInversion(inv_prob, directiveList=directive_list)
        m_recovered = inv.run(model.values)

        result = InversionResult(method=self.method_type, converged=True)
        for snap in collector.snapshots:
            result.add_iteration(snap)

        if not result.iterations:
            result.add_iteration(IterationSnapshot(
                iteration=0,
                model_values=np.array(m_recovered, copy=True),
                phi_d=float(inv_prob.phi_d),
                phi_m=float(inv_prob.phi_m),
                phi_total=float(inv_prob.phi_d + inv_prob.beta * inv_prob.phi_m),
                beta=float(inv_prob.beta),
            ))

        result.final_model = PhysicalModel(
            mesh=model.mesh,
            values=m_recovered,
            prop=model.prop,
            name=f"{model.name} (sparse recovered)",
        )

        return result

    def set_norms(self, norms: tuple[float, ...]) -> None:
        if norms != self.norms:
            self.norms = norms
            self.invalidate()

    def set_max_iter(self, max_iter: int) -> None:
        if max_iter != self.max_iter:
            self.max_iter = max_iter
            self.invalidate()

    def params(self) -> dict:
        return {
            "method_type": self.method_type,
            "method_kwargs": self.method_kwargs,
            "max_iter": self.max_iter,
            "beta0_ratio": self.beta0_ratio,
            "norms": list(self.norms),
            "irls_cooling_factor": self.irls_cooling_factor,
            "max_irls_iterations": self.max_irls_iterations,
            "use_preconditioner": self.use_preconditioner,
            "name": self.name,
        }

    @classmethod
    def from_params(cls, params: dict, inputs: list[Node]) -> SparseInversionNode:
        return cls(
            model_node=inputs[0],
            survey_node=inputs[1],
            reg_node=inputs[2],
            method_type=params["method_type"],
            method_kwargs=params.get("method_kwargs", {}),
            max_iter=params.get("max_iter", 30),
            beta0_ratio=params.get("beta0_ratio", 1.0),
            norms=tuple(params.get("norms", [0.0, 2.0, 2.0, 1.0])),
            irls_cooling_factor=params.get("irls_cooling_factor", 1.1),
            max_irls_iterations=params.get("max_irls_iterations", 30),
            use_preconditioner=params.get("use_preconditioner", True),
            name=params.get("name", "SparseInversion"),
        )


@register_node
class JointInversionNode(Node[JointInversionResult]):
    """Joint inversion combining multiple methods.

    Inputs: [model1, survey1, reg1, model2, survey2, reg2, ...]
       (groups of 3 per method)
    Output: JointInversionResult with full iteration history.
    """

    evictable = False

    def __init__(
        self,
        method_types: list[str],
        method_kwargs_list: list[dict] | None = None,
        weights: list[float] | None = None,
        max_iter: int = 30,
        beta0_ratio: float = 1.0,
        cooling_factor: float = 2.0,
        cross_gradient_weight: float = 0.0,
        name: str = "JointInversion",
        *,
        inputs: list[Node] | None = None,
    ) -> None:
        super().__init__(name, inputs=inputs or [])
        self.method_types = method_types
        self.method_kwargs_list = method_kwargs_list or [{} for _ in method_types]
        self.weights = weights or [1.0] * len(method_types)
        self.max_iter = max_iter
        self.beta0_ratio = beta0_ratio
        self.cooling_factor = cooling_factor
        self.cross_gradient_weight = cross_gradient_weight

    def _compute(self, inputs: list[Any]) -> JointInversionResult:
        from ..methods.joint import JointInversion, MethodSetup

        n_methods = len(self.method_types)
        setups = []

        for i in range(n_methods):
            base = i * 3
            model: PhysicalModel = inputs[base]
            survey: SurveyData = inputs[base + 1]
            reg_config: RegularizationConfig = inputs[base + 2]

            method = _get_method(
                self.method_types[i], self.method_kwargs_list[i]
            )

            setups.append(MethodSetup(
                method=method,
                survey=survey,
                mesh=model.mesh,
                initial_model=model.values,
                weight=self.weights[i],
            ))

        joint = JointInversion(
            setups=setups,
            max_iter=self.max_iter,
            beta0_ratio=self.beta0_ratio,
            cooling_factor=self.cooling_factor,
            cross_gradient_weight=self.cross_gradient_weight,
        )

        return joint.run()

    def set_weights(self, weights: list[float]) -> None:
        if weights != self.weights:
            self.weights = weights
            self.invalidate()

    def set_max_iter(self, max_iter: int) -> None:
        if max_iter != self.max_iter:
            self.max_iter = max_iter
            self.invalidate()

    def params(self) -> dict:
        return {
            "method_types": self.method_types,
            "method_kwargs_list": self.method_kwargs_list,
            "weights": self.weights,
            "max_iter": self.max_iter,
            "beta0_ratio": self.beta0_ratio,
            "cooling_factor": self.cooling_factor,
            "cross_gradient_weight": self.cross_gradient_weight,
            "name": self.name,
        }

    @classmethod
    def from_params(cls, params: dict, inputs: list[Node]) -> JointInversionNode:
        return cls(
            method_types=params["method_types"],
            method_kwargs_list=params.get("method_kwargs_list"),
            weights=params.get("weights"),
            max_iter=params.get("max_iter", 30),
            beta0_ratio=params.get("beta0_ratio", 1.0),
            cooling_factor=params.get("cooling_factor", 2.0),
            cross_gradient_weight=params.get("cross_gradient_weight", 0.0),
            name=params.get("name", "JointInversion"),
            inputs=inputs,
        )


REGULARIZATION_TYPES = ("l1l2", "mgs", "tv", "sparse", "l2")


def selection_summary(result: dict, n_data: int) -> dict | None:
    """The lambda path / beta sweep of a worker result in one viewer format.

    Returns {"parameter": "λ" | "β", "criterion", "selected", "n_data", "values",
    "misfit" (chi^2), "penalty" (phi_m), "gcv" (or None), "chosen":
    {criterion: value}, "warnings"} or None when the run had no sweep.
    """
    def floats(v):
        return None if v is None else [float(x) for x in v]

    info = result.get("l1l2")
    if info is not None:  # Utsugi CDA lambda path
        chosen = {"L-curve": info.get("lambda_lcurve"), "Discrepancy": info.get(
            "lambda_discrepancy"), "GCV": info.get("lambda_gcv")}
        return {
            "parameter": "λ", "criterion": info["criterion"],
            "selected": float(info["lambda_opt"]), "n_data": int(n_data),
            "values": floats(info["lambdas"]), "misfit": floats(info["chi2"]),
            "penalty": floats(info["penalty"]), "gcv": floats(info.get("gcv")),
            "chosen": {k: float(v) for k, v in chosen.items() if v is not None},
            "weighting": info.get("weighting"),
            "warnings": list(info.get("warnings", [])),
        }
    sel = result.get("beta_selection")
    if sel is not None:  # IRLS fixed-beta sweep
        chosen = {"L-curve": sel.get("beta_lcurve"), "GCV": sel.get("beta_gcv"),
                  "Discrepancy": sel.get("beta_discrepancy")}
        return {
            "parameter": "β", "criterion": sel["criterion"],
            "selected": float(sel["beta_chosen"]), "n_data": int(n_data),
            "values": floats(sel["beta"]), "misfit": floats(sel["phi_d"]),
            "penalty": floats(sel["phi_m"]),
            "gcv": floats(sel["gcv"]) if all(g is not None for g in sel["gcv"]) else None,
            "chosen": {k: float(v) for k, v in chosen.items() if v is not None},
            "warnings": list(sel.get("warnings", [])),
        }
    return None


@register_node
class RegularizedInversionNode(Node[InversionResult]):
    """Single-method inversion with any of the worker's regularizations.

    Runs :func:`geoinv3d.cloud.worker.run_single_inversion`, so it offers the
    same choices as cloud jobs:

    * ``regularization_type``: "l1l2" (L1–L2 elastic net), "mgs" (minimum
      gradient support focusing), "tv" (total variation), "sparse" (lp-norm
      IRLS, ``norms``) or "l2" (smooth);
    * L1–L2: ``l1_ratio`` (alpha), ``l1l2_solver`` ("cda" = Utsugi 2019
      lambda path, or "irls"), ``l1l2_weighting`` ("S1" or "S2"),
      ``lambda_decades``;
    * MGS/TV: ``focusing_percentile``, ``focusing_scale``;
    * ``beta_selection``: "auto", "discrepancy", "lcurve" or "gcv".

    Inputs: [model_node (start model and mesh), survey_node, reg_node (alphas)]
    Output: InversionResult; ``extras`` holds the regularization label, the
    per-iteration statistics and, when a sweep ran, the selection curve.
    """

    evictable = False

    def __init__(
        self,
        model_node: Node[PhysicalModel],
        survey_node: Node[SurveyData],
        reg_node: Node[RegularizationConfig],
        method_type: str = "magnetics",
        method_kwargs: dict | None = None,
        regularization_type: str = "l1l2",
        l1_ratio: float = 0.8,
        l1l2_solver: str = "cda",
        l1l2_weighting: str = "S1",
        lambda_decades: float = 4.0,
        beta_selection: str = "auto",
        focusing_percentile: float = 95.0,
        focusing_scale: float | None = None,
        norms: tuple[float, ...] = (0.0, 2.0, 2.0, 1.0),
        bounds: tuple[float | None, float | None] = (None, None),
        max_iter: int = 30,
        max_irls_iterations: int = 30,
        beta0_ratio: float = 1.0,
        use_preconditioner: bool = True,
        name: str = "RegularizedInversion",
    ) -> None:
        super().__init__(name, inputs=[model_node, survey_node, reg_node])
        if regularization_type not in REGULARIZATION_TYPES:
            raise ValueError(f"regularization_type must be one of {REGULARIZATION_TYPES}")
        self.method_type = method_type
        self.method_kwargs = method_kwargs or {}
        self.regularization_type = regularization_type
        self.l1_ratio = l1_ratio
        self.l1l2_solver = l1l2_solver
        self.l1l2_weighting = l1l2_weighting
        self.lambda_decades = lambda_decades
        self.beta_selection = beta_selection
        self.focusing_percentile = focusing_percentile
        self.focusing_scale = focusing_scale
        self.norms = tuple(norms)
        self.bounds = tuple(bounds)
        self.max_iter = max_iter
        self.max_irls_iterations = max_irls_iterations
        self.beta0_ratio = beta0_ratio
        self.use_preconditioner = use_preconditioner

    def _task(self, model: PhysicalModel, survey: SurveyData, reg: RegularizationConfig):
        from ..cloud.task import InversionTask

        return InversionTask(
            task_id=self.name,
            method_type=self.method_type, method_kwargs=self.method_kwargs,
            regularization_type=self.regularization_type,
            max_iter=self.max_iter, max_irls_iterations=self.max_irls_iterations,
            beta0_ratio=self.beta0_ratio, use_preconditioner=self.use_preconditioner,
            alpha_s=reg.alpha_s, alpha_x=reg.alpha_x, alpha_y=reg.alpha_y,
            alpha_z=reg.alpha_z, norms=self.norms,
            l1_ratio=self.l1_ratio, l1l2_solver=self.l1l2_solver,
            l1l2_weighting=self.l1l2_weighting, lambda_decades=self.lambda_decades,
            beta_selection=self.beta_selection,
            focusing_percentile=self.focusing_percentile,
            focusing_scale=self.focusing_scale,
            bounds_lower=self.bounds[0], bounds_upper=self.bounds[1],
            station_locations=survey.locations, observed_data=survey.observed,
            data_std=survey.std, initial_model=np.asarray(model.values, dtype=float),
        )

    def _compute(self, inputs: list[Any]) -> InversionResult:
        from ..cloud.worker import run_single_inversion

        model, survey, reg = inputs[0], inputs[1], inputs[2]
        out = run_single_inversion(self._task(model, survey, reg), mesh=model.mesh)
        m = np.asarray(out["recovered_model"], dtype=float)

        result = InversionResult(method=self.method_type, converged=out["converged"])
        stats = [dict(it) for it in out["iterations"]]
        for it in stats:
            result.add_iteration(IterationSnapshot(
                iteration=it["iteration"],
                model_values=np.array([it["model_min"], it["model_mean"], it["model_max"]]),
                phi_d=it["phi_d"], phi_m=it["phi_m"], phi_total=it["phi_total"],
                beta=it["beta"],
            ))
        result.extras = {"regularization": out["regularization"], "iteration_stats": stats}
        selection = selection_summary(out, len(survey.observed))
        if selection is not None:
            result.extras["selection"] = selection
        result.final_model = PhysicalModel(
            mesh=model.mesh, values=m, prop=model.prop,
            name=f"{model.name} ({out['regularization']})",
        )
        return result

    def params(self) -> dict:
        return {
            "method_type": self.method_type,
            "method_kwargs": self.method_kwargs,
            "regularization_type": self.regularization_type,
            "l1_ratio": self.l1_ratio,
            "l1l2_solver": self.l1l2_solver,
            "l1l2_weighting": self.l1l2_weighting,
            "lambda_decades": self.lambda_decades,
            "beta_selection": self.beta_selection,
            "focusing_percentile": self.focusing_percentile,
            "focusing_scale": self.focusing_scale,
            "norms": list(self.norms),
            "bounds": list(self.bounds),
            "max_iter": self.max_iter,
            "max_irls_iterations": self.max_irls_iterations,
            "beta0_ratio": self.beta0_ratio,
            "use_preconditioner": self.use_preconditioner,
            "name": self.name,
        }

    @classmethod
    def from_params(cls, params: dict, inputs: list[Node]) -> RegularizedInversionNode:
        p = dict(params)
        return cls(
            model_node=inputs[0], survey_node=inputs[1], reg_node=inputs[2],
            method_type=p.get("method_type", "magnetics"),
            method_kwargs=p.get("method_kwargs", {}),
            regularization_type=p.get("regularization_type", "l1l2"),
            l1_ratio=p.get("l1_ratio", 0.8),
            l1l2_solver=p.get("l1l2_solver", "cda"),
            l1l2_weighting=p.get("l1l2_weighting", "S1"),
            lambda_decades=p.get("lambda_decades", 4.0),
            beta_selection=p.get("beta_selection", "auto"),
            focusing_percentile=p.get("focusing_percentile", 95.0),
            focusing_scale=p.get("focusing_scale"),
            norms=tuple(p.get("norms", [0.0, 2.0, 2.0, 1.0])),
            bounds=tuple(p.get("bounds", [None, None])),
            max_iter=p.get("max_iter", 30),
            max_irls_iterations=p.get("max_irls_iterations", 30),
            beta0_ratio=p.get("beta0_ratio", 1.0),
            use_preconditioner=p.get("use_preconditioner", True),
            name=p.get("name", "RegularizedInversion"),
        )
