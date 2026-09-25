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
