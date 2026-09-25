"""Joint inversion orchestrator using SimPEG's ComboObjectiveFunction.

Uses the jif3d approach: each method gets its own sub-model extracted from
a concatenated model vector via projection maps (SimPEG Wires).  Individual
regularizations apply to each sub-model; an optional cross-gradient term
couples them structurally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from ..datamodel.mesh import Mesh3D
from ..datamodel.model import PhysicalModel
from ..datamodel.result import InversionResult, JointInversionResult, IterationSnapshot
from ..datamodel.survey import SurveyData
from .base import MethodBase
from .directives import IterationCollector


@dataclass
class MethodSetup:
    """Configuration for one method in a joint inversion."""
    method: MethodBase
    survey: SurveyData
    mesh: Mesh3D
    initial_model: NDArray
    weight: float = 1.0
    ref_model: Optional[NDArray] = None
    active_cells: Optional[NDArray] = None

    @property
    def n_params(self) -> int:
        """Length of this method's slice of the joint model vector."""
        if self.active_cells is not None:
            return int(np.count_nonzero(self.active_cells))
        return self.mesh.n_cells


def _pair_cross_gradient(mesh, projection, **kwargs):
    """CrossGradient between two slices of a longer joint model vector.

    ``projection`` (2n x n_total) extracts [m_i | m_j] from the full model;
    the value, gradient and Hessian are mapped back with its transpose.
    """
    from simpeg import maps
    from simpeg.regularization import CrossGradient

    class PairCrossGradient(CrossGradient):
        def __init__(self, mesh, projection, **kwargs):
            n = projection.shape[0] // 2
            super().__init__(mesh, wire_map=maps.Wires(("m1", n), ("m2", n)),
                             approx_hessian=True, **kwargs)
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

    return PairCrossGradient(mesh, projection, **kwargs)


class JointInversion:
    """Orchestrate a joint inversion of multiple geophysical methods.

    Each method operates on its own physical property (density,
    susceptibility, etc.).  The combined model vector is
    [model_0 | model_1 | ...], and SimPEG Wires projections extract
    each slice so the correct values reach each simulation.
    """

    def __init__(
        self,
        setups: list[MethodSetup],
        max_iter: int = 30,
        beta0_ratio: float = 1.0,
        cooling_factor: float = 2.0,
        cross_gradient_weight: float = 0.0,
        reg_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Args:
            reg_kwargs: Extra keyword arguments for each method's
                WeightedLeastSquares regularization (e.g. ``alpha_s``,
                ``length_scale_x``).  None keeps SimPEG's defaults.
        """
        self.setups = setups
        self.max_iter = max_iter
        self.beta0_ratio = beta0_ratio
        self.cooling_factor = cooling_factor
        self.cross_gradient_weight = cross_gradient_weight
        self.reg_kwargs = dict(reg_kwargs or {})

    def build(self) -> dict[str, Any]:
        """Build all SimPEG components and return them as a dict."""
        from simpeg import (
            maps,
            optimization,
            inverse_problem,
            inversion,
            directives,
            regularization,
            objective_function,
        )

        n_methods = len(self.setups)

        # Build projection maps: one per method
        wire_args = []
        for i, setup in enumerate(self.setups):
            wire_args.append((f"m{i}", setup.n_params))
        wires = maps.Wires(*wire_args)

        dmis_list = []
        reg_list = []
        simulations = []

        for i, setup in enumerate(self.setups):
            wire_map = getattr(wires, f"m{i}")

            act_kwargs = {}
            if setup.active_cells is not None:
                act_kwargs["active_cells"] = setup.active_cells
            sim = setup.method.make_simulation_mapped(
                setup.mesh, setup.survey, wire_map, **act_kwargs,
            )
            simulations.append(sim)

            dmis = setup.method.make_dmis(setup.survey, sim)
            dmis_list.append(dmis)

            dmesh = setup.mesh.to_discretize()
            reg = regularization.WeightedLeastSquares(
                dmesh, mapping=wire_map, **act_kwargs, **self.reg_kwargs,
            )
            if setup.ref_model is not None:
                reg.reference_model = setup.ref_model
            reg_list.append(reg)

        # Combine data misfits with weights
        combo_dmis = dmis_list[0] * self.setups[0].weight
        for d, s in zip(dmis_list[1:], self.setups[1:]):
            combo_dmis = combo_dmis + d * s.weight

        # Combine regularizations
        combo_reg = reg_list[0]
        for r in reg_list[1:]:
            combo_reg = combo_reg + r

        # Cross-gradient coupling between each pair of methods
        cross_grad_list = []
        if self.cross_gradient_weight > 0 and n_methods >= 2:
            from simpeg.regularization import CrossGradient

            # Cross-gradient compares models cell by cell, so all methods
            # share the first method's mesh and active cells.
            dmesh = self.setups[0].mesh.to_discretize()
            cg_kwargs = {}
            if self.setups[0].active_cells is not None:
                cg_kwargs["active_cells"] = self.setups[0].active_cells

            if n_methods == 2:
                cg = CrossGradient(
                    dmesh, wire_map=wires, approx_hessian=True, **cg_kwargs,
                )
                cross_grad_list.append(cg)
                combo_reg = combo_reg + self.cross_gradient_weight * cg
            else:
                # For 3+ methods, couple each pair (i, j).  SimPEG's
                # CrossGradient expects a model of exactly [m_i | m_j], so
                # project the full joint model onto that pair first.
                import scipy.sparse as sp
                from itertools import combinations
                for i, j in combinations(range(n_methods), 2):
                    projection = sp.vstack([wires.maps[i][1].P, wires.maps[j][1].P]).tocsr()
                    cg = _pair_cross_gradient(dmesh, projection, **cg_kwargs)
                    cross_grad_list.append(cg)
                    combo_reg = combo_reg + self.cross_gradient_weight * cg

        opt = optimization.InexactGaussNewton(
            maxIter=self.max_iter, cg_maxiter=20,
        )

        inv_prob = inverse_problem.BaseInvProblem(combo_dmis, combo_reg, opt)

        collector = IterationCollector()

        n_data_total = sum(s.survey.observed.size for s in self.setups)
        directive_list = [
            collector,
            directives.BetaEstimate_ByEig(beta0_ratio=self.beta0_ratio),
            directives.BetaSchedule(
                coolingFactor=self.cooling_factor, coolingRate=1,
            ),
            directives.TargetMisfit(chifact=1.0),
        ]

        inv = inversion.BaseInversion(inv_prob, directiveList=directive_list)

        return {
            "simulations": simulations,
            "dmis_list": dmis_list,
            "reg_list": reg_list,
            "cross_grad_list": cross_grad_list,
            "combo_dmis": combo_dmis,
            "combo_reg": combo_reg,
            "opt": opt,
            "inv_prob": inv_prob,
            "inv": inv,
            "collector": collector,
            "wires": wires,
        }

    def run(self, starting_model: NDArray | None = None) -> JointInversionResult:
        """Build and run the joint inversion, returning full results."""
        # Concatenate initial models if not supplied
        if starting_model is None:
            starting_model = np.concatenate(
                [s.initial_model for s in self.setups]
            )

        components = self.build()
        inv = components["inv"]
        collector: IterationCollector = components["collector"]

        m_recovered = inv.run(starting_model)

        method_names = [s.method.method_name for s in self.setups]
        result = JointInversionResult(
            methods=method_names,
            weights=[s.weight for s in self.setups],
            converged=True,
        )

        for snap in collector.snapshots:
            result.add_iteration(snap)

        if not result.iterations:
            inv_prob = components["inv_prob"]
            result.add_iteration(IterationSnapshot(
                iteration=0,
                model_values=np.array(m_recovered, copy=True),
                phi_d=float(inv_prob.phi_d),
                phi_m=float(inv_prob.phi_m),
                phi_total=float(inv_prob.phi_d + inv_prob.beta * inv_prob.phi_m),
                beta=float(inv_prob.beta),
            ))

        # Split recovered model by method
        offset = 0
        for i, setup in enumerate(self.setups):
            nc = setup.n_params
            result.recovered_models[method_names[i]] = np.array(
                m_recovered[offset:offset + nc], copy=True
            )
            offset += nc

        return result
