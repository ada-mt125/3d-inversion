"""Custom SimPEG directives: iteration tracking and L1–L2 depth weighting."""

from __future__ import annotations

from typing import Any

import numpy as np
from simpeg.directives import InversionDirective, UpdateIRLS

from ..datamodel.result import IterationSnapshot


class IterationCollector(InversionDirective):
    """Records every iteration's state during a SimPEG inversion.

    Usage:
        collector = IterationCollector()
        directive_list = [collector, ...]
        inv = BaseInversion(inv_prob, directiveList=directive_list)
        inv.run(m0)
        snapshots = collector.snapshots
    """

    def __init__(self) -> None:
        super().__init__()
        self.snapshots: list[IterationSnapshot] = []

    def initialize(self) -> None:
        super().initialize()
        self.snapshots = []

    def endIter(self) -> None:
        try:
            iteration = self.opt.iter if self.opt else len(self.snapshots)
            model_values = np.array(self.invProb.model, copy=True)
            phi_d = float(self.invProb.phi_d)
            phi_m = float(self.invProb.phi_m)
            beta = float(self.invProb.beta)

            snap = IterationSnapshot(
                iteration=iteration,
                model_values=model_values,
                phi_d=phi_d,
                phi_m=phi_m,
                phi_total=phi_d + beta * phi_m,
                beta=beta,
            )
            self.snapshots.append(snap)
        except Exception:
            pass


class ElasticNetSensitivityWeights(InversionDirective):
    """Depth weighting for the L1–L2 regularization (Utsugi 2019).

    Sets the ``sensitivity`` cell weights of every regularization term to
    ``||g_j||``, the norm of sensitivity column j in data units (no data
    weighting, no normalization), so the regularized variable is
    ``x~_j = ||g_j||^(1/2) m_j`` as in the paper.  Computed once, at the
    start of the inversion; place it before the beta estimator.

    Args:
        floor: Weights are clipped below at ``floor * max(||g_j||)``.
    """

    def __init__(self, floor: float = 1e-10, **kwargs) -> None:
        super().__init__(**kwargs)
        self.floor = floor

    def initialize(self) -> None:
        m = self.invProb.model
        gtg = 0.0
        for sim in self.simulation:
            gtg = gtg + sim.getJtJdiag(m)
        column_norms = np.sqrt(gtg)
        column_norms = np.maximum(column_norms, self.floor * column_norms.max())
        for reg in self.reg.objfcts:
            for term in getattr(reg, "objfcts", [reg]):
                term.set_weights(sensitivity=term.mapping * column_norms)


class DampedUpdateIRLS(UpdateIRLS):
    """``UpdateIRLS`` whose beta corrections shrink after each overshoot.

    During IRLS, SimPEG rescales beta by the same factor up or down to steer
    phi_d to the target.  When phi_d is steep in beta this can settle into a
    two-cycle that never meets the target (e.g. phi_d alternating 149 / 320
    for a target of 169).  Here each reversal of the correction's direction
    halves its size in log(beta), so beta brackets the target and converges.
    Used for both the lp-norm (sparse) and L1–L2 paths.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._damping = 1.0
        self._last_direction = 0.0

    def adjust_cooling_schedule(self) -> None:
        super().adjust_cooling_schedule()
        if self.metrics.start_irls_iter is None or self.cooling_factor == 1.0:
            return
        direction = np.sign(np.log(self.cooling_factor))
        if self._last_direction and direction != self._last_direction:
            self._damping *= 0.5
        self._last_direction = direction
        self.cooling_factor = self.cooling_factor ** self._damping
