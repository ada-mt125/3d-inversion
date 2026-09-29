"""Custom SimPEG directives: iteration tracking and L1–L2 depth weighting."""

from __future__ import annotations

from typing import Any

import numpy as np
from simpeg.directives import InversionDirective, UpdateIRLS, UpdatePreconditioner

from ..datamodel.result import IterationSnapshot


class IterationCollector(InversionDirective):
    """Records every iteration's state during a SimPEG inversion.

    Usage:
        collector = IterationCollector()
        directive_list = [collector, ...]
        inv = BaseInversion(inv_prob, directiveList=directive_list)
        inv.run(m0)
        snapshots = collector.snapshots

    ``IterationCollector.on_iteration``, if set, is called with each new
    snapshot (the cloud worker uses it to report progress).

    ``IterationCollector.stop_check``, if set, is asked after every iteration
    whether the user wants the inversion stopped with its result kept (see
    :func:`stop_requested`); the inversion then ends after that iteration with
    its current model, and ``stopped_at`` holds the iteration.
    """

    on_iteration = None
    stop_check = None

    def __init__(self) -> None:
        super().__init__()
        self.snapshots: list[IterationSnapshot] = []
        self.stopped_at = None

    def initialize(self) -> None:
        super().initialize()
        self.snapshots = []
        self.stopped_at = None

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
            return
        callback = IterationCollector.on_iteration
        if callback is not None:
            try:
                callback(snap)
            except Exception as e:   # progress reporting must never stop an inversion
                print(f"[IterationCollector] on_iteration failed: {e}")
        if stop_requested() and self.opt is not None:
            print(f"[IterationCollector] Stop requested: ending after iteration {iteration} "
                  "with the current model")
            self.stopped_at = iteration
            self.opt.stopNextIteration = True


def stop_requested() -> bool:
    """Whether the user asked the running job to stop and keep its result."""
    check = IterationCollector.stop_check
    if check is None:
        return False
    try:
        return bool(check())
    except Exception:   # an unreadable flag must not stop an inversion
        return False


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


class JointSensitivityWeights(InversionDirective):
    """Depth (sensitivity) weighting of a joint inversion, model by model.

    SimPEG's ``UpdateSensitivityWeights`` normalizes the weights of all
    models together, so the model whose data are the more sensitive sets the
    maximum and the other model's weights (and its regularization) nearly
    vanish.  Here each model is weighted from its own data, as a
    single-method inversion would be:

    * ``"rms"``: the rms sensitivity per unit cell volume of the model's
      datasets (``getJtJdiag`` with the data weights, as
      ``UpdateSensitivityWeights``), clipped at ``threshold`` of its maximum
      and normalized to a maximum of 1;
    * ``"column_norm"``: ``||g_j||`` without data weights or normalization,
      the L1–L2 depth weighting (see :class:`ElasticNetSensitivityWeights`).

    Computed once, at the start; place it before the beta estimator.

    Args:
        targets: (regularization, mode) or (regularization, mode, threshold)
            of each weighted model; the regularization's mapping selects the
            model's slice.  ``threshold`` (default ``threshold``) clips the rms
            weights at that share of their maximum.
    """

    def __init__(self, targets, threshold: float = 1e-12, **kwargs) -> None:
        super().__init__(**kwargs)
        self.targets = [tuple(t) if len(t) == 3 else (t[0], t[1], threshold) for t in targets]
        self.threshold = threshold

    def initialize(self) -> None:
        m = self.invProb.model
        modes = {mode for _, mode, _ in self.targets}
        weighted = unweighted = None
        sims, dmis = self.simulation, self.dmisfit.objfcts
        if "rms" in modes:
            weighted = sum(sim.getJtJdiag(m, W=d.W) for sim, d in zip(sims, dmis))
        if "column_norm" in modes:
            unweighted = sum(sim.getJtJdiag(m) for sim in sims)
        for reg, mode, threshold in self.targets:
            if mode == "rms":
                vol = reg.regularization_mesh.vol
                w = np.sqrt(np.asarray(reg.mapping * weighted) / vol**2)
                w = np.maximum(w, threshold * w.max())
                w = w / w.max()
            elif mode == "column_norm":
                w = np.sqrt(np.asarray(reg.mapping * unweighted))
                w = np.maximum(w, 1e-10 * w.max())
            else:
                raise ValueError(f"Unknown sensitivity weighting mode '{mode}'")
            for term in getattr(reg, "objfcts", [reg]):
                term.set_weights(sensitivity=w)


def _slice_eigenvalue(apply, n_total, projection, rng, n_iter: int = 4) -> float:
    """Largest eigenvalue of P H P^T by power iteration; ``apply(v)`` is H v."""
    x = rng.normal(size=projection.shape[0])
    x /= np.linalg.norm(x)
    for _ in range(n_iter):
        y = projection @ apply(projection.T @ x)
        norm = np.linalg.norm(y)
        if not norm > 0:
            return 0.0
        x = y / norm
    return float(x @ (projection @ apply(projection.T @ x)))


class JointRegularizationBalance(InversionDirective):
    """Balance each model's regularization against its data, for one beta.

    One beta multiplies the sum of all models' regularizations, each in its
    own units (e.g. (g/cc)^2 and SI^2), so without scaling one model is
    barely regularized.  For every model k this estimates, by power iteration
    as ``BetaEstimate_ByEig`` does for the whole problem, the ratio

        r_k = lambda_max(H_d restricted to model k) / lambda_max(H_m,k)

    (H_d from the data misfits of model k's datasets, H_m,k its
    regularization's Hessian) and sets the model's multiplier in the combined
    regularization to r_k / (geometric mean of r).  The beta estimated after
    it then suits every model at the start: each is regularized as its own
    single-method inversion would be at the same beta ratio.

    That alone does not make the models fit their data equally well: one
    beta steers the total chi^2 onto N, and it can be split very unevenly
    (e.g. chi^2 = 103 and 1 for two datasets of 49).  So after every
    iteration (``adaptive``) the multipliers move regularization from the
    models that fit their data better than the others to those that fit
    worse:

        log mult_k -= rate * (log(chi2_k / N_k) - mean_j log(chi2_j / N_j))

    with each step capped at a factor ``max_step`` and the geometric mean
    of the multipliers kept.  It starts once the total chi^2 is below
    ``chifact_start`` N (with the IRLS stage): earlier, while beta is large and
    the model small, chi2_k / N_k measures each dataset's signal, not how
    evenly the models fit.  The overall level stays with beta; the fixed
    point has chi2_k / N_k equal for all models (so chi2_k = N_k once the
    total is on target).  chi2_k sums the model's datasets (their weights
    are not applied); datasets of one model are balanced by their weights.
    Cross-gradient terms keep their weights.  Place it after the
    sensitivity weights and before ``BetaEstimate_ByEig``.

    Args:
        slices: the model's projection map (Wires entry) of each model, in
            the order of the combined regularization's first terms.
        model_of_dmis: the model index of each data misfit term.
        names: model names (for printing).
    """

    def __init__(self, slices, model_of_dmis, names=None, n_pw_iter: int = 4,
                 random_seed: int = 42, adaptive: bool = True, rate: float = 1.0,
                 max_step: float = 2.0, chifact_start: float = 3.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.slices = list(slices)
        self.model_of_dmis = list(model_of_dmis)
        self.names = list(names) if names is not None else [str(i) for i in range(len(slices))]
        self.n_pw_iter = n_pw_iter
        self.random_seed = random_seed
        self.adaptive = adaptive
        self.rate = rate
        self.max_step = max_step
        self.chifact_start = chifact_start
        self.ratios = None
        self.multipliers = None
        self.misfit_ratios = None   # chi2_k / N_k after the last iteration

    def model_misfits(self) -> np.ndarray:
        """chi^2 / N of every model's datasets at the current model."""
        dmis = self.dmisfit.objfcts
        # the predicted data of the last evaluation (the accepted model), all
        # datasets stacked
        dpred = getattr(self.invProb, "dpred", None)
        sizes = [d.nD for d in dmis]
        if dpred is None or np.size(dpred) != sum(sizes):
            dpred = np.hstack([d.simulation.dpred(self.invProb.model) for d in dmis])
        parts = np.split(np.asarray(dpred), np.cumsum(sizes)[:-1])
        chi2 = np.zeros(len(self.slices))
        n = np.zeros(len(self.slices))
        for d, k, pred in zip(dmis, self.model_of_dmis, parts):
            r = d.W @ (pred - d.data.dobs)
            chi2[k] += float(r @ r)
            n[k] += d.nD
        return chi2 / n

    def endIter(self) -> None:
        if not self.adaptive or self.multipliers is None:
            return
        ratios = self.model_misfits()
        self.misfit_ratios = ratios
        n_total = sum(d.nD for d in self.dmisfit.objfcts)
        if not np.all(ratios > 0) or self.invProb.phi_d > self.chifact_start * n_total:
            return
        log_r = np.log(ratios)
        step = np.clip(-self.rate * (log_r - log_r.mean()), -np.log(self.max_step),
                       np.log(self.max_step))
        mult = self.multipliers * np.exp(step)
        mult /= np.exp(np.mean(np.log(mult)))
        reg = self.reg
        multipliers = list(reg.multipliers)
        for k, value in enumerate(mult):
            multipliers[k] = float(value)
        reg.multipliers = multipliers
        self.multipliers = mult

    def initialize(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        m = self.invProb.model
        dmis = self.dmisfit
        fields = [d.simulation.fields(m) for d in dmis.objfcts]
        reg = self.reg
        ratios = []
        for k, wire in enumerate(self.slices):
            P = wire.P
            terms = [j for j, mk in enumerate(self.model_of_dmis) if mk == k]

            def data_hessian(v, terms=terms):
                out = 0.0
                for j in terms:
                    out = out + dmis.multipliers[j] * dmis.objfcts[j].deriv2(m, v, f=fields[j])
                return out
            eig_d = _slice_eigenvalue(data_hessian, len(m), P, rng, self.n_pw_iter)
            eig_m = _slice_eigenvalue(lambda v, k=k: reg.objfcts[k].deriv2(m, v), len(m), P,
                                      rng, self.n_pw_iter)
            if not (eig_d > 0 and eig_m > 0):
                raise ValueError(f"Cannot balance model '{self.names[k]}': its data or "
                                 "regularization Hessian has no positive eigenvalue")
            ratios.append(eig_d / eig_m)
        ratios = np.asarray(ratios)
        mult = ratios / np.exp(np.mean(np.log(ratios)))
        multipliers = list(reg.multipliers)
        for k, value in enumerate(mult):
            multipliers[k] = float(value)
        reg.multipliers = multipliers
        self.ratios, self.multipliers = ratios, mult
        print("[JointRegularizationBalance] regularization multipliers: " + ", ".join(
            f"{name} {v:.3g}" for name, v in zip(self.names, mult)))


class JointUpdatePreconditioner(UpdatePreconditioner):
    """``UpdatePreconditioner`` that applies the combined regularization's multipliers.

    SimPEG's adds up the Hessian diagonals of the regularization terms
    without their multipliers, which is exact for one regularization but not
    for a joint one whose terms are balanced (:class:`JointRegularizationBalance`)
    or weighted (cross-gradient).
    """

    def _update(self) -> None:
        from simpeg.utils import Zero, sdiag

        m = self.invProb.model
        reg_diag = np.zeros_like(m)
        for mult, reg in zip(self.reg.multipliers, self.reg.objfcts):
            d = reg.deriv2(m)
            if not isinstance(d, Zero):
                reg_diag += mult * d.diagonal()
        jtj = np.zeros_like(m)
        for sim, dmis in zip(self.simulation, self.dmisfit.objfcts):
            jtj += sim.getJtJdiag(m, W=dmis.W)
        diag = jtj + self.invProb.beta * reg_diag
        diag[diag != 0] = 1.0 / diag[diag != 0]
        self.opt.approxHinv = sdiag(diag)

    def initialize(self) -> None:
        self._update()

    def endIter(self) -> None:
        if self.update_every_iteration:
            self._update()


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
