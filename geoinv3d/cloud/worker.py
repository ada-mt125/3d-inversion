"""Cloud worker: runs on AWS Batch to execute an inversion task.

Reads environment variables set by the Batch job:
    TASK_BUCKET     — S3 bucket
    TASK_KEY        — S3 key for input.zip
    TASK_ID         — task identifier
    RESULT_PREFIX   — S3 prefix for uploading results

    PIPELINE_MODE   — "task" (default, pre-packed InversionTask) or
                      "data" (raw data files on S3 + params JSON)
    PIPELINE_PARAMS — S3 key for pipeline parameters JSON (data mode)
    DATA_PREFIX     — S3 prefix where raw data files live (data mode)

Writes to RESULT_PREFIX: progress.json while running (see S3Progress),
then result.zip and result.json.  Exits with status 1 when the inversion
failed (after uploading the error), so AWS Batch marks the job FAILED.

Local mode (the EC2 backend runs this over SSH; no S3 involved):
    python -m geoinv3d.cloud.worker --local params.json DATA_DIR OUT_DIR
writes OUT_DIR/progress.json while running, then OUT_DIR/result.zip and
OUT_DIR/result.json, and exits with status 1 on failure.

Usage (on the cloud instance):
    python -m geoinv3d.cloud.worker
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .task import (
    GROUP_LASSO_KEYS, JOINT_REG_KEYS, LENGTH_SCALE_ALPHA_S, InversionTask, effective_alpha_s,
    unpack_task, pack_task,
)


def _build_mesh(task: InversionTask):
    """Build a discretize mesh from task spec."""
    from ..datamodel.mesh import Mesh3D
    if task.hx is not None:
        mesh = Mesh3D(hx=task.hx, hy=task.hy, hz=task.hz, origin=task.origin)
    else:
        mesh = Mesh3D.uniform(
            task.nx, task.ny, task.nz,
            task.dx, task.dy, task.dz,
            origin=task.origin,
        )
    return mesh


# Names accepted for each method (the upload page sends "magnetic" and "dc").
_METHOD_ALIASES = {
    "gravity": "gravity", "grav": "gravity",
    "magnetics": "magnetics", "magnetic": "magnetics", "mag": "magnetics",
    "dc_resistivity": "dc_resistivity", "dc": "dc_resistivity",
    "mt": "mt",
}


def canonical_method(name: str) -> str:
    """Map a method name or alias to its canonical name (e.g. "magnetic" -> "magnetics")."""
    try:
        return _METHOD_ALIASES[str(name).lower()]
    except KeyError:
        raise ValueError(
            f"Unknown method '{name}'. Expected one of: {sorted(set(_METHOD_ALIASES))}"
        ) from None


def _make_method(method_type: str, method_kwargs: dict | None = None):
    """Instantiate a geophysical method by (alias) name."""
    from ..methods.gravity import GravityMethod
    from ..methods.magnetics import MagneticsMethod
    from ..methods.dc_resistivity import DCResistivityMethod
    from ..methods.mt import MTMethod

    method_cls = {
        "gravity": GravityMethod,
        "magnetics": MagneticsMethod,
        "dc_resistivity": DCResistivityMethod,
        "mt": MTMethod,
    }[canonical_method(method_type)]
    return method_cls(**(method_kwargs or {}))


def _get_method(task: InversionTask):
    """Instantiate the geophysical method."""
    return _make_method(task.method_type, task.method_kwargs)


def _single_problem(task: InversionTask, mesh, sim=None):
    """Fresh data misfit, regularization, optimizer and start model for a task.

    The regularization and optimizer carry IRLS state, so every run needs
    new ones; pass ``sim`` to reuse a simulation (and its sensitivities).

    Returns a namespace with ``kind`` ("l2", "smooth", "sparse", "l1l2", "mgs"
    or "tv"), sim,
    dmis, reg, opt, m0 and the bounds ``lo``/``hi`` (None when unbounded).
    """
    from types import SimpleNamespace

    from ..datamodel.survey import SurveyData
    from ..methods.regularization import ElasticNet, Focusing
    from simpeg import optimization, regularization

    # "l2": smooth L2 with depth (sensitivity) weighting, length-scale alphas and
    # bounds, like the other regularizations.  Any other unknown type falls back
    # to the legacy "smooth" path (raw SimPEG alphas, no depth weighting), kept
    # so pre-packed tasks run unchanged.
    kind = task.regularization_type \
        if task.regularization_type in ("l2", "sparse", "l1l2", "mgs", "tv") else "smooth"
    method = _get_method(task)
    survey = SurveyData(
        locations=task.station_locations,
        observed=task.observed_data,
        std=task.data_std,
        method=canonical_method(task.method_type),
    )
    has_active = task.active_cells is not None
    if sim is None:
        if has_active:
            sim = method.make_simulation_active(mesh, survey, task.active_cells)
        else:
            sim = method.make_simulation_full(mesh, survey)
    dmis = method.make_dmis(survey, sim)
    dmesh = mesh.to_discretize()

    m0 = task.initial_model
    if has_active:
        n_active = int(task.active_cells.sum())
        if len(m0) != n_active:
            m0 = np.full(n_active, float(method.default_model_value))

    # the reference model: the task's (e.g. from geology constraints), else the method's
    # background (0, or log sigma for MT / DC)
    reference = np.full(len(m0), float(method.default_model_value))
    if task.reference_model is not None:
        if len(task.reference_model) != len(m0):
            raise ValueError(f"reference_model has {len(task.reference_model)} values for "
                             f"{len(m0)} active cells")
        reference = np.asarray(task.reference_model, dtype=float)
    if kind == "smooth":
        reg_kwargs = dict(
            alpha_s=effective_alpha_s(task, length_scales=False),
            alpha_x=task.alpha_x,
            alpha_y=task.alpha_y,
            alpha_z=task.alpha_z,
        )
        if task.reference_model is not None:
            reg_kwargs["reference_model"] = reference
    elif kind == "l1l2":
        reg_kwargs = dict(l1_ratio=task.l1_ratio, reference_model=reference)
    else:
        reg_kwargs = dict(
            alpha_s=effective_alpha_s(task),
            length_scale_x=task.alpha_x,
            length_scale_y=task.alpha_y,
            length_scale_z=task.alpha_z,
            reference_model=reference,
        )
        if kind == "sparse":
            reg_kwargs["norms"] = list(task.norms)
        elif kind == "l2":
            pass
        else:
            reg_kwargs.update(stabilizer=kind,
                              threshold_percentile=task.focusing_percentile,
                              threshold_scale=task.focusing_scale)
    if has_active:
        reg_kwargs["active_cells"] = task.active_cells
    depth_weighting = getattr(task, "depth_weighting", "sensitivity")
    if depth_weighting not in ("sensitivity", "depth"):
        raise ValueError(f"Unknown depth_weighting '{depth_weighting}' "
                         "(expected 'sensitivity' or 'depth')")
    reg = {
        "smooth": regularization.WeightedLeastSquares,
        "l2": regularization.WeightedLeastSquares,
        "l1l2": ElasticNet,
        "sparse": regularization.Sparse,
        "mgs": Focusing,
        "tv": Focusing,
    }[kind](dmesh, **reg_kwargs)
    if depth_weighting == "depth" and kind in ("l2", "sparse", "mgs", "tv"):
        reg.set_weights(depth=_depth_weights(task, dmesh))
    if task.smallness_weights is not None:
        # how strongly each cell is pulled to the reference (geology constraints): on the
        # smallness terms only, so the smoothness terms keep their usual weights
        smallness = [o for o in getattr(reg, "objfcts", [reg])
                     if isinstance(o, regularization.Smallness)]
        if not smallness:
            raise ValueError(f"The {kind} regularization has no smallness term to weight")
        for term in smallness:
            term.set_weights(geology=np.asarray(task.smallness_weights, dtype=float))
    if task.smoothness_labels is not None:
        # sharp boundaries: faces between cells of different labels get a small smoothness
        # weight, so the model may jump there (ModEM's covariance "tears")
        labels = np.asarray(task.smoothness_labels)
        if len(labels) != len(m0):
            raise ValueError(f"smoothness_labels has {len(labels)} values for {len(m0)} cells")
        for term in getattr(reg, "objfcts", [reg]):
            if isinstance(term, regularization.SmoothnessFirstOrder):
                term.set_weights(geology=_face_breaks(term, labels, task.smoothness_break_factor))

    lo = hi = None
    bounded = any(v is not None for v in (task.bounds_lower, task.bounds_upper,
                                          task.cell_lower, task.cell_upper))
    if kind == "smooth":
        opt = optimization.InexactGaussNewton(maxIter=task.max_iter, cg_maxiter=20)
    elif bounded:
        # per-cell bounds (geology constraints) where given, else the job's
        lo = np.asarray(task.cell_lower, dtype=float) if task.cell_lower is not None \
            else (task.bounds_lower if task.bounds_lower is not None else -np.inf)
        hi = np.asarray(task.cell_upper, dtype=float) if task.cell_upper is not None \
            else (task.bounds_upper if task.bounds_upper is not None else np.inf)
        # A start exactly on a bound (e.g. m0 = 0 with lower = 0) leaves every
        # cell in ProjectedGNCG's active set and the model never moves, so
        # start just inside the bounds.
        span = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
        nudge = np.where(np.isfinite(span), 1e-4 * span, 1e-4)
        m0 = np.clip(m0, lo + nudge, hi - nudge)
        # The CG tolerances the deprecated tolCG=1e-4 set, which differ by class in SimPEG
        # 0.25: absolute 1e-4 (relative 0) for ProjectedGNCG, relative 1e-4 for
        # InexactGaussNewton
        opt = optimization.ProjectedGNCG(
            maxIter=task.max_iter, lower=lo, upper=hi,
            maxIterLS=20, cg_maxiter=30, cg_atol=1e-4, cg_rtol=0.0,
        )
    else:
        opt = optimization.InexactGaussNewton(
            maxIter=task.max_iter, maxIterLS=20, cg_maxiter=30, cg_rtol=1e-4,
        )
    return SimpleNamespace(kind=kind, sim=sim, dmis=dmis, reg=reg, opt=opt, m0=m0,
                           lo=lo, hi=hi)


def _depth_weights(task: InversionTask, dmesh) -> np.ndarray:
    """Li & Oldenburg depth weights as SimPEG cell weights.

    SimPEG applies the square root of the cell weights in the model norm, so
    these are (z + z0)^(-beta) for a norm weight of (z + z0)^(-beta/2); z is
    the depth below the nearest station, z0 half the smallest cell.
    """
    from simpeg.utils import depth_weighting
    beta = float(task.depth_weighting_exponent)
    if not beta >= 0:
        raise ValueError(f"depth_weighting_exponent must be >= 0, got {beta}")
    return depth_weighting(dmesh, np.asarray(task.station_locations, dtype=float)[:, :3],
                           active_cells=task.active_cells, exponent=2.0 * beta,
                           threshold=0.5 * float(dmesh.h_gridded.min()))


# Sensitivity weights are clipped at this share of their maximum: 1e-12 (SimPEG's
# default) for potential fields, 1e-2 for MT / DC as SimPEG's DC examples do.  With
# 1e-12 the padding cells of a DC inversion were all but unregularized and went to
# log10 sigma of -10 and +5.
SENSITIVITY_THRESHOLD = {True: 1e-12, False: 1e-2}


def _sensitivity_directive(task):
    """UpdateSensitivityWeights with the task's method's threshold."""
    from simpeg import directives
    return directives.UpdateSensitivityWeights(
        threshold_value=SENSITIVITY_THRESHOLD[bool(_get_method(task).linear)])


def _uses_sensitivity_weights(task, kind: str) -> bool:
    return kind == "l1l2" or getattr(task, "depth_weighting", "sensitivity") != "depth"


def _regularization_label(kind: str) -> str:
    return {"smooth": "smooth_L2", "l2": "smooth_L2", "sparse": "sparse_IRLS",
            "l1l2": "elastic_net_IRLS",
            "mgs": "focusing_MGS", "tv": "total_variation"}[kind]


def _finish_result(task, p, collector, m_recovered, inv_prob) -> dict:
    result = _collect_result(task, collector, m_recovered, inv_prob,
                             _regularization_label(p.kind))
    try:
        result["predicted"] = np.asarray(p.dmis.simulation.dpred(m_recovered), dtype=float)
    except Exception as e:   # the result is still useful without it
        print(f"[Worker] Could not compute the predicted data: {e}")
    if p.kind == "l1l2":
        result["l1_ratio"] = task.l1_ratio
        result.pop("norms")
    elif p.kind == "l2":
        result.pop("norms")
    elif p.kind in ("mgs", "tv"):
        result.pop("norms")
        result["focusing_threshold"] = p.reg.focusing_threshold
    if p.kind in ("l2", "sparse", "mgs", "tv"):
        result["depth_weighting"] = getattr(task, "depth_weighting", "sensitivity")
        if result["depth_weighting"] == "depth":
            result["depth_weighting_exponent"] = float(task.depth_weighting_exponent)
    return result


def _run_problem(task, p, beta=None, irls_thresholds=None):
    """Run a prepared problem: discrepancy-principle schedule, or fixed ``beta``.

    ``irls_thresholds`` (fixed-beta IRLS only) fixes eps; see FixedBetaIRLS.
    """
    from ..methods.directives import (
        IterationCollector, ElasticNetSensitivityWeights, DampedUpdateIRLS,
    )
    from ..methods.regparam import FixedBetaIRLS
    from simpeg import inverse_problem, inversion, directives

    inv_prob = inverse_problem.BaseInvProblem(p.dmis, p.reg, p.opt)
    collector = IterationCollector()
    if p.kind == "smooth":
        directive_list = [collector]
        if beta is None:
            directive_list += [
                directives.BetaEstimate_ByEig(beta0_ratio=task.beta0_ratio),
                directives.BetaSchedule(coolingFactor=task.cooling_factor, coolingRate=1),
                directives.TargetMisfit(),
            ]
        if task.use_preconditioner:
            directive_list.insert(1, directives.UpdatePreconditioner())
    elif p.kind == "l2":
        directive_list = ([_sensitivity_directive(task)]
                          if _uses_sensitivity_weights(task, p.kind) else [])
        if beta is None:
            # No IRLS terms here: DampedUpdateIRLS only steers beta, up or down,
            # onto chi^2 = N.  A plain BetaSchedule halves beta each iteration
            # and overshoots (chi^2/N ~ 0.45 on the test synthetic).
            directive_list += [
                directives.BetaEstimate_ByEig(beta0_ratio=task.beta0_ratio, random_seed=42),
                directives.TargetMisfit(chifact=1.0),
                DampedUpdateIRLS(
                    f_min_change=1e-4,
                    max_irls_iterations=task.max_irls_iterations,
                    chifact_start=3.0,
                    cooling_factor=task.cooling_factor,
                ),
            ]
        directive_list.append(collector)
        if task.use_preconditioner or p.lo is not None:
            directive_list.append(directives.UpdatePreconditioner())
    else:
        weights = (ElasticNetSensitivityWeights() if p.kind == "l1l2"
                   else _sensitivity_directive(task))
        # with depth weighting the weights are set on the regularization
        weights = [weights] if _uses_sensitivity_weights(task, p.kind) else []
        if beta is None:
            directive_list = [
                *weights,
                directives.BetaEstimate_ByEig(beta0_ratio=task.beta0_ratio, random_seed=42),
                directives.TargetMisfit(chifact=1.0),
                DampedUpdateIRLS(
                    f_min_change=1e-4,
                    max_irls_iterations=task.max_irls_iterations,
                    chifact_start=3.0,
                    irls_cooling_factor=task.irls_cooling_factor,
                    cooling_factor=task.cooling_factor,
                ),
                collector,
            ]
        else:
            directive_list = [
                *weights,
                FixedBetaIRLS(
                    irls_thresholds=irls_thresholds,
                    f_min_change=1e-4,
                    max_irls_iterations=task.max_irls_iterations,
                    irls_cooling_factor=task.irls_cooling_factor,
                ),
                collector,
            ]
        if task.use_preconditioner or p.lo is not None:
            directive_list.append(directives.UpdatePreconditioner())
    if beta is not None:
        inv_prob.beta = float(beta)

    inv = inversion.BaseInversion(inv_prob, directiveList=directive_list)
    m_recovered = inv.run(p.m0)
    return _finish_result(task, p, collector, m_recovered, inv_prob), inv_prob


def run_smooth_inversion(task: InversionTask, mesh=None) -> dict:
    """L2 smooth inversion (backward-compatible with existing tasks)."""
    if mesh is None:
        mesh = _build_mesh(task)
    return _run_problem(task, _single_problem(task, mesh))[0]


def run_sparse_inversion(task: InversionTask, mesh=None) -> dict:
    """Sparse IRLS inversion with deep-mesh production settings.

    ``regularization_type == "l1l2"`` swaps the lp-norm ``Sparse``
    regularization for the L1–L2 elastic net of Utsugi (2019), with its
    ``||g_j||^(1/2)`` depth weighting; norms and alpha_x/y/z do not apply.
    """
    if mesh is None:
        mesh = _build_mesh(task)
    return _run_problem(task, _single_problem(task, mesh))[0]


def run_fixed_beta(task: InversionTask, beta: float, mesh=None, sim=None,
                   irls_thresholds=None, focusing_threshold=None):
    """Run the task's inversion at a constant ``beta`` (IRLS for sparse/L1–L2).

    Returns ``(result, problem, inv_prob)``; pass ``sim`` to reuse sensitivities
    and ``irls_thresholds`` to fix eps (see FixedBetaIRLS); for MGS/TV,
    ``focusing_threshold`` fixes e.
    """
    from ..methods.regparam import sparse_terms

    if mesh is None:
        mesh = _build_mesh(task)
    p = _single_problem(task, mesh, sim)
    if focusing_threshold is not None and p.kind in ("mgs", "tv"):
        p.reg.focusing_threshold = float(focusing_threshold)
    result, inv_prob = _run_problem(task, p, beta=beta, irls_thresholds=irls_thresholds)
    result["irls_thresholds"] = [float(o.irls_threshold) for o in sparse_terms(p.reg)] \
        if p.kind not in ("smooth", "l2") else None
    return result, p, inv_prob


def _selection_point(p, inv_prob, m) -> dict:
    """phi_d, phi_m and the GCV ingredients of a fixed-beta solution."""
    from ..methods.regparam import gcv_score, influence_trace

    phi_d = float(p.dmis(m))
    reg = p.reg
    phi_m = float(reg.elastic_net_value(m) if hasattr(reg, "elastic_net_value") else reg(m))
    point = {"beta": float(inv_prob.beta), "phi_d": phi_d, "phi_m": phi_m}
    G = getattr(p.sim, "G", None)
    if G is not None:
        J = np.asarray(G, dtype=float) * (1.0 / p.dmis.data.standard_deviation)[:, None]
        free = None
        if p.lo is not None:   # scalar or per-cell bounds
            span = np.asarray(p.hi, dtype=float) - np.asarray(p.lo, dtype=float)
            tol = 1e-8 * np.where(np.isfinite(span), span, 1.0)
            free = (m > p.lo + tol) & (m < p.hi - tol)
        trace = influence_trace(J, reg.deriv2(m) / 2.0, point["beta"], free=free)
        point["trace_A"] = trace
        point["gcv"] = gcv_score(phi_d, trace, J.shape[0])
    return point


def run_beta_selection(task: InversionTask, mesh=None) -> dict:
    """Choose beta by L-curve or GCV, then return the inversion at that beta.

    First runs the usual discrepancy-principle inversion; its final beta
    centres the sweep ``beta_disc * task.beta_sweep_factors`` (or the explicit
    ``task.beta_sweep``).  Every sweep point is a fixed-beta inversion (IRLS
    for sparse/L1–L2) sharing one set of sensitivities.  The result is that
    of the chosen beta, plus ``beta_selection`` with the whole curve.
    GCV needs a simulation with an explicit sensitivity matrix (potential
    fields); the L-curve works for any method.

    For IRLS the sweep reuses the discrepancy run's final eps (IRLS
    threshold) and does not cool it, so all betas minimize the same objective
    and the trade-off curve is monotone.
    """
    from ..methods.directives import stop_requested
    from ..methods.regparam import (
        LCURVE_NO_CORNER, gcv_minimum, lcurve_corner_info, sparse_terms,
    )

    criterion = task.beta_selection
    if criterion not in ("lcurve", "gcv"):
        raise ValueError(f"Unknown beta_selection '{criterion}'")
    if mesh is None:
        mesh = _build_mesh(task)

    p = _single_problem(task, mesh)
    base, _ = _run_problem(task, p)

    def keep_base(when):
        # stopped by the user: the discrepancy-principle run is the result
        base["converged"] = False
        base["stopped_early"] = {**base.get("stopped_early", {}), "reason": "stopped by the user "
                                 f"{when}; the discrepancy-principle result is kept"}
        return base

    if stop_requested():
        return keep_base("before the beta sweep")
    beta_disc = base["iterations"][-1]["beta"]
    sim = p.sim
    eps = [float(o.irls_threshold) for o in sparse_terms(p.reg)] \
        if p.kind not in ("smooth", "l2") else None
    focus_e = getattr(p.reg, "focusing_threshold", None)
    if task.beta_sweep is not None:
        betas = np.asarray(task.beta_sweep, dtype=float)
    else:
        betas = beta_disc * np.asarray(task.beta_sweep_factors, dtype=float)
    betas = np.sort(betas)[::-1]

    points, models = [], []
    for beta in betas:
        if stop_requested():
            return keep_base(f"after {len(points)} of {len(betas)} sweep points")
        result, p_fixed, inv_prob = run_fixed_beta(task, beta, mesh, sim=sim,
                                                   irls_thresholds=eps,
                                                   focusing_threshold=focus_e)
        m = np.asarray(result["recovered_model"])
        points.append(_selection_point(p_fixed, inv_prob, m))
        models.append(result)
        print(f"[beta sweep] beta={beta:.3e} phi_d={points[-1]['phi_d']:.4g} "
              f"phi_m={points[-1]['phi_m']:.4g}"
              + (f" GCV={points[-1]['gcv']:.4g}" if "gcv" in points[-1] else ""))

    curve = {k: [pt.get(k) for pt in points] for k in ("beta", "phi_d", "phi_m",
                                                        "trace_A", "gcv")}
    selection = {"criterion": criterion, "beta_discrepancy": float(beta_disc),
                 "n_data": int(len(task.observed_data)), "irls_thresholds": eps, **curve}
    corner = lcurve_corner_info(curve["beta"], curve["phi_d"], curve["phi_m"])
    selection["beta_lcurve"] = corner["beta"]
    warnings = []
    if not corner["valid"]:
        warnings.append(LCURVE_NO_CORNER + (
            "; the chosen beta is unreliable" if criterion == "lcurve" else ""))
    if all(g is not None for g in curve["gcv"]):
        selection["beta_gcv"] = gcv_minimum(curve["beta"], curve["gcv"])
    elif criterion == "gcv":
        raise ValueError("GCV needs a simulation with an explicit sensitivity matrix")
    if "beta_gcv" in selection and selection["beta_gcv"] in (betas.min(), betas.max()):
        warnings.append("GCV minimum is at the edge of the sweep; widen beta_sweep")
    selection["warnings"] = warnings
    for w in warnings:
        print(f"[beta sweep] Warning: {w}")
    chosen = selection[f"beta_{criterion}"]
    selection["beta_chosen"] = chosen

    result, _, _ = run_fixed_beta(task, chosen, mesh, sim=sim, irls_thresholds=eps,
                                  focusing_threshold=focus_e)
    result["beta_selection"] = selection
    result["discrepancy_model"] = base["recovered_model"]
    return result


def _collect_result(task, collector, m_recovered, inv_prob, method_label):
    """Gather iteration snapshots into a result dict."""
    iterations = []
    for snap in collector.snapshots:
        vals = np.asarray(snap.model_values)
        iterations.append({
            "iteration": snap.iteration,
            "phi_d": snap.phi_d,
            "phi_m": snap.phi_m,
            "phi_total": snap.phi_total,
            "beta": snap.beta,
            "model_min": float(vals.min()),
            "model_max": float(vals.max()),
            "model_mean": float(vals.mean()),
        })

    result = {
        "task_id": task.task_id,
        "method": task.method_type,
        "regularization": method_label,
        "converged": True,
        "n_iterations": len(iterations),
        "iterations": iterations,
        "recovered_model": m_recovered,
        "norms": list(task.norms),
    }
    stopped = getattr(collector, "stopped_at", None)
    if stopped is not None:   # the user stopped it: the model is that of this iteration
        result["converged"] = False
        result["stopped_early"] = {"reason": "stopped by the user", "at_iteration": int(stopped)}
    return result


def run_l1l2_cda(task: InversionTask, mesh=None) -> dict:
    """L1–L2 inversion as in Utsugi (2019): CDA along a lambda path.

    The elastic net is solved by coordinate descent with warm starts from
    lambda_max down ``task.lambda_decades`` decades in steps of
    ``task.lambda_step`` (log10), with ``task.l1l2_weighting`` depth weighting
    and ``l1_ratio`` fixed a priori; lambda is chosen by ``beta_selection``
    ("auto" = L-curve, as in the paper; also "discrepancy" or "gcv").  Needs
    an explicit sensitivity matrix (gravity, magnetics).  Magnetic models are
    solved in magnetization (A/m) as in the paper and returned as
    susceptibility.

    Each path point is reported as an "iteration" with ``beta`` = lambda,
    ``phi_d`` = chi^2 and ``phi_m`` = P(b; a).
    """
    from types import SimpleNamespace

    from ..methods.directives import IterationCollector, stop_requested
    from ..methods.l1l2_cda import invert_l1l2

    if mesh is None:
        mesh = _build_mesh(task)
    p = _single_problem(task, mesh)
    G = getattr(p.sim, "G", None)
    if G is None:
        raise ValueError("l1l2_solver='cda' needs a potential-field method with an "
                         "explicit sensitivity matrix; use l1l2_solver='irls'")
    model_unit = 1.0
    if canonical_method(task.method_type) == "magnetics":
        amplitude_nT = _get_method(task).inducing_field[0]
        model_unit = amplitude_nT * 1e-9 / (4e-7 * np.pi)  # SI -> A/m
    criterion = "lcurve" if task.beta_selection == "auto" else task.beta_selection

    def on_point(i, n_lambdas, lam, chi2, sweeps):
        # progress per lambda point, through the same hook as IRLS iterations
        callback = IterationCollector.on_iteration
        if callback is not None:
            callback(SimpleNamespace(iteration=i + 1, phi_d=chi2, beta=lam,
                                     extra={"max_iter": n_lambdas, "sweeps": sweeps}))

    # A reference model (geology constraints) makes the elastic net act on the deviation
    # from it: invert d - G m_ref for m - m_ref, with the bounds shifted the same way.
    # (Its per-cell weights have no place in the coordinate descent: use l1l2_solver="irls".)
    lower = task.cell_lower if task.cell_lower is not None else task.bounds_lower
    upper = task.cell_upper if task.cell_upper is not None else task.bounds_upper
    data = task.observed_data
    ref = None
    if task.reference_model is not None:
        ref = np.asarray(task.reference_model, dtype=float)
        data = data - np.asarray(G, dtype=float) @ ref
        lower = None if lower is None else np.asarray(lower, dtype=float) - ref
        upper = None if upper is None else np.asarray(upper, dtype=float) - ref
        if task.smallness_weights is not None:
            print("[L1–L2 CDA] Note: the geology weights are not used by the coordinate "
                  "descent (its reference model and bounds are)")
    res, scale = invert_l1l2(
        np.asarray(G), data, task.l1_ratio,
        weighting=task.l1l2_weighting, model_unit=model_unit, std=task.data_std,
        criterion=criterion, lower=lower, upper=upper,
        n_decades=task.lambda_decades, step=task.lambda_step,
        fallback=task.beta_selection == "auto", callback=on_point, should_stop=stop_requested,
    )
    if ref is not None:
        res.model = res.model + ref
    path = res.path
    chi2 = res.chi2
    iterations = []
    for i, (lam, beta) in enumerate(zip(path.lambdas, path.betas)):
        m = beta / (scale * model_unit) + (0.0 if ref is None else ref)
        iterations.append({
            "iteration": i,
            "phi_d": float(chi2[i]),
            "phi_m": float(path.penalty[i]),
            "phi_total": float(chi2[i] + lam * path.penalty[i]),
            "beta": float(lam),
            "model_min": float(m.min()),
            "model_max": float(m.max()),
            "model_mean": float(m.mean()),
        })
    predicted = np.asarray(G) @ res.model
    resid = (predicted - task.observed_data) / task.data_std
    for w in res.warnings:
        print(f"[L1–L2 CDA] Warning: {w}")
    stopped = {"stopped_early": {"reason": "stopped by the user", "at_iteration": len(path.lambdas),
                                 "of": path.n_planned}} if path.stopped else {}
    return {
        **stopped,
        "task_id": task.task_id,
        "method": task.method_type,
        "regularization": "elastic_net_CDA",
        "converged": not path.stopped,
        "n_iterations": len(iterations),
        "iterations": iterations,
        "recovered_model": res.model,
        "predicted": predicted,
        "l1_ratio": task.l1_ratio,
        "l1l2": {
            "solver": "cda",
            "weighting": task.l1l2_weighting,
            "criterion": res.criterion,
            "model_unit": model_unit,
            "lambda_opt": res.lambda_opt,
            "lambda_lcurve": res.lambda_lcurve,
            "lambda_discrepancy": res.lambda_discrepancy,
            "lambda_gcv": res.lambda_gcv,
            "chi2_opt": float(resid @ resid),
            "lambdas": path.lambdas,
            "residual_norm": path.residual_norm,
            "penalty": path.penalty,
            "chi2": chi2,
            "gcv": res.gcv,
            "sweeps": path.sweeps,
            "warnings": res.warnings,
        },
    }


def run_single_inversion(task: InversionTask, mesh=None) -> dict:
    """Route to smooth or sparse (lp-norm or L1–L2) based on task config.

    L1–L2 with ``l1l2_solver="cda"`` (the default) runs Utsugi's (2019)
    coordinate-descent path (:func:`run_l1l2_cda`).  Otherwise
    ``task.beta_selection`` "lcurve" or "gcv" chooses beta by that criterion
    (see :func:`run_beta_selection`); "auto"/"discrepancy" cools beta to the
    target misfit.
    """
    if getattr(_get_method(task), "vector", False):
        if task.regularization_type == "bayes":
            raise ValueError("The Bayesian inversion is for induced magnetization (and gravity): "
                             "the magnetization vector is not linear-Gaussian")
        return run_mvi_inversion(task, mesh)
    if task.regularization_type == "bayes":
        return run_bayesian_inversion(task, mesh)
    if task.regularization_type == "l1l2":
        if task.l1l2_solver == "cda":
            return run_l1l2_cda(task, mesh)
        if task.l1l2_solver != "irls":
            raise ValueError(f"Unknown l1l2_solver '{task.l1l2_solver}' "
                             "(expected 'cda' or 'irls')")
    if task.beta_selection not in ("auto", "discrepancy"):
        return run_beta_selection(task, mesh)
    if task.regularization_type in ("sparse", "l1l2", "mgs", "tv"):
        return run_sparse_inversion(task, mesh)
    return run_smooth_inversion(task, mesh)


BAYES_COMPACT_NORMS = (0.0, 1.0, 1.0, 1.0)   # the compact prior: Lp norms of the most probable model
BAYES_MAG_UPPER = 1.0      # SI: the compact prior's susceptibility cap unless an upper bound is given
# The Laplace approximation's IRLS eps, at least this share of the body threshold (per cell
# size for the gradients): with eps -> 0 a p = 0 weight is ~(m_max / eps)^2 larger in the
# empty cells than in the bodies, which then cannot grow or move at all in the samples.
BAYES_EPS_SHARE = float(os.environ.get("GEOINV3D_BAYES_EPS_SHARE", "0.25"))


def _under_stations(task, mesh):
    """The active cells under the stations' extent, and the active cells' volumes."""
    dmesh = mesh.to_discretize()
    act = np.ones(dmesh.n_cells, bool) if task.active_cells is None else np.asarray(task.active_cells, bool)
    cc, vol = dmesh.cell_centers[act], dmesh.cell_volumes[act]
    st = np.asarray(task.station_locations, dtype=float)
    under = ((cc[:, 0] >= st[:, 0].min()) & (cc[:, 0] <= st[:, 0].max())
             & (cc[:, 1] >= st[:, 1].min()) & (cc[:, 1] <= st[:, 1].max()))
    return under, vol


def _temper_irls(reg, mode, task, mesh):
    """Floor the IRLS eps of the sparse terms at BAYES_EPS_SHARE of the mode's body threshold
    (divided by the core cell size for the gradients) and recompute their weights at the
    mode.  Returns the floors, {smallness, gradient}, or None when there is nothing to do."""
    from simpeg.regularization import Sparse
    from simpeg.regularization.sparse import SparseSmallness

    from ..methods.bayes import body_threshold
    if BAYES_EPS_SHARE <= 0:
        return None
    under, vol = _under_stations(task, mesh)
    thr = body_threshold(mode, under, weights=vol)
    if not np.isfinite(thr) or thr <= 0:
        return None
    h = float(min(np.asarray(mesh.to_discretize().h_gridded).min(axis=0)))
    floors = {"smallness": BAYES_EPS_SHARE * thr, "gradient": BAYES_EPS_SHARE * thr / max(h, 1e-9)}
    regs = [reg] if isinstance(reg, Sparse) else [r for r in getattr(reg, "objfcts", []) if isinstance(r, Sparse)]
    for r in regs:
        for f in r.objfcts:
            key = "smallness" if isinstance(f, SparseSmallness) else "gradient"
            f.irls_threshold = max(float(f.irls_threshold), floors[key])
        r.update_weights(mode)
    return floors


def run_bayesian_inversion(task: InversionTask, mesh=None) -> dict:
    """The posterior of a gravity or (induced) magnetic task (methods/bayes.py), with one
    of two priors (``task.bayes_prior``):

    * "compact" (default): the sources are compact bodies.  The most probable model is the
      Lp inversion (norms BAYES_COMPACT_NORMS, susceptibility >= 0 unless bounds are given),
      and the samples are the Gaussian (Laplace) approximation around it, from its last
      IRLS weights, cut at the bounds.  The spread is that of models like it: where the
      bodies are, how strong, how far they reach given that they are compact.
    * "smooth": the smooth L2 regularization as a Gaussian prior; the posterior is exact
      (linear-Gaussian) and wide, its mean the smooth L2 model.

    beta: chi^2 = N with the errors given (``bayes_beta`` "evidence": beta and an error
    scale by the largest evidence, smooth prior).  ``task.bayes_samples`` RML samples.  The
    model is the mode (compact) or mean (smooth); the result adds ``posterior_std``,
    ``prob_body`` (the share of samples above the threshold: ``bayes_threshold``, else half
    the model's 98th percentile under the stations, by volume), ``prior_std``, ``samples``.
    """
    from dataclasses import replace
    from types import SimpleNamespace

    from ..methods.bayes import LinearGaussian, body_threshold, prior_terms
    from ..methods.directives import IterationCollector
    from simpeg import inverse_problem, inversion

    if mesh is None:
        mesh = _build_mesh(task)
    prior_kind = str(getattr(task, "bayes_prior", "compact") or "compact")
    if prior_kind not in ("compact", "smooth"):
        raise ValueError(f"Unknown bayes_prior '{prior_kind}' (expected 'compact' or 'smooth')")
    magnetic = canonical_method(task.method_type) == "magnetics"
    lo = task.bounds_lower if task.bounds_lower is not None else (0.0 if magnetic else None)
    hi = task.bounds_upper
    if prior_kind == "compact" and magnetic and hi is None:
        hi = BAYES_MAG_UPPER   # unbounded, p = 0 packs the anomaly into a few cells of 10+ SI
    callback = IterationCollector.on_iteration
    map_result = None

    def report(step, i, n, extra=None):
        if callback is not None:
            callback(SimpleNamespace(iteration=i, phi_d=None, beta=None,
                                     extra={"max_iter": n, "step": step, **(extra or {})}))

    if prior_kind == "compact":
        sp_task = replace(task, regularization_type="sparse", norms=BAYES_COMPACT_NORMS,
                          bounds_lower=lo, bounds_upper=hi, beta_selection="auto")
        p = _single_problem(sp_task, mesh)
        print(f"[Bayes] the most probable compact model: Lp {BAYES_COMPACT_NORMS}, bounds [{lo}, {hi}]")
        map_result, _ = _run_problem(sp_task, p)
        mode = np.asarray(map_result["recovered_model"], dtype=float)
        eps_floor = _temper_irls(p.reg, mode, task, mesh)
        terms = prior_terms(p.reg, mode)          # its last IRLS weights (eps floored)
    else:
        l2 = replace(task, regularization_type="l2")
        p = _single_problem(l2, mesh)
        if _uses_sensitivity_weights(l2, "l2"):
            # the weights a smooth L2 run would set, from the same directive
            directive = _sensitivity_directive(l2)
            inv_prob = inverse_problem.BaseInvProblem(p.dmis, p.reg, p.opt)
            inversion.BaseInversion(inv_prob, directiveList=[directive])
            inv_prob.model = p.m0
            directive.initialize()
        terms = prior_terms(p.reg, p.m0)
        mode = None
        eps_floor = None
    G = getattr(p.sim, "G", None)
    if G is None:
        raise ValueError("The Bayesian inversion needs a gravity or magnetic task (an explicit "
                         "sensitivity matrix)")
    m_ref = np.asarray(getattr(p.reg, "reference_model", None) if getattr(p.reg, "reference_model", None)
                       is not None else np.zeros(len(p.m0)), dtype=float)
    d, sigma = np.asarray(task.observed_data, dtype=float), np.asarray(task.data_std, dtype=float)
    n_data, n_cells = len(d), len(p.m0)
    print(f"[Bayes] {prior_kind} prior, {n_data} data x {n_cells} cells: factorizing the prior's "
          f"precision, then {n_data} solves")
    post = LinearGaussian(np.asarray(G), d, sigma, terms, m_ref=m_ref,
                          progress=lambda a, n: report("prior solves", a, n))
    rule = str(getattr(task, "bayes_beta", "discrepancy") or "discrepancy")
    beta_chi2 = post.beta_for(float(n_data))
    beta_ev, noise2_ev = post.evidence_beta_noise()
    beta, noise2 = (beta_ev, noise2_ev) if (rule == "evidence" and prior_kind == "smooth") else (beta_chi2, 1.0)
    n = int(getattr(task, "bayes_samples", 30) or 30)
    seed = int(getattr(task, "bayes_seed", 0) or 0)
    if prior_kind == "compact":
        # the Laplace approximation around the mode: its spread, added to it, cut at the bounds
        delta, prior = post.perturbations(beta, n, seed=seed, noise2=noise2,
                                          progress=lambda k, n: report("samples", k, n))
        samples = (mode[None, :] + delta).astype(np.float32)
        model = mode
    else:
        samples, prior = post.samples(beta, n, seed=seed, noise2=noise2,
                                      progress=lambda k, n: report("samples", k, n))
        model = post.mean(beta * noise2)
    # cut at the bounds: per cell where geology constraints give them, else the job's
    clo = np.asarray(task.cell_lower, dtype=float) if task.cell_lower is not None else lo
    chi = np.asarray(task.cell_upper, dtype=float) if task.cell_upper is not None else hi
    if clo is not None or chi is not None:
        np.clip(samples, -np.inf if clo is None else clo, np.inf if chi is None else chi, out=samples)
    chi2 = float(np.sum(((np.asarray(G) @ model - d) / sigma) ** 2))
    print(f"[Bayes] beta {beta:.4g} ({rule}): chi^2/N of the model {chi2 / n_data:.3f}; "
          f"timing {post.timing}")
    # a body: above half the model's 98th percentile under the stations, weighted by cell
    # volume (an octree's many small cells near the surface would raise it otherwise)
    under, vol = _under_stations(task, mesh)
    thr = getattr(task, "bayes_threshold", None)
    thr = float(thr) if thr is not None else body_threshold(model, under, weights=vol)
    prob = (samples >= thr).mean(axis=0)
    std = samples.std(axis=0, ddof=1)
    prior_std = prior.std(axis=0, ddof=1)
    if map_result is not None:
        iterations = map_result.get("iterations", [])
    else:
        iterations = [{"iteration": 0, "phi_d": chi2, "phi_m": None, "phi_total": None, "beta": beta,
                       "model_min": float(model.min()), "model_max": float(model.max()),
                       "model_mean": float(model.mean())}]
    bounds_label = "per cell (geology)" if (task.cell_lower is not None or task.cell_upper is not None) \
        else f"[{lo}, {hi}]"
    ref_label = ", the geology reference" if task.reference_model is not None else ""
    prior_label = (f"compact (Lp {', '.join(f'{v:g}' for v in BAYES_COMPACT_NORMS)}, bounds {bounds_label}{ref_label}): "
                   "Laplace approximation around the most probable model") if prior_kind == "compact" \
        else f"smooth L2 (Gaussian), reference {'the geology' if task.reference_model is not None else 'model 0'}"
    return {
        "task_id": task.task_id, "method": task.method_type, "regularization": "bayesian_L2",
        "converged": True, "n_iterations": len(iterations), "iterations": iterations,
        "recovered_model": model, "predicted": np.asarray(G) @ model,
        "norms": list(BAYES_COMPACT_NORMS if prior_kind == "compact" else task.norms),
        "posterior_std": std, "prob_body": prob, "prior_std": prior_std, "samples": samples,
        "bayes_spectrum": {"lam": post.lam, "c": post._coefficients(post.d, post.m_ref)},
        "bayes": {"prior": prior_label, "prior_kind": prior_kind, "bounds": [lo, hi],
                  "eps_floor": eps_floor if prior_kind == "compact" else None,
                  "beta": beta, "beta_rule": rule if prior_kind == "smooth" else "discrepancy",
                  "error_scale": math.sqrt(noise2), "beta_evidence": beta_ev,
                  "error_scale_evidence": math.sqrt(noise2_ev), "beta_discrepancy": beta_chi2,
                  "chi2_per_datum": chi2 / n_data, "chi2_per_datum_scaled": chi2 / n_data / noise2,
                  "n_samples": n, "threshold": thr,
                  "threshold_rule": "given" if getattr(task, "bayes_threshold", None) is not None
                  else "half the model's 98th percentile",
                  "std_median": float(np.median(std)), "variance_reduction_median":
                  float(np.median(1.0 - (std / np.maximum(prior_std, 1e-30)) ** 2)),
                  "timing_s": {k: round(v, 1) for k, v in post.timing.items()}},
    }


def magnetization_directions(M: np.ndarray, share: float = 0.1) -> dict:
    """Directions of the magnetization vectors (n, 3: east, north, up) of the cells whose
    amplitude exceeds ``share`` of the largest: inclination (degrees, positive down) and
    declination (degrees east of north) of their amplitude-weighted resultant, and the
    median of the cells' own."""
    M = np.asarray(M, dtype=float)
    amp = np.linalg.norm(M, axis=1)
    if not amp.size or not amp.max() > 0:
        return {"n_cells": 0}
    strong = amp > share * amp.max()
    S = M[strong]
    inc = np.degrees(np.arctan2(-S[:, 2], np.hypot(S[:, 0], S[:, 1])))
    dec = np.degrees(np.arctan2(S[:, 0], S[:, 1]))
    r = S.sum(axis=0)       # the amplitude-weighted resultant of the strong cells
    return {"n_cells": int(strong.sum()), "share_of_max": share,
            "amplitude_max": float(amp.max()),
            "resultant_inclination": float(np.degrees(np.arctan2(-r[2], np.hypot(r[0], r[1])))),
            "resultant_declination": float(np.degrees(np.arctan2(r[0], r[1]))),
            "resultant_coherence": float(np.linalg.norm(r) / np.sum(amp[strong])),
            "median_inclination": float(np.median(inc)),
            "median_declination": float(np.degrees(np.angle(np.median(np.cos(np.radians(dec)))
                                                             + 1j * np.median(np.sin(np.radians(dec))))))}


def run_mvi_inversion(task: InversionTask, mesh=None) -> dict:
    """Magnetization-vector inversion (MVI; SimPEG's Cartesian formulation).

    Three components per cell (effective susceptibility along east, north and up), so a
    remanent magnetization in any direction can be fitted, which a susceptibility along
    the present field cannot (the systematic residual of the Karnataka magnetic data).
    The regularization is SimPEG's VectorAmplitude: the task's norms (sparse) or 2 (l2) on
    the amplitude |m| and its gradients, with the task's alphas, depth weighting and
    reference 0; bounds_upper bounds every component (|m_i| <= upper).  beta is cooled to
    chi^2 = N and steered there during IRLS, as for the scalar sparse inversion.

    The result's ``recovered_model`` is the amplitude (SI) of each cell,
    ``magnetization_vector`` the (n, 3) components, ``magnetization`` the directions of the
    strong cells (:func:`magnetization_directions`) next to the inducing field's.
    """
    from ..datamodel.survey import SurveyData
    from ..methods.directives import DampedUpdateIRLS, IterationCollector
    from simpeg import directives, inverse_problem, inversion, maps, optimization, regularization

    if task.regularization_type not in ("sparse", "l2"):
        raise ValueError("The magnetization-vector inversion takes regularization_type "
                         f"'sparse' or 'l2', not '{task.regularization_type}'")
    if task.beta_selection not in ("auto", "discrepancy"):
        raise ValueError("The magnetization-vector inversion chooses beta by chi^2 = N")
    if mesh is None:
        mesh = _build_mesh(task)
    method = _get_method(task)
    survey = SurveyData(locations=task.station_locations, observed=task.observed_data,
                        std=task.data_std, method="magnetics")
    active = task.active_cells
    sim = (method.make_simulation_active(mesh, survey, active) if active is not None
           else method.make_simulation_full(mesh, survey))
    dmis = method.make_dmis(survey, sim)
    dmesh = mesh.to_discretize()
    n = int(active.sum()) if active is not None else dmesh.n_cells
    norms = list(task.norms) if task.regularization_type == "sparse" else [2.0, 2.0, 2.0, 2.0]
    reg_kwargs = dict(alpha_s=effective_alpha_s(task), length_scale_x=task.alpha_x,
                      length_scale_y=task.alpha_y, length_scale_z=task.alpha_z,
                      reference_model=np.zeros(3 * n), norms=norms)
    if active is not None:
        reg_kwargs["active_cells"] = active
    reg = regularization.VectorAmplitude(dmesh, mapping=maps.IdentityMap(nP=3 * n), **reg_kwargs)
    depth = getattr(task, "depth_weighting", "sensitivity") == "depth"
    if depth:
        reg.set_weights(depth=_depth_weights(task, dmesh))
    if task.bounds_lower is not None and task.bounds_lower > 0:
        print("[MVI] Note: bounds_lower does not apply (the components have either sign)")
    upper = float(task.bounds_upper) if task.bounds_upper is not None else np.inf
    m0 = np.full(3 * n, 1e-4 if not np.isfinite(upper) else min(1e-4, 1e-4 * upper))
    opt = optimization.ProjectedGNCG(maxIter=task.max_iter, lower=-upper, upper=upper,
                                     maxIterLS=20, cg_maxiter=30, cg_atol=1e-4, cg_rtol=0.0)
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    collector = IterationCollector()
    directive_list = ([] if depth else [_sensitivity_directive(task)]) + [
        directives.BetaEstimate_ByEig(beta0_ratio=task.beta0_ratio, random_seed=42),
        directives.TargetMisfit(chifact=1.0),
        DampedUpdateIRLS(f_min_change=1e-4, max_irls_iterations=task.max_irls_iterations,
                         chifact_start=3.0, irls_cooling_factor=task.irls_cooling_factor,
                         cooling_factor=task.cooling_factor),
        directives.UpdatePreconditioner(),
        collector,
    ]
    print(f"[MVI] {n} cells x 3 components, norms {norms}, "
          f"{'depth' if depth else 'sensitivity'} weighting, |m_i| <= {upper:g}")
    m_rec = inversion.BaseInversion(inv_prob, directiveList=directive_list).run(m0)
    M = np.asarray(m_rec, dtype=float).reshape(3, n).T
    amplitude = np.linalg.norm(M, axis=1)
    label = "mvi_sparse_IRLS" if task.regularization_type == "sparse" else "mvi_L2"
    result = _collect_result(task, collector, amplitude, inv_prob, label)
    if task.regularization_type == "l2":
        result.pop("norms")
    result["magnetization_vector"] = M
    field = method.inducing_field
    result["magnetization"] = {**magnetization_directions(M),
                               "inducing_inclination": float(field[1]),
                               "inducing_declination": float(field[2]),
                               "components": "east, north, up (effective susceptibility, SI)"}
    result["depth_weighting"] = "depth" if depth else "sensitivity"
    if depth:
        result["depth_weighting_exponent"] = float(task.depth_weighting_exponent)
    try:
        result["predicted"] = np.asarray(sim.dpred(m_rec), dtype=float)
    except Exception as e:   # the result is still useful without it
        print(f"[MVI] Could not compute the predicted data: {e}")
    return result


JOINT_REGULARIZATION_TYPES = ("l2", "sparse", "l1l2", "mgs", "tv")


def joint_regularization(task: InversionTask, i: int):
    """The ModelRegularization of joint dataset ``i``'s model, or None (legacy path).

    The task's settings, overridden by ``task.joint_regularizations[i]``
    (keys JOINT_REG_KEYS).  A regularization_type outside
    JOINT_REGULARIZATION_TYPES selects the legacy WeightedLeastSquares (None).
    """
    from ..methods.joint import ModelRegularization

    over = {}
    if task.joint_regularizations and i < len(task.joint_regularizations):
        over = dict(task.joint_regularizations[i] or {})
    unknown = set(over) - set(JOINT_REG_KEYS)
    if unknown:
        raise ValueError(f"Unknown joint regularization settings {sorted(unknown)} "
                         f"(expected some of {JOINT_REG_KEYS})")

    def get(key):
        return over[key] if key in over else getattr(task, key)
    kind = get("regularization_type")
    if kind not in JOINT_REGULARIZATION_TYPES:
        return None
    alpha_s = over.get("alpha_s", task.alpha_s)
    return ModelRegularization(
        kind=kind,
        alpha_s=LENGTH_SCALE_ALPHA_S if alpha_s is None else float(alpha_s),
        length_scale_x=float(get("alpha_x")), length_scale_y=float(get("alpha_y")),
        length_scale_z=float(get("alpha_z")),
        norms=tuple(float(v) for v in get("norms")), l1_ratio=float(get("l1_ratio")),
        focusing_percentile=float(get("focusing_percentile")),
        focusing_scale=get("focusing_scale"),
        depth_weighting=get("depth_weighting"),
        depth_weighting_exponent=float(get("depth_weighting_exponent")),
        lower=get("bounds_lower"), upper=get("bounds_upper"),
    )


def _joint_setups(task: InversionTask, mesh, regularized: bool = True):
    """MethodSetups of a joint task: one per dataset, on the task's active cells."""
    from ..datamodel.survey import SurveyData
    from ..methods.joint import MethodSetup

    active = task.active_cells
    n_params = int(active.sum()) if active is not None else mesh.n_cells
    labels = list(task.joint_models or [])
    setups = []
    for i, method_name in enumerate(task.joint_methods):
        method_name = canonical_method(method_name)
        kwargs = (task.joint_kwargs_list[i] if task.joint_kwargs_list
                  and i < len(task.joint_kwargs_list) else None) or {}
        method = _make_method(method_name, kwargs)
        sd = task.joint_surveys[i]
        survey = SurveyData(locations=sd["locations"], observed=sd["observed"], std=sd["std"],
                            method=method_name, name=str(sd.get("name") or ""))
        weight = task.joint_weights[i] if task.joint_weights else 1.0
        initial = task.joint_initial_models[i] if task.joint_initial_models \
            and i < len(task.joint_initial_models) else None
        if initial is None or len(initial) != n_params:
            initial = np.full(n_params, float(method.default_model_value))
        setups.append(MethodSetup(
            method=method, survey=survey, mesh=mesh, initial_model=initial, weight=weight,
            active_cells=active, model=(labels[i] if i < len(labels) else None) or None,
            regularization=joint_regularization(task, i) if regularized else None,
        ))
    return setups


def run_joint_inversion(task: InversionTask, mesh=None) -> dict:
    """Execute a joint inversion from a task specification.

    Each model is regularized as :func:`joint_regularization` says: l2,
    sparse, l1l2 (IRLS), mgs or tv as in a single-method inversion, with
    depth weighting, bounds and per-model overrides; any other
    regularization_type keeps the original joint path (WeightedLeastSquares
    with alpha_s and alpha_x/y/z as length scales, BetaSchedule cooling).
    The models are coupled as ``task.coupling`` says (cross-gradient, joint total
    variation, linear correspondence, PGI or none; see methods/coupling.py) with the
    unit-free ``coupling_weight``.  Beta follows the discrepancy principle (chi^2 = N);
    L-curve / GCV sweeps are single-method.
    """
    from ..methods.joint import JointInversion

    if mesh is None:
        mesh = _build_mesh(task)
    setups = _joint_setups(task, mesh)
    joint = JointInversion(
        setups=setups,
        max_iter=task.max_iter,
        beta0_ratio=task.beta0_ratio,
        cooling_factor=task.cooling_factor,
        coupling=task.coupling,
        coupling_weight=task.effective_coupling_weight,
        coupling_options=task.coupling_options,
        reg_kwargs=dict(
            alpha_s=effective_alpha_s(task),
            length_scale_x=task.alpha_x,
            length_scale_y=task.alpha_y,
            length_scale_z=task.alpha_z,
        ),
        max_irls_iterations=task.max_irls_iterations,
        irls_cooling_factor=task.irls_cooling_factor,
        use_preconditioner=task.use_preconditioner,
        balance=task.joint_balance,
    )

    result = joint.run()

    iterations = []
    for snap in result.iterations:
        vals = np.asarray(snap.model_values)
        iterations.append({
            "iteration": snap.iteration,
            "phi_d": snap.phi_d,
            "phi_m": snap.phi_m,
            "phi_total": snap.phi_total,
            "beta": snap.beta,
            "model_min": float(vals.min()),
            "model_max": float(vals.max()),
            "model_mean": float(vals.mean()),
        })

    models = result.extras["models"]
    labels = sorted({info["regularization"] for info in models.values()})
    joint_data = {}
    for (label, info), setup in zip(result.extras["datasets"].items(), setups):
        if "predicted" in info:
            joint_data[label] = {"locations": setup.survey.locations,
                                 "observed": setup.survey.observed, "std": setup.survey.std,
                                 "predicted": info["predicted"]}
    stopped = {} if result.converged else {
        "stopped_early": {"reason": "stopped by the user", "at_iteration": len(iterations)}}
    return {
        **stopped,
        "task_id": task.task_id,
        "methods": [canonical_method(m) for m in task.joint_methods],
        "regularization": "joint_pgi" if joint.coupling == "pgi" else "joint_L2" if joint.legacy
        else "joint_" + (labels[0] if len(labels) == 1 else "mixed"),
        "coupling": result.extras["coupling"],
        **({"pgi": result.extras["pgi"]} if "pgi" in result.extras else {}),
        "cross_gradient_weight": joint.cross_gradient_weight,
        "converged": result.converged,
        "n_iterations": len(iterations),
        "iterations": iterations,
        "recovered_models": {k: v for k, v in result.recovered_models.items()},
        "models": {name: {k: v for k, v in info.items()} for name, info in models.items()},
        "dataset_models": list(joint.model_labels),
        "dataset_labels": list(joint.dataset_labels),
        "chi2": {label: info.get("chi2") for label, info in result.extras["datasets"].items()},
        # SimPEG sign convention; the pipeline turns them back into the files' convention
        "joint_data": joint_data,
    }


GROUP_LASSO_SELECTIONS = ("lcurve", "discrepancy", "fixed", "search")


def _group_lasso_problem(task: InversionTask, mesh):
    """The GroupLassoProblem of a joint task, with its setups and labels.

    One model per model label (see :func:`geoinv3d.methods.joint.assign_models`):
    by default one per dataset, e.g. "gravity" and "magnetics"; datasets
    with the same label share a model.  Potential-field datasets contribute
    their SimPEG sensitivity matrices (float32, as stored); MT and DC
    datasets their simulations (log-conductivity, relative to the method's
    background), linearized at every Gauss–Newton step.
    """
    from ..methods.group_lasso import GroupLassoData, GroupLassoProblem
    from ..methods.joint import assign_models

    setups = _joint_setups(task, mesh, regularized=False)
    model_labels, dataset_labels = assign_models(setups)
    names = list(dict.fromkeys(model_labels))
    if len(names) < 2:
        raise ValueError("The group lasso couples at least two models (e.g. a gravity and a "
                         f"magnetic dataset); these datasets make one model: {names}")
    active = task.active_cells
    n_params = int(active.sum()) if active is not None else mesh.n_cells
    datasets, refs, kinds = [], [None] * len(names), [None] * len(names)
    for setup, model, label in zip(setups, model_labels, dataset_labels):
        p = names.index(model)
        method, survey = setup.method, setup.survey
        kind = (bool(method.linear), float(method.default_model_value))
        if kinds[p] is not None and kinds[p] != kind:
            raise ValueError(f"The datasets of model '{model}' measure different properties "
                             "(a linear and a nonlinear method, or different backgrounds)")
        kinds[p] = kind
        if getattr(method, "vector", False):
            raise ValueError("The magnetization-vector inversion (MVI) is for single magnetic "
                             "inversions in this version; the group lasso couples the induced "
                             "susceptibility")
        sim = (method.make_simulation_active(mesh, survey, active) if active is not None
               else method.make_simulation_full(mesh, survey))
        n = len(survey.observed)
        if method.linear:
            print(f"[Group lasso] {label}: computing the {n} x {n_params} sensitivity matrix")
            datasets.append(GroupLassoData(label, survey.observed, p, survey.std,
                                           operator=sim.G))
        else:
            print(f"[Group lasso] {label}: {n} data, nonlinear (Gauss–Newton)")
            datasets.append(GroupLassoData(label, survey.observed, p, survey.std,
                                           simulation=sim))
        if refs[p] is None:
            refs[p] = np.full(n_params, float(method.default_model_value))
    dmesh = mesh.to_discretize() if task.gl_cross_gradient > 0 else None
    bounds = group_lasso_bounds(task, model_labels, names)
    if any(b != (None, None) for b in bounds):
        print("[Group lasso] bounds: " + ", ".join(
            f"{n} {'per cell' if np.ndim(b[0]) else b[0]}..{'per cell' if np.ndim(b[1]) else b[1]}"
            for n, b in zip(names, bounds)))
    problem = GroupLassoProblem(
        datasets, task.gl_mu, model_names=names, references=refs, gamma=task.gl_gamma,
        data_scaling=task.gl_data_scaling, mesh=dmesh,
        active_cells=active if dmesh is not None else None, bounds=bounds,
        cell_weights=group_lasso_cell_weights(task, mesh, setups, model_labels, names),
        cell_factors=group_lasso_cell_factors(task, mesh, names),
        relaxation=float(task.gl_relaxation))
    return problem, setups, model_labels, dataset_labels


GROUP_LASSO_WEIGHTINGS = ("sensitivity", "sensitivity_volume", "depth")
BALANCE_TOLERANCE = 1.2     # each model's chi^2 / N within 1/1.2 .. 1.2
DISCREPANCY_TOLERANCE = 0.05


def _discrepancy_solve(problem, lam1, lam2, state, n_data, solve_kw, should_stop=None,
                       max_solves: int = 6, log=print):
    """Solve for the lambda1 whose total chi^2 = N, from ``lam1`` and the warm start
    ``state``: a secant in log(lambda1) against log(chi^2) (chi^2 grows with lambda1), at
    most ``max_solves`` solves, each step at most a decade.  Returns the solve closest
    to chi^2 = N and its lambda1."""
    points, best = [], None
    for _ in range(max_solves):
        res = problem.solve(lam1, lam2, state=state, should_stop=should_stop, **solve_kw)
        chi2 = problem.chi2_of(res)["total"]
        log(f"[Group lasso]   lambda1 = {lam1:.4g}: chi2 = {chi2:.4g} (target {n_data}), "
            f"{res.n_iterations} ADMM iterations")
        miss = abs(np.log(chi2 / n_data))
        if best is None or miss < best[2]:
            best = (res, lam1, miss)
        if miss <= DISCREPANCY_TOLERANCE or res.stopped:
            break
        state = res.state
        points.append((np.log(lam1), np.log(chi2)))
        slope = 0.5       # d log chi^2 / d log lambda1 near chi^2 = N on the Karnataka sweeps
        if len(points) >= 2 and points[-1][0] != points[-2][0]:
            slope = max((points[-1][1] - points[-2][1]) / (points[-1][0] - points[-2][0]), 0.05)
        step = float(np.clip((np.log(n_data) - points[-1][1]) / slope, -np.log(10), np.log(10)))
        lam1 = float(np.exp(points[-1][0] + step))
    return best[0], best[1]


def _balance_group_lasso(problem, result, lam1, lam2, solve_kw, rounds, should_stop=None,
                         log=print):
    """Reweigh the models' data until each model fits its data to chi^2 = N (gl_balance).

    Each round multiplies model p's data weight by sqrt(chi2_p / N_p), normalized to a
    geometric mean of 1 over the data, at most 4x per round (GroupLassoProblem.scale_models:
    only the data terms change; the operators and their factors stay), and solves for the
    total chi^2 = N again, warm-started from the same physical models.  Stops when each
    chi2_p / N_p is within BALANCE_TOLERANCE, after ``rounds`` rounds, or when a round does
    not narrow the spread.  Returns (result, lambda1, history)."""
    n_p = np.array([sum(problem.datasets[i].data.size for i in ds)
                    for ds in problem.model_datasets], dtype=float)
    n_data = float(n_p.sum())
    history = []
    for k in range(rounds + 1):
        chi = problem.chi2_of(result)
        r = np.array([sum(chi[problem.datasets[i].name] for i in ds)
                      for ds in problem.model_datasets]) / n_p
        history.append({"round": k, "lambda1": float(lam1),
                        "data_weights": [float(w) for w in problem.data_weights],
                        "chi2_per_datum": {n: float(x) for n, x in zip(problem.model_names, r)},
                        "admm_iterations": int(result.n_iterations)})
        log(f"[Group lasso] balance round {k}: chi2 / N " + ", ".join(
            f"{n} {x:.3g}" for n, x in zip(problem.model_names, r)) + ", data weights " +
            ", ".join(f"{w:.3g}" for w in problem.data_weights))
        spread = float(np.max(np.abs(np.log(r))))
        history[-1]["spread"] = spread
        if spread <= np.log(BALANCE_TOLERANCE) or k == rounds or result.stopped \
                or (should_stop is not None and should_stop()):
            break
        if k and spread >= 0.9 * history[-2]["spread"]:
            log("[Group lasso] balance: the last round did not narrow the spread; stopping")
            break
        f = np.clip(np.sqrt(r), 0.25, 4.0)
        f = f / np.exp(np.sum(n_p * np.log(f)) / n_data)
        state = problem.scale_models(f, result.state)
        result, lam1 = _discrepancy_solve(problem, lam1, lam2, state, n_data, solve_kw,
                                          should_stop, log=log)
    return result, lam1, history


def group_lasso_cell_factors(task: InversionTask, mesh, names):
    """gl_weighting "sensitivity_volume": each cell's volume over the smallest one, for every
    model (GroupLassoProblem cell_factors); else None."""
    if task.gl_weighting != "sensitivity_volume":
        return None
    dmesh = mesh.to_discretize()
    volume = np.asarray(dmesh.cell_volumes)
    if task.active_cells is not None:
        volume = volume[task.active_cells]
    factor = volume / volume.min()
    print(f"[Group lasso] cells weighed by sensitivity (gamma = {task.gl_gamma:g}) x volume / "
          f"smallest volume (1 to {factor.max():.3g})")
    return [factor] * len(names)


def group_lasso_cell_weights(task: InversionTask, mesh, setups, model_labels, names):
    """The group lasso's cell weights (GroupLassoProblem cell_weights), or None.

    gl_weighting "depth": per model, cell volume x the Li & Oldenburg depth weight below
    its datasets' stations, with the depth_weighting_exponent of its regularization (its
    first dataset's joint_regularizations, else the task's) — the weights a SimPEG
    regularization with depth weighting gives its cells.
    """
    from ..methods.joint import depth_weights

    if task.gl_weighting not in GROUP_LASSO_WEIGHTINGS:
        raise ValueError(f"Unknown gl_weighting '{task.gl_weighting}' "
                         f"(expected one of {GROUP_LASSO_WEIGHTINGS})")
    if task.gl_weighting in ("sensitivity", "sensitivity_volume"):
        return None
    dmesh = mesh.to_discretize()
    active = task.active_cells if task.active_cells is not None \
        else np.ones(dmesh.n_cells, dtype=bool)
    volume = np.asarray(dmesh.cell_volumes)[active]
    overrides = task.joint_regularizations or []
    out = []
    for name in names:
        idx = [i for i, m in enumerate(model_labels) if m == name]
        over = dict(overrides[idx[0]] or {}) if idx[0] < len(overrides) else {}
        beta = float(over.get("depth_weighting_exponent", task.depth_weighting_exponent))
        locs = np.vstack([np.asarray(setups[i].survey.locations)[:, :3] for i in idx])
        wz = depth_weights(dmesh, locs, active, beta)
        out.append(volume * wz)
        print(f"[Group lasso] {name}: cells weighed by volume x depth weight (beta = {beta:g})")
    return out


def group_lasso_bounds(task: InversionTask, model_labels, names):
    """(lower, upper) of each group-lasso model: its first dataset's
    joint_regularizations bounds_lower / bounds_upper, else the task's."""
    overrides = task.joint_regularizations or []
    bounds = []
    for name in names:
        i = model_labels.index(name)
        over = dict(overrides[i] or {}) if i < len(overrides) else {}
        bounds.append((over.get("bounds_lower", task.bounds_lower),
                       over.get("bounds_upper", task.bounds_upper)))
    return bounds


def run_group_lasso_joint(task: InversionTask, mesh=None) -> dict:
    """Joint inversion with L2 + group lasso (Utsugi 2025), by ADMM.

    Any number of datasets and models (see :func:`_group_lasso_problem`):
    gravity and magnetics as in the paper, several datasets of one property,
    and MT / DC resistivity (nonlinear: Levenberg–Marquardt Gauss–Newton
    around ADMM), with an optional cross-gradient between the models
    (``gl_cross_gradient``).  lambda1 comes from a sweep down from
    lambda1_max (warm-started; one factorization for all points of the
    linear models) by the L-curve corner, falling back to chi^2 = N when
    the curve has no corner, or by chi^2 = N; or it is fixed.  Each sweep
    point is reported as an "iteration" with ``beta`` = lambda1, ``phi_d`` =
    chi^2 of all datasets and ``phi_m`` = the group penalty.  The user's stop
    request ends the sweep (lambda1 is then chosen from the points done) or
    the solve (its current iterate is kept).
    """
    from types import SimpleNamespace

    from ..methods.directives import IterationCollector, stop_requested
    from ..methods.l1l2_cda import _discrepancy_lambda
    from ..methods.regparam import LCURVE_NO_CORNER

    selection = task.gl_lambda1_selection
    if selection not in GROUP_LASSO_SELECTIONS:
        raise ValueError(f"Unknown gl_lambda1_selection '{selection}' "
                         f"(expected one of {GROUP_LASSO_SELECTIONS})")
    if mesh is None:
        mesh = _build_mesh(task)
    problem, setups, model_labels, dataset_labels = _group_lasso_problem(task, mesh)
    n_data = sum(len(s.survey.observed) for s in setups)
    weights0 = None
    if task.gl_data_weights is not None:   # start the balance from given weights
        w0 = task.gl_data_weights
        if isinstance(w0, dict):
            unknown = set(w0) - set(problem.model_names)
            if unknown:
                raise ValueError(f"gl_data_weights names unknown models {sorted(unknown)} "
                                 f"(models: {problem.model_names})")
            w0 = [float(w0.get(n, 1.0)) for n in problem.model_names]
        weights0 = [float(x) for x in w0]
        if len(weights0) != problem.n_models:
            raise ValueError(f"gl_data_weights has {len(weights0)} values for "
                             f"{problem.n_models} models {problem.model_names}")
        problem.scale_models(weights0)
        print("[Group lasso] data weights to start from: " + ", ".join(
            f"{n} {w:g}" for n, w in zip(problem.model_names, weights0)))
    lam_max = problem.lambda1_max()
    lam2 = float(task.gl_lambda2)
    lam3 = float(task.gl_cross_gradient)
    if task.gl_coupling not in ("group", "none"):
        raise ValueError(f"gl_coupling must be 'group' or 'none', got '{task.gl_coupling}'")
    solve_kw = dict(max_iter=int(task.gl_max_iter), tol_primal=float(task.gl_tol),
                    tol_dual=float(task.gl_tol), history_every=10, cross_gradient=lam3,
                    gn_max_iter=int(task.gl_gn_max_iter), gn_tol=float(task.gl_gn_tol),
                    coupling=task.gl_coupling)
    scales = {d.name: (float(sc) if np.ndim(sc) == 0 else "per datum")
              for d, sc in zip(problem.datasets, problem.scales)}
    print(f"[Group lasso] {len(problem.model_names)} models {problem.model_names}, "
          f"{problem.solver_name}, mu = {problem.mu:.4g}, lambda1_max = {lam_max:.4g}, "
          f"data scales {task.gl_data_scaling} {scales}"
          + (f", cross-gradient {lam3:g}" if lam3 else "")
          + ("" if problem.linear else ", Gauss–Newton")
          + f" (set up in {problem.setup_seconds:.1f} s)")

    def report(i, n, lam, chi2, extra=None):
        callback = IterationCollector.on_iteration
        if callback is not None:
            callback(SimpleNamespace(iteration=i, phi_d=chi2, beta=lam,
                                     extra={"max_iter": n, **(extra or {})}))

    warnings_ = list(problem.notes)
    lc = None
    lam_lcurve = lam_disc = None
    stopped_at = None
    if selection == "fixed":
        lam1 = float(task.gl_lambda1) if task.gl_lambda1 is not None \
            else float(task.gl_lambda1_ratio) * lam_max
        result = problem.solve(lam1, lam2, should_stop=stop_requested, **solve_kw)
        criterion = "fixed"
    elif selection == "search":
        # no sweep: a short descent for a warm start, then a secant search for chi^2 = N
        start = float(task.gl_lambda1_ratio) * lam_max
        state = None
        steps = [lam_max / 10.0, np.sqrt(lam_max / 10.0 * start)] if start < lam_max / 10.0 else []
        for i, lam in enumerate(steps):
            res = problem.solve(lam, lam2, state=state, should_stop=stop_requested, **solve_kw)
            state = res.state
            chi = problem.chi2_of(res)
            report(i + 1, len(steps) + 1, lam, chi["total"] if chi else None,
                   {"admm_iterations": res.n_iterations})
            print(f"[Group lasso] descent lambda1 = {lam:.4g}: chi2 = "
                  f"{chi['total'] if chi else float('nan'):.4g}, {res.n_iterations} ADMM iterations")
        result, lam1 = _discrepancy_solve(problem, start, lam2, state, n_data, solve_kw,
                                          should_stop=stop_requested, max_solves=8)
        criterion = "chi^2 = N (search)"
    else:
        def on_point(i, n, p):
            report(i + 1, n, p["lambda1"], p["chi2"], {"admm_iterations": p["n_iterations"]})
            print(f"[Group lasso] lambda1 {i + 1}/{n} = {p['lambda1']:.4g}: chi2 = {p['chi2']:.4g} "
                  f"(target {n_data}), group penalty {p['group_penalty']:.4g}, "
                  f"{p['n_active']} active cells, {p['n_iterations']} ADMM iterations"
                  + (f", {p['gn_iterations']} Gauss–Newton steps" if "gn_iterations" in p else "")
                  + ("" if p["converged"] else " (not converged)"))
        lc = problem.lcurve(lam2, n_lambda1=int(task.gl_n_lambda1),
                            decades=float(task.gl_lambda1_decades), point_callback=on_point,
                            should_stop=stop_requested, **solve_kw)
        if not lc.points:
            raise ValueError("Stopped before the first lambda1 point finished: nothing to keep")
        a = lc.arrays()
        chi2 = np.array([p["chi2"] for p in lc.points])
        lam_lcurve = lc.lambda1_corner if lc.corner and lc.corner["valid"] else None
        lam_disc = _discrepancy_lambda(a["lambda1"], chi2, n_data)
        if lc.stopped:
            stopped_at = len(lc.points)
            warnings_.append(f"Stopped by the user after {stopped_at} of {lc.n_planned} lambda1 "
                             "points; lambda1 is chosen from those")
        if selection == "lcurve" and lam_lcurve is None:
            warnings_.append(LCURVE_NO_CORNER + ("; lambda1 from chi^2 = N instead"
                                                 if lam_disc is not None else ""))
        lam1 = lam_lcurve if selection == "lcurve" else lam_disc
        criterion = selection
        if lam1 is None and selection == "lcurve" and lam_disc is not None:
            lam1, criterion = lam_disc, "discrepancy"
        if lam1 is None:
            # neither criterion on this sweep: the point whose chi^2 is closest to N
            k = int(np.argmin(np.abs(np.log(chi2 / n_data))))
            lam1, criterion = float(a["lambda1"][k]), "nearest chi^2 = N"
            warnings_.append(f"chi^2 = N is not reached on the lambda1 sweep "
                             f"(chi^2 from {chi2.max():.4g} to {chi2.min():.4g}); the point "
                             "closest to it is kept")
        if chi2.min() > n_data and not lc.stopped:
            warnings_.append(f"chi^2 stays above N = {n_data} down to the smallest lambda1 "
                             f"(min {chi2.min():.4g}): lambda2 = {lam2:g} may damp the model "
                             "too much for these data errors; try a smaller lambda2")
        if lc.stopped:   # no new solve: the finished point nearest lambda1
            k = int(np.argmin(np.abs(np.log(a["lambda1"] / lam1))))
            lam1 = float(a["lambda1"][k])
            result = problem.solve(lam1, lam2, state=lc.states[k], max_iter=0,
                                   cross_gradient=lam3, gn_max_iter=0)
        else:
            result = problem.solve_at(lc, lam1, lam2, should_stop=stop_requested, **solve_kw)

    balance = None
    if task.gl_balance:
        why = ("a fixed lambda1" if selection == "fixed" else
               "a stopped sweep" if lc is not None and lc.stopped else
               "nonlinear datasets" if not problem.linear else
               "one model" if problem.n_models < 2 else
               "datasets without errors" if any(d.std is None for d in problem.datasets)
               else None)
        if why:
            warnings_.append(f"gl_balance is not applied ({why}): it reweighs the datasets "
                             "of two or more linear models until each fits chi^2 = N")
        else:
            result, lam1, balance = _balance_group_lasso(
                problem, result, lam1, lam2, solve_kw, int(task.gl_balance_rounds),
                should_stop=stop_requested)
            criterion = "chi^2 = N for each model (balanced)"
            last = balance[-1]["chi2_per_datum"]
            total = sum(problem.chi2_of(result)[d.name] for d in problem.datasets)
            if abs(np.log(total / n_data)) <= 2 * DISCREPANCY_TOLERANCE:
                # the sweep alone did not reach chi^2 = N; the reweighted data did
                warnings_ = [w for w in warnings_ if not w.startswith(
                    ("chi^2 = N is not reached on the lambda1 sweep", "chi^2 stays above N"))]
            if any(abs(np.log(x)) > np.log(BALANCE_TOLERANCE) for x in last.values()):
                warnings_.append("The datasets are not balanced after "
                                 f"{task.gl_balance_rounds} rounds (chi^2 / N " + ", ".join(
                                     f"{n} {x:.3g}" for n, x in last.items()) + ")")
    chi2_final = problem.chi2_of(result)
    if result.stopped and stopped_at is None:
        stopped_at = result.n_iterations
    if not result.converged and not result.stopped and result.n_iterations:
        warnings_.append(f"ADMM did not converge in {task.gl_max_iter} iterations at lambda1 = "
                         f"{lam1:.4g}; raise gl_max_iter or try another mu"
                         if problem.linear else
                         f"Gauss–Newton did not converge at lambda1 = {lam1:.4g} "
                         f"({getattr(problem, 'gn_stop', '')}); raise gl_gn_max_iter")
    for w in warnings_:
        print(f"[Group lasso] Warning: {w}")

    if lc is not None:
        iterations = [{"iteration": i, "phi_d": p["chi2"], "phi_m": p["group_penalty"],
                       "phi_total": p["objective"], "beta": p["lambda1"],
                       "admm_iterations": p["n_iterations"], "n_active": p["n_active"]}
                      for i, p in enumerate(lc.points)]
    else:   # fixed lambda1: the ADMM iterations (every 10th), misfit in scaled units
        iterations = [{"iteration": 10 * (i + 1), "phi_d": 2.0 * m, "phi_m": gp,
                       "phi_total": o, "beta": lam1}
                      for i, (m, gp, o) in enumerate(zip(result.misfit_history,
                                                         result.group_penalty_history,
                                                         result.objective_history))]
    stopped = {"stopped_early": {"reason": "stopped by the user", "at_iteration": stopped_at,
                                 **({"of": lc.n_planned} if lc is not None and lc.stopped else {})}} \
        if stopped_at is not None else {}
    info = {
        "reference": "Utsugi (2025), Earth Planets Space 77:146",
        "criterion": criterion, "lambda1": result.lambda1, "lambda2": lam2, "mu": problem.mu,
        "lambda1_max": lam_max, "lambda1_lcurve": lam_lcurve, "lambda1_discrepancy": lam_disc,
        "data_scaling": task.gl_data_scaling, "data_scales": scales,
        "bounds": ({name: [None if v is None else float(v) if np.ndim(v) == 0 else "per cell"
                           for v in b]
                    for name, b in zip(problem.model_names, problem.bounds)
                    if b is not None and (b[0] is not None or b[1] is not None)}
                   if problem.bounds is not None else {}),
        "balance": balance, "data_weights": [float(w) for w in problem.data_weights],
        "data_weights_start": weights0, "admm_iterations_total": int(problem.admm_iterations_total),
        "gamma": task.gl_gamma, "weighting": task.gl_weighting,
        "relaxation": problem.relaxation, "solver": problem.solver_name,
        "admm_iterations": result.n_iterations, "admm_converged": result.converged,
        "n_active_cells": result.n_active, "chi2": chi2_final,
        "primal_residual": (result.primal_residual_history or [None])[-1],
        "dual_residual": (result.dual_residual_history or [None])[-1],
        "models": {name: [dataset_labels[i] for i, m in enumerate(model_labels) if m == name]
                   for name in problem.model_names},
        "warnings": warnings_,
    }
    if "gravity" in scales:
        info["gravity_scale"] = scales["gravity"]
    if lam3:
        info["cross_gradient"] = lam3
        info["cross_gradient_value"] = result.final_terms.get("cross_gradient")
    if not problem.linear:
        info["gauss_newton"] = [dict(r) for r in result.gauss_newton]
        info["gauss_newton_stop"] = getattr(problem, "gn_stop", None)
    if lc is not None:
        info["sweep"] = [dict(p) for p in lc.points]
    joint_data = {}
    for label, setup, pred in zip(dataset_labels, setups, result.predicted):
        # SimPEG sign convention; the pipeline turns them back into the files' convention
        joint_data[label] = {"locations": setup.survey.locations,
                             "observed": setup.survey.observed, "std": setup.survey.std,
                             "predicted": pred}
    return {
        **stopped,
        "task_id": task.task_id,
        "methods": [canonical_method(m) for m in task.joint_methods],
        "regularization": "group_lasso_ADMM",
        "converged": bool(result.converged and stopped_at is None),
        "n_iterations": len(iterations),
        "iterations": iterations,
        "recovered_models": {name: m for name, m in zip(problem.model_names,
                                                        result.models_physical)},
        "dataset_models": list(model_labels),
        "dataset_labels": list(dataset_labels),
        "joint_data": joint_data,
        "group_lasso": info,
        "coupling": _group_lasso_coupling(task, problem.model_names),
    }


def _group_lasso_coupling(task: InversionTask, model_names) -> dict:
    """The coupling record of a group-lasso run; with a cross-gradient it is the hybrid."""
    from ..methods.coupling import COUPLINGS, coupling_label
    hybrid = task.gl_cross_gradient > 0
    key = "group_lasso+cross_gradient" if hybrid else "group_lasso"
    if task.gl_coupling == "none":   # the control: the same solver, no coupling
        key = "group_lasso_uncoupled"
    return {"kind": key, "label": coupling_label(key), "family": "sparsity",
            "reference": COUPLINGS["group_lasso"].reference
            + ("; the cross-gradient (λ3) is our extension" if hybrid else ""),
            "models": list(model_names), "lambda2": task.gl_lambda2,
            **({"lambda3": task.gl_cross_gradient} if hybrid else {})}


def execute_task(task: InversionTask, mesh=None) -> dict:
    """Dispatch to single or joint inversion based on task spec.

    Args:
        mesh: Optional prebuilt mesh (Mesh3D or DiscretizeMesh); by default
            it is built from the task's mesh fields.
    """
    if task.joint_methods:
        if task.coupling == "group_lasso":
            return run_group_lasso_joint(task, mesh)
        return run_joint_inversion(task, mesh)
    if task.regularization_type == "group_lasso":
        raise ValueError("The group lasso couples several models (e.g. a gravity and a "
                         "magnetic dataset): it needs a joint inversion")
    return run_single_inversion(task, mesh)


# ── Raw-data pipeline ───────────────────────────────────────────────────

_GRID_FORMATS = ("geotiff", "surfer_grd", "esri_ascii")
_DATA_EXTENSIONS = (".tif", ".tiff", ".grd", ".asc", ".csv", ".txt", ".dat", ".xyz",
                    ".obs", ".npy", ".npz")
# Methods whose surveys can be built from station/grid files alone
_PIPELINE_METHODS = ("gravity", "magnetics")
# Methods whose data come in one .npz file with their geometry (see _load_em_dataset)
_EM_METHODS = ("dc_resistivity", "mt")
_DEFAULT_COMPONENT = {"gravity": "gz", "magnetics": "tmi"}
# Keys the upload page only sends in manual mode; auto mode ignores them.
# (Iteration limits are editable in both modes.)
_MANUAL_KEYS = (
    "norms", "alpha_s", "alpha_x", "alpha_y", "alpha_z", "beta0_ratio",
    "cooling_factor", "use_preconditioner",
    "bounds_lower", "bounds_upper", "joint_weights", "cross_gradient_weight", "coupling_weight",
    "l1_ratio", "beta_selection", "beta_sweep",
    "l1l2_solver", "l1l2_weighting", "lambda_decades", "lambda_step",
    "focusing_percentile", "focusing_scale",
    "depth_weighting", "depth_weighting_exponent",
    "bayes_samples", "bayes_seed", "bayes_threshold", "bayes_beta", "bayes_prior",
    *GROUP_LASSO_KEYS,
)
# Warn when more than this share of the recovered anomaly lies in padding cells
PADDING_WARNING_SHARE = 0.5
# The iterations kept in progress.json for the page's live curves (the last ones)
HISTORY_POINTS = 500
# A run fits its data when phi_d is within this share of the target (SimPEG's UpdateIRLS
# ends the IRLS within the same tolerance)
MISFIT_TOLERANCE = 0.1
# Results whose beta follows the discrepancy principle (chi^2 = N), by their label (joint
# runs add "joint_", MVI "mvi_"): the others (the L1–L2 coordinate-descent path, the
# group lasso, PGI) judge themselves
_DISCREPANCY_LABELS = ("smooth_L2", "sparse_IRLS", "elastic_net_IRLS", "focusing_MGS",
                       "total_variation", "L2", "mixed")


def _follows_discrepancy(label) -> bool:
    label = str(label or "")
    for prefix in ("joint_", "mvi_"):
        if label.startswith(prefix):
            label = label[len(prefix):]
    return label in _DISCREPANCY_LABELS


def assess_convergence(result: dict, n_data: int, max_iter) -> dict | None:
    """Did a discrepancy-principle run fit its data (chi^2 = N) before its iteration limit?

    Sets ``result["converged"]`` and returns what was found: the final and target
    phi_d, whether the limit of ``max_iter`` iterations ended the run, and advice
    when it did not converge.  Returns None (the flag is left as it is) for runs
    judged otherwise: beta chosen by L-curve or GCV, the L1–L2 coordinate-descent
    path, the group lasso, PGI, and runs without iterations.
    """
    iterations = result.get("iterations") or []
    if (not _follows_discrepancy(result.get("regularization")) or "beta_selection" in result
            or not iterations or not n_data):
        return None
    phi_d = iterations[-1].get("phi_d")
    if phi_d is None or not np.isfinite(phi_d):
        return None
    target = float(n_data)
    ratio = float(phi_d) / target
    at_limit = bool(max_iter) and len(iterations) >= int(max_iter)
    fits = ratio <= 1.0 + MISFIT_TOLERANCE
    info = {"phi_d": float(phi_d), "target": target, "chi2_per_datum": ratio,
            "tolerance": MISFIT_TOLERANCE, "max_iter": int(max_iter) if max_iter else None,
            "n_iterations": len(iterations), "at_iteration_limit": at_limit, "advice": []}
    if result.get("stopped_early"):
        info.update(status="stopped", reason="stopped by the user before it converged")
        result["converged"] = False
        return info
    if fits and not at_limit:
        info.update(status="converged", reason=f"χ²/N = {ratio:.2f} reached the target "
                    f"(1 ± {MISFIT_TOLERANCE:g}) after {len(iterations)} iterations")
        result["converged"] = True
        return info
    result["converged"] = False
    if fits:
        info.update(status="at_limit", reason=f"χ²/N = {ratio:.2f} reached the target, but the "
                    f"run ended at its limit of {max_iter} iterations: the model may still "
                    f"have been changing")
        info["advice"].append(f"Run again with more iterations (max_iter {2 * int(max_iter)})")
    elif at_limit:
        info.update(status="not_converged", reason=f"χ²/N = {ratio:.2f} is still above the "
                    f"target (1 ± {MISFIT_TOLERANCE:g}) after the limit of {max_iter} iterations")
        info["advice"].append(f"Run again with more iterations (max_iter {2 * int(max_iter)})")
        info["advice"].append("If the misfit has levelled off, the data errors may be too "
                              "small: raise the noise floor")
    else:
        info.update(status="not_converged", reason=f"χ²/N = {ratio:.2f} is still above the "
                    f"target (1 ± {MISFIT_TOLERANCE:g}): the IRLS ended after "
                    f"{len(iterations)} iterations without reaching it")
        info["advice"].append("Raise max_irls_iterations, or raise the noise floor if the "
                              "data errors are too small")
    return info
# The group lasso in auto mode: the paper's settings (sensitivity weighting, amplitude
# balance, L-curve) put most of the Karnataka models into the padding and fitted gravity to
# 0.4 N and magnetics to 2.6 N; weighing the cells like the other inversions (volume x depth
# weight, beta = 1), each datum by its error and balancing the datasets to chi^2 = N each
# kept them in the core (12 % and 33 % outside) and fitted both (0.96 and 1.02 N; 2 km mesh)
GROUP_LASSO_AUTO = {"gl_weighting": "depth", "depth_weighting_exponent": 1.0,
                    "gl_data_scaling": "std", "gl_lambda1_selection": "discrepancy",
                    "gl_balance": True}

# params key -> InversionTask field for the regularization/optimizer settings
_REG_PARAM_KEYS = (
    "max_iter", "beta0_ratio", "cooling_factor",
    "alpha_s", "alpha_x", "alpha_y", "alpha_z",
    "irls_cooling_factor", "max_irls_iterations", "use_preconditioner",
    "bounds_lower", "bounds_upper", "l1_ratio", "beta_selection", "beta_sweep",
    "l1l2_solver", "l1l2_weighting", "lambda_decades", "lambda_step",
    "focusing_percentile", "focusing_scale",
    "depth_weighting", "depth_weighting_exponent",
    "bayes_samples", "bayes_seed", "bayes_threshold", "bayes_beta", "bayes_prior",
    *GROUP_LASSO_KEYS,
)


# Automatic errors (noise_floor "auto", noise_pct "auto"): a floor of 1.5 % of the 5-95 %
# spread of the data inverted, and 2 % of each datum.  Errors in proportion to |d| alone give
# the zero crossings of the anomalies a few nT, which no model fits without near-surface
# speckle: on the Block-8 window (5 % + 2 nT) a cross-validation fitted the nodes it saw to
# χ²/N 0.54 and those it did not to 2.38; with 2 % + 90 nT (1.5 % of the spread) both 0.89.
AUTO_FLOOR_SHARE = 0.015
AUTO_NOISE_PCT = 0.02


def auto_noise(values, noise_pct, noise_floor) -> dict | None:
    """The errors of a dataset whose noise_floor or noise_pct is "auto" (else None):
    {"noise_pct", "noise_floor", "spread", "floor_share"}; a given number is kept."""
    is_auto = lambda v: isinstance(v, str) and v.strip().lower() == "auto"
    if not (is_auto(noise_floor) or is_auto(noise_pct)):
        return None
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    p5, p95 = np.percentile(v, [5, 95]) if v.size else (0.0, 0.0)
    spread = float(p95 - p5)
    floor = AUTO_FLOOR_SHARE * spread if is_auto(noise_floor) else float(noise_floor)
    if is_auto(noise_floor) and not floor > 0:   # constant data: a floor from their size
        floor = max(AUTO_FLOOR_SHARE * float(np.max(np.abs(v))) if v.size else 0.0, 1e-12)
    return {"noise_pct": AUTO_NOISE_PCT if is_auto(noise_pct) else float(noise_pct),
            "noise_floor": float(floor), "spread": spread, "floor_share": AUTO_FLOOR_SHARE}


@dataclass
class PipelineDataset:
    """One observed dataset after loading, ready for inversion."""
    method: str               # canonical method name
    component: str
    locations: np.ndarray     # (n, 3); z is NaN until stations are placed
    observed: np.ndarray
    std: np.ndarray
    method_kwargs: dict
    files: list
    noise_pct: float
    noise_floor: float
    noise_auto: dict | None = None    # the errors were chosen from the data (see auto_noise)
    spacing: float | None = None      # data spacing (m) of the finest file
    spacing_kind: str = ""            # "grid" or "points"
    regional: dict | None = None      # summary of the removed regional field
    trend: np.ndarray | None = None   # the removed regional field at the stations
    rtp: dict | None = None           # the data were reduced to the pole (see _reduce_to_pole)
    continuation: dict | None = None  # the data were continued upwards first (see _full_grid_filters)
    igrf: dict | None = None          # the inducing field was computed from IGRF-14 (see _igrf_field)
    bouguer_check: dict | None = None  # per station table: reduction density, terrain correction
    # observed (file convention) x sign = SimPEG convention; see _simpeg_sign
    sign: float = 1.0
    decimation: dict | None = None    # how the data were thinned (target spacing, per file)
    model: str | None = None          # joint: model label (datasets with one label share a model)
    regularization: dict | None = None  # joint, manual mode: overrides for its model (JOINT_REG_KEYS)
    # MT / DC: the points the mesh must cover (stations, electrodes) and one (x, y, z)
    # per datum for plotting (DC: the electrodes' centre; MT: the station)
    extent_points: np.ndarray | None = None
    plot_locations: np.ndarray | None = None

    @property
    def mesh_points(self) -> np.ndarray:
        return self.locations if self.extent_points is None else self.extent_points

    @property
    def data_locations(self) -> np.ndarray:
        return self.locations if self.plot_locations is None else self.plot_locations


def grid_strides(dx: float, dy: float, target: float | None, stride: int = 1) -> tuple:
    """Row/column strides that thin a grid to about ``target`` metres.

    Without a target, ``stride`` applies to both axes (the older
    ``decimate_stride`` parameter).
    """
    if not target:
        return max(int(stride), 1), max(int(stride), 1)
    sx = max(int(round(target / dx)), 1) if dx > 0 else 1
    sy = max(int(round(target / dy)), 1) if dy > 0 else 1
    return sx, sy


def thin_points(xy: np.ndarray, spacing: float, origin=None) -> np.ndarray:
    """Indices keeping the first point in each ``spacing`` x ``spacing`` cell.

    Cells are counted from ``origin`` (default: the south-west corner of the
    points), so the result is deterministic and matches the upload page's
    estimate.
    """
    xy = np.asarray(xy, dtype=float)
    if not spacing or spacing <= 0 or len(xy) == 0:
        return np.arange(len(xy))
    x0, y0 = origin if origin is not None else (xy[:, 0].min(), xy[:, 1].min())
    ix = np.floor((xy[:, 0] - x0) / spacing).astype(np.int64)
    iy = np.floor((xy[:, 1] - y0) / spacing).astype(np.int64)
    key = ix * (int(iy.max()) + 1) + iy if len(iy) else ix
    _, first = np.unique(key, return_index=True)
    return np.sort(first)


def _read_observations(path: str, component: str, stride: int, aoi,
                       target_spacing: float | None = None, crs: str | None = None) -> tuple:
    """Read one data file -> (locations (n, 3), values (n,), metadata).

    Gridded files give stations at the grid nodes with unknown elevation
    (z = NaN); point files keep their z if present.  Point files in
    longitude/latitude are projected to ``crs``, the job's working CRS (see
    :func:`_job_crs`), before anything else.  ``aoi`` = [west, east, south,
    north] (m) crops both.  ``target_spacing`` (m) thins grids by whole
    strides per axis and points to one per cell (see :func:`thin_points`);
    without it, grids use ``stride``.  ``metadata["decimation"]`` records
    what was done.
    """
    from ..io.crs import GEOGRAPHIC, looks_geographic, project
    from ..io.readers import GridData, detect_format, read_auto, read_station_table

    name = os.path.basename(path)
    fmt = detect_format(path)
    if fmt in _GRID_FORMATS:
        grid = read_auto(path)
        if crs and grid.crs and grid.crs != crs:
            raise ValueError(f"'{name}' is in {grid.crs} but the job works in {crs}: "
                             "reproject the grid, or give the job's 'crs'")
        x, y, values = grid.x, grid.y, grid.values
        if aoi:
            west, east, south, north = aoi
            mask_x = (x >= west) & (x <= east)
            mask_y = (y >= south) & (y <= north)
            values = values[np.ix_(mask_y, mask_x)]
            x, y = x[mask_x], y[mask_y]
        dx = abs(float(x[1] - x[0])) if len(x) > 1 else 0.0
        dy = abs(float(y[1] - y[0])) if len(y) > 1 else 0.0
        sx, sy = grid_strides(dx, dy, target_spacing, stride)
        x, y, values = x[::sx], y[::sy], values[::sy, ::sx]
        xx, yy = np.meshgrid(x, y)
        locs = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, np.nan)])
        steps = [abs(float(a[1] - a[0])) for a in (x, y) if len(a) > 1]
        meta = dict(grid.metadata)
        if steps:
            meta["grid_spacing"] = max(steps)
        if sx > 1 or sy > 1:
            meta["decimation"] = {"kind": "grid", "stride": [sx, sy],
                                  "native_spacing_m": [dx, dy]}
        return locs, values.ravel(), meta

    if fmt == "csv":
        points = read_station_table(path, value_name=component)
    elif fmt in ("ubc_obs", "npz", "npy"):
        points = read_auto(path)
        if isinstance(points, GridData):
            raise ValueError(
                f"'{name}' is a bare array without coordinates; save an (N, 4) array "
                f"of x, y, z, value instead"
            )
    else:
        raise ValueError(
            f"Unsupported data file '{name}'. Use a grid (.tif/.grd/.asc) or station "
            f"data (.csv/.xyz/.txt/.dat/.obs/.npy/.npz)"
        )
    if points.values is None:
        raise ValueError(f"'{name}' has coordinates but no data column")

    locs = np.array(points.locations, dtype=np.float64)
    if locs.shape[1] == 2:
        locs = np.column_stack([locs, np.full(len(locs), np.nan)])
    values = np.asarray(points.values, dtype=np.float64)
    projected = None
    if looks_geographic(locs[:, 0], locs[:, 1]):
        if not crs:
            raise ValueError(f"'{name}' is in longitude/latitude; give the job's 'crs'")
        locs[:, 0], locs[:, 1] = project(locs[:, 0], locs[:, 1], GEOGRAPHIC, crs)
        projected = {"from": GEOGRAPHIC, "to": crs}
        print(f"[Pipeline] {name}: projected from longitude/latitude to {crs}")
    if aoi:
        west, east, south, north = aoi
        keep = ((locs[:, 0] >= west) & (locs[:, 0] <= east)
                & (locs[:, 1] >= south) & (locs[:, 1] <= north))
        locs, values = locs[keep], values[keep]
    meta = dict(points.metadata)
    if projected:
        meta["projected"] = projected
    if target_spacing:
        n_before = len(locs)
        origin = (aoi[0], aoi[2]) if aoi else None
        keep = thin_points(locs[:, :2], float(target_spacing), origin)
        locs, values = locs[keep], values[keep]
        meta["decimation"] = {"kind": "points", "cell_m": float(target_spacing),
                              "n_before": int(n_before)}
    return locs, values, meta


def _simpeg_sign(method: str, component: str, spec: dict, params: dict) -> float:
    """-1 for gravity gz given positive downward, as field data are.

    Bouguer and free-air anomalies are positive over excess mass (gravity
    positive downward); SimPEG's gz is the z-up component, negative over
    excess mass, so such data are negated before inverting (and the observed
    and predicted data are reported back in the file's convention).  Set
    ``gz_convention: "simpeg"`` for data in SimPEG's convention (e.g. made
    with SimPEG's forward modelling).  gzz is the same in both conventions.
    """
    if method != "gravity" or component != "gz":
        return 1.0
    convention = spec.get("gz_convention", params.get("gz_convention", "positive_down"))
    if convention not in ("positive_down", "simpeg"):
        raise ValueError(f"Unknown gz_convention '{convention}' "
                         "(expected 'positive_down' or 'simpeg')")
    return -1.0 if convention == "positive_down" else 1.0


def _file_spacing(locs, values, meta) -> tuple:
    """(spacing in m, "grid" | "points") of one data file; (None, "") if unknown."""
    from .meshing import points_spacing

    if meta.get("grid_spacing"):
        return float(meta["grid_spacing"]), "grid"
    valid = np.isfinite(values) & np.isfinite(locs[:, :2]).all(axis=1)
    try:
        return points_spacing(locs[valid, :2]), "points"
    except ValueError:
        return None, ""


# Below this |inclination| the reduction to the pole is damped (methods/enhance.rtp) and
# distorts the anomalies; the upload page refuses it there
RTP_MIN_INCLINATION = 30.0


# When a survey's date is not known, the field of this date is used (the latest definitive
# IGRF epoch); the summary says so, and the page shows how much 2000-2025 would differ
DEFAULT_IGRF_DATE = "2020-01-01"


def _igrf_field(igrf: dict, locs, crs, spec: dict, params: dict):
    """[F, I, D from grid north] from IGRF-14 at the median of the stations, ``igrf["date"]``
    (default DEFAULT_IGRF_DATE), the stations' height above the ground (the ground itself is
    not known yet: a few hundred metres change F by a few nT), and a summary."""
    from ..methods.igrf import inducing_field
    if not crs:
        raise ValueError("The inducing field from IGRF needs the data's coordinate system: give "
                         "the job's crs (e.g. EPSG:32643) or the field itself")
    date = igrf.get("date") or None
    height = float(spec.get("continue_to_m") or spec.get("station_height", params.get("station_height", 0.0)) or 0.0)
    x, y = float(np.median(locs[:, 0])), float(np.median(locs[:, 1]))
    f = inducing_field(x, y, crs, alt_m=height, date=date or DEFAULT_IGRF_DATE)
    info = {"model": "IGRF-14", "date": date or DEFAULT_IGRF_DATE, "date_known": bool(date),
            "x": x, "y": y, "lon": f["lon"], "lat": f["lat"], "F": float(f["F"]), "I": float(f["I"]),
            "D_true": float(f["D"]), "convergence": float(f["convergence"]),
            "D_grid": float(f["D_grid"])}
    return tuple(float(v) for v in f["inducing_field"]), info


def _continuation_distance(spec: dict, params: dict) -> float:
    """How far (m) a dataset is continued upwards: from its height above the ground
    (``station_height``, e.g. a flight height) to ``continue_to_m``; 0 for none."""
    to = spec.get("continue_to_m", params.get("continue_to_m"))
    if not to:
        return 0.0
    height = float(spec.get("station_height", params.get("station_height", 0.0)))
    by = float(to) - height
    if by <= 0:
        raise ValueError(f"continue_to_m ({float(to):g} m) must be above the data's height above "
                         f"the ground ({height:g} m)")
    return by


# Above this many nodes the full-resolution grid is averaged down before the filters
FULL_GRID_MAX_NODES = 4_000_000


def grid_window(path: str, crs=None, aoi=None):
    """A grid file as (x, y, values) with x and y ascending (rows south to north), cropped
    to ``aoi``; nodata is NaN.  For filters on whole grids: no point arrays are made."""
    from ..io.readers import read_auto
    grid = read_auto(path)
    if crs and grid.crs and grid.crs != crs:
        raise ValueError(f"'{os.path.basename(path)}' is in {grid.crs} but the job works in {crs}")
    x, y, v = np.asarray(grid.x, dtype=float), np.asarray(grid.y, dtype=float), np.asarray(grid.values)
    if x[0] > x[-1]:
        x, v = x[::-1], v[:, ::-1]
    if y[0] > y[-1]:
        y, v = y[::-1], v[::-1]
    if aoi is not None:
        cx, cy = (x >= aoi[0]) & (x <= aoi[1]), (y >= aoi[2]) & (y <= aoi[3])
        x, y, v = x[cx], y[cy], v[np.ix_(cy, cx)]
    if x.size < 3 or y.size < 3:
        raise ValueError(f"'{os.path.basename(path)}' has too few nodes in the window")
    return x, y, v.astype(float, copy=False)


def _full_grid_filters(files, data_dir, method, component, crs, locs, kwargs, continue_by=0.0,
                       rtp=False, aoi=None, thin_m=None):
    """The data continued ``continue_by`` metres upwards and/or reduced to the pole, at the
    job's stations: (values, method kwargs, rtp summary, continuation summary).

    Each file is read at full resolution (no thinning), within the window widened by a
    margin (10 km, and ten times the continuation), so the window's edges and the thinning
    do not cut into the filters; the result is sampled at the stations the job keeps.  A
    survey flown at a constant height above the ground is treated as on a level plane, as
    usual.  The reduction to the pole assumes induced magnetization (see _reduce_to_pole).
    """
    from ..io.readers import detect_format
    from ..methods.enhance import coarsen, rtp as reduce, sample, to_grid, upward
    strength = inc = dec = None
    if rtp:
        if method != "magnetics" or component != "tmi":
            raise ValueError("The reduction to the pole applies to total-field magnetic data (tmi)")
        if kwargs.get("magnetization") == "vector":
            raise ValueError("The magnetization-vector inversion models the field as measured: "
                             "invert the total field, not its reduction to the pole")
        field = kwargs.get("inducing_field")
        if not field or len(field) != 3:
            raise ValueError("The reduction to the pole needs the inducing field [F, I, D]")
        strength, inc, dec = (float(v) for v in field)
    wide = None
    if aoi is not None:
        m = max(10000.0, 10.0 * continue_by)
        wide = [aoi[0] - m, aoi[1] + m, aoi[2] - m, aoi[3] + m]
    if len(files) == 1 and detect_format(os.path.join(data_dir, files[0])) in _GRID_FORMATS:
        # a grid file stays a grid (no point arrays: a survey grid has tens of millions of nodes)
        xg, yg, grid = grid_window(os.path.join(data_dir, files[0]), crs, wide)
    else:
        xs, ys, vs = [], [], []
        for fname in files:
            l, v, _ = _read_observations(os.path.join(data_dir, fname), component, 1, wide, None, crs)
            ok = np.isfinite(v) & np.isfinite(l[:, :2]).all(axis=1)
            xs.append(l[ok, 0]); ys.append(l[ok, 1]); vs.append(v[ok])
        xg, yg, grid = to_grid(np.concatenate(xs), np.concatenate(ys), np.concatenate(vs))
    shape = [int(grid.shape[0]), int(grid.shape[1])]
    # a survey grid of tens of millions of nodes (a whole aeromagnetic block) is averaged
    # down first, by blocks no wider than half the spacing the job thins to (or a quarter of
    # the continuation): what the filters would remove or the thinning drop anyway
    averaged = 1
    if grid.size > FULL_GRID_MAX_NODES:
        d0 = float(xg[1] - xg[0])
        limit = max(d0, 0.5 * float(thin_m or 0.0), 0.25 * float(continue_by or 0.0))
        averaged = max(1, min(int(np.ceil(np.sqrt(grid.size / FULL_GRID_MAX_NODES))), int(limit // d0)))
        if averaged > 1:
            xg, yg, grid = coarsen(xg, yg, grid, averaged)
    dx, dy = float(xg[1] - xg[0]), float(yg[1] - yg[0])
    cont_info = rtp_info = None
    if continue_by:
        grid = upward(grid, dx, dy, continue_by)
        cont_info = {"by_m": float(continue_by), "grid_spacing_m": dx, "grid_shape": shape,
                     **({"averaged": averaged} if averaged > 1 else {})}
    if rtp:
        damping = max(0.0, np.sin(np.radians(RTP_MIN_INCLINATION)) ** 2 - np.sin(np.radians(abs(inc))) ** 2)
        grid = reduce(grid, dx, dy, inc, dec, damping=damping)
        kwargs = {**kwargs, "inducing_field": (strength, 90.0, 0.0)}
        rtp_info = {"field_inclination": inc, "field_declination": dec, "damping": float(damping),
                    "grid_spacing_m": dx, "grid_shape": shape}
    return sample(xg, yg, grid, locs[:, 0], locs[:, 1]), kwargs, rtp_info, cont_info


def _reduce_to_pole(files, data_dir, method, component, crs, locs, kwargs, aoi=None):
    """Total-field data reduced to the pole at the job's stations, the method's kwargs for
    a vertical field (magnetization along it), and a summary.

    Induced magnetization is assumed: remanence in another direction would be wrongly
    reduced, so the magnetization-vector inversion takes the measured field instead.
    """
    values, kwargs, info, _ = _full_grid_filters(files, data_dir, method, component, crs, locs,
                                                 kwargs, rtp=True, aoi=aoi)
    return values, kwargs, info


def _dataset_specs(params: dict, data_dir: str, exclude: set) -> list:
    """The `datasets` list from params, or one synthesised from legacy keys."""
    specs = params.get("datasets")
    if specs:
        return specs

    method = params.get("method_type", "gravity")
    if method == "joint":
        raise ValueError("Joint jobs need a 'datasets' list describing each data type")
    data_file = params.get("data_file", "")
    if data_file and os.path.exists(os.path.join(data_dir, data_file)):
        files = [data_file]
    else:
        candidates = sorted(
            f for f in os.listdir(data_dir)
            if f.lower().endswith(_DATA_EXTENSIONS) and f not in exclude
        )
        if not candidates:
            raise FileNotFoundError(f"No recognized data files in {data_dir}")
        files = candidates[:1]
    return [{
        "method": method,
        "files": files,
        "noise_pct": params.get("noise_pct", 0.05),
        "noise_floor": params.get("noise_floor", 0.5),
        "method_kwargs": params.get("method_kwargs", {}),
    }]


def _job_crs(params: dict, data_dir: str, specs: list, topo_file: str | None) -> str | None:
    """The job's working CRS: ``params["crs"]``, else that of its first georeferenced
    grid (data or DEM), else the UTM zone of its longitude/latitude station tables,
    else None (all coordinates already in metres).  See io/crs.py."""
    from ..io.crs import looks_geographic, working_crs
    from ..io.readers import detect_format, read_station_table

    names = [f for s in specs for f in (s.get("files") or ([s["file"]] if s.get("file") else []))]
    if topo_file:
        names.append(topo_file)
    grid_crs, lonlat, geographic_grids = [], [], []
    for fname in names:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        fmt = detect_format(path)
        try:
            if fmt == "geotiff":
                import rasterio
                with rasterio.open(path) as src:   # a geographic grid (e.g. SRTM) cannot be it
                    projected = src.crs is not None and not src.crs.is_geographic
                    grid_crs.append(str(src.crs) if projected else None)
                    if src.crs is not None and src.crs.is_geographic:
                        b = src.bounds
                        geographic_grids.append(((b.left + b.right) / 2, (b.bottom + b.top) / 2))
            elif fmt == "csv":
                pts = read_station_table(path)
                if looks_geographic(pts.locations[:, 0], pts.locations[:, 1]):
                    lonlat.append((pts.locations[:, 0], pts.locations[:, 1]))
        except Exception:   # unreadable here: the loaders report it properly
            continue
    crs = working_crs(grid_crs, lonlat, params.get("crs"))
    if crs is None and geographic_grids:
        # data in metres with no CRS, and a longitude/latitude DEM (e.g. SRTM): the data
        # lie on the DEM, so take the UTM zone of its centre
        from ..io.crs import utm_crs
        crs = utm_crs([geographic_grids[0][0]], [geographic_grids[0][1]])
        print(f"[Pipeline] Note: the DEM is in longitude/latitude and nothing else gives a "
              f"CRS; the data are taken to be in {crs} (set 'crs' if not)")
    if crs:
        print(f"[Pipeline] Working coordinate system: {crs}")
    return crs


def _load_dataset(spec: dict, params: dict, data_dir: str, single: bool,
                  crs: str | None = None) -> PipelineDataset:
    """Load all files of one dataset spec and attach its noise model."""
    method = canonical_method(spec.get("method") or spec.get("type")
                              or params.get("method_type", "gravity"))
    if method in _EM_METHODS:
        return _load_em_dataset(spec, params, data_dir, method)
    if method not in _PIPELINE_METHODS:
        raise NotImplementedError(
            f"The raw-data pipeline supports {', '.join(_PIPELINE_METHODS + _EM_METHODS)}; "
            f"got '{method}'"
        )

    files = list(spec.get("files") or ([spec["file"]] if spec.get("file") else []))
    if not files:
        raise ValueError(f"No files listed for the {method} dataset")
    missing = [f for f in files if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        raise FileNotFoundError(f"{method} data file(s) not found in {data_dir}: {missing}")

    kwargs = dict(spec.get("method_kwargs")
                  or (params.get("method_kwargs") if single else None) or {})
    component = spec.get("component") or kwargs.get("component") or _DEFAULT_COMPONENT[method]
    kwargs["component"] = component

    stride = int(params.get("decimate_stride", 1) or 1)
    aoi = params.get("aoi")
    if aoi is not None:
        aoi = [float(v) for v in aoi]
        if len(aoi) != 4 or not (aoi[0] < aoi[1] and aoi[2] < aoi[3]):
            raise ValueError(f"aoi must be [west, east, south, north] with west < east "
                             f"and south < north, got {params.get('aoi')}")
    # Thinning: per dataset ("decimate_spacing_m"), else the job-wide value
    target = spec.get("decimate_spacing_m", params.get("decimate_spacing_m"))
    target = float(target) if target else None
    if target is not None and target < 0:
        raise ValueError(f"decimate_spacing_m must be positive, got {target}")
    locs_all, values_all, spacings, thinning = [], [], [], {}
    for fname in files:
        path = os.path.join(data_dir, fname)
        print(f"[Pipeline] Loading {method} ({component}) data from {fname}")
        locs, values, meta = _read_observations(path, component, stride, aoi, target, crs)
        if meta.get("decimation"):
            thinning[fname] = meta["decimation"]
            print(f"[Pipeline] {fname}: thinned to {len(values)} data ({meta['decimation']})")
        if method == "magnetics" and "inducing_field" not in kwargs \
                and meta.get("inducing_field"):
            kwargs["inducing_field"] = tuple(meta["inducing_field"])
        locs_all.append(locs)
        values_all.append(values)
        spacings.append(_file_spacing(locs, values, meta))

    locs = np.vstack(locs_all)
    dobs = np.concatenate(values_all)
    valid = np.isfinite(dobs) & np.isfinite(locs[:, :2]).all(axis=1)
    locs, dobs = locs[valid], dobs[valid]
    if dobs.size == 0:
        raise ValueError(f"No valid {method} observations in {files}")

    # The inducing field from IGRF-14 at the centre of the job's stations, when it is not
    # given ("igrf": {"date": ...}; the date may be unknown: then DEFAULT_IGRF_DATE)
    igrf_info = None
    if method == "magnetics" and "inducing_field" not in kwargs and "igrf" in spec:
        kwargs["inducing_field"], igrf_info = _igrf_field(spec.get("igrf") or {}, locs, crs, spec, params)
        print(f"[Pipeline] {method}: inducing field from IGRF-14 for {igrf_info['date']}"
              f"{'' if igrf_info['date_known'] else ' (survey date unknown)'} at "
              f"{igrf_info['lon']:.3f}° E, {igrf_info['lat']:.3f}° N: F {igrf_info['F']:.1f} nT, "
              f"I {igrf_info['I']:.2f}°, D {igrf_info['D_grid']:.2f}° from grid north")

    # Continued upwards on the full-resolution data before thinning (thinning a low-flown grid
    # aliases it), and/or reduced to the pole (as e.g. Yang et al. 2026 invert them: then a
    # vertical field and magnetization); the regional field is taken from the result
    rtp_info = cont_info = None
    continue_by = _continuation_distance(spec, params)
    if spec.get("rtp") or continue_by:
        dobs, kwargs, rtp_info, cont_info = _full_grid_filters(
            files, data_dir, method, component, crs, locs, kwargs, continue_by=continue_by,
            rtp=bool(spec.get("rtp")), aoi=aoi, thin_m=target)
        if cont_info:
            locs = locs.copy()
            locs[:, 2] = locs[:, 2] + continue_by     # stations with an elevation go up with the data
            print(f"[Pipeline] {method}: continued {continue_by:g} m upwards on the "
                  f"{cont_info['grid_spacing_m']:g} m grid before thinning")
        if rtp_info:
            print(f"[Pipeline] {method}: reduced to the pole (field I {rtp_info['field_inclination']:g}°, "
                  f"D {rtp_info['field_declination']:g}°{', damped' if rtp_info['damping'] else ''}); "
                  f"the inversion models a vertical field")

    # Regional field (e.g. a polynomial trend surface) removed before inverting
    from ..methods.regional import remove_regional
    raw = dobs
    dobs, regional = remove_regional(locs[:, :2], dobs, spec.get("regional",
                                                                  params.get("regional")))
    if regional:
        print(f"[Pipeline] {method}: removed regional field ({regional['label']}): data std "
              f"{regional['data_std_before']:.4g} -> {regional['data_std_after']:.4g}")

    noise_pct = spec.get("noise_pct", params.get("noise_pct", 0.05))
    noise_floor = spec.get("noise_floor", params.get("noise_floor", 0.5))
    noise_auto = auto_noise(dobs, noise_pct, noise_floor)
    if noise_auto:
        noise_pct, noise_floor = noise_auto["noise_pct"], noise_auto["noise_floor"]
        print(f"[Pipeline] {method}: automatic errors: {100 * noise_pct:g} % + {noise_floor:.4g} "
              f"(floor {100 * AUTO_FLOOR_SHARE:g} % of the 5-95 % spread of the data, "
              f"{noise_auto['spread']:.4g})")
    noise_pct, noise_floor = float(noise_pct), float(noise_floor)
    std = noise_pct * np.abs(dobs) + noise_floor
    if np.any(std <= 0):
        raise ValueError(
            f"{method}: uncertainty is zero for {int(np.sum(std <= 0))} data; "
            f"set a noise floor > 0"
        )

    known = [sp for sp in spacings if sp[0]]
    spacing, spacing_kind = min(known) if known else (None, "")

    return PipelineDataset(
        method=method, component=component, locations=locs, observed=dobs, std=std,
        method_kwargs=kwargs, files=files, noise_pct=noise_pct, noise_floor=noise_floor,
        noise_auto=noise_auto, spacing=spacing, spacing_kind=spacing_kind, regional=regional,
        trend=(raw - dobs) if regional else None, rtp=rtp_info, continuation=cont_info, igrf=igrf_info,
        sign=_simpeg_sign(method, component, spec, params),
        decimation={"target_spacing_m": target, "files": thinning} if thinning else None,
        model=spec.get("model") or None,
        regularization=dict(spec["regularization"]) if spec.get("regularization") else None,
        bouguer_check=_bouguer_checks(files, data_dir) if method == "gravity" else None,
    )


def _bouguer_checks(files, data_dir) -> dict | None:
    """Reduction density and terrain correction of each gravity station table that has
    the columns for it (see methods/bouguer.py); recorded with the run."""
    from ..methods.bouguer import check_table

    out = {}
    for fname in files:
        if not fname.lower().endswith((".csv", ".txt", ".xyz", ".dat")):
            continue
        try:
            r = check_table(os.path.join(data_dir, fname))
        except Exception as e:   # a diagnostic only: never fail the job for it
            r = {"verdict": "unclear", "message": f"check failed: {e}"}
        if r is not None:
            out[fname] = r
            print(f"[Pipeline] {fname}: {r['message']}")
    return out or None


def _nearest_spacing(points: np.ndarray) -> float | None:
    """Median distance from each distinct (x, y) point to its nearest neighbour."""
    from scipy.spatial import cKDTree
    xy = np.unique(np.round(np.asarray(points, dtype=float)[:, :2], 6), axis=0)
    if len(xy) < 2:
        return None
    dist, _ = cKDTree(xy).query(xy, k=2)
    return float(np.median(dist[:, 1]))


def _load_em_dataset(spec: dict, params: dict, data_dir: str, method: str) -> PipelineDataset:
    """An MT or DC resistivity dataset from one .npz file.

    DC: ``electrodes`` (n, 12: A, B, M, N as x, y, z; NaN B / N for poles) or
    ``a``, ``b``, ``m``, ``n`` (n, 3 each; b, n optional), ``values`` (n), and
    optionally ``std`` and ``data_type`` ("volt" or "apparent_resistivity").
    MT: ``locations`` (stations, 3), ``frequencies``, ``components`` (e.g.
    "xy_real"), ``values`` ordered frequency by frequency, then component by
    component, then station by station, and optionally ``std``.  Without
    ``std`` the spec's noise_pct and noise_floor (required: the data have no
    natural floor) give it.  Elevations must be given (z of every point);
    ``method_kwargs`` (e.g. sigma_background) go to the method, whose
    background is the reference model.
    """
    from ..methods.dc_resistivity import electrode_rows

    files = list(spec.get("files") or ([spec["file"]] if spec.get("file") else []))
    if len(files) != 1 or not files[0].lower().endswith(".npz"):
        raise ValueError(f"{method} data come in one .npz file with their geometry; got {files}")
    path = os.path.join(data_dir, files[0])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{method} data file(s) not found in {data_dir}: {files}")
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if "values" not in d:
        raise ValueError(f"{files[0]}: no 'values' array")
    values = np.asarray(d["values"], dtype=float).ravel()
    kwargs = dict(spec.get("method_kwargs") or {})
    if method == "dc_resistivity":
        if "electrodes" in d:
            rows = np.asarray(d["electrodes"], dtype=float)
        else:
            rows = electrode_rows(d["a"], d.get("b"), d["m"], d.get("n"))
        if "data_type" in d:
            kwargs.setdefault("data_type", str(d["data_type"]))
        if rows.shape != (values.size, 12):
            raise ValueError(f"{files[0]}: electrodes have shape {rows.shape}, expected "
                             f"({values.size}, 12)")
        points = rows.reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        centres = np.nanmean(rows.reshape(len(rows), 4, 3), axis=1)
        locations, plot = rows, centres
        component = kwargs.get("data_type", "volt")
    else:
        stations = np.atleast_2d(np.asarray(d["locations"], dtype=float))
        freqs = [float(f) for f in np.ravel(d["frequencies"])]
        comps = [str(c) for c in np.ravel(d["components"])]
        kwargs.setdefault("frequencies", freqs)
        kwargs.setdefault("components", comps)
        n = len(kwargs["frequencies"]) * len(kwargs["components"]) * len(stations)
        if values.size != n:
            raise ValueError(f"{files[0]}: {values.size} values for {len(kwargs['frequencies'])} "
                             f"frequencies x {len(kwargs['components'])} components x "
                             f"{len(stations)} stations = {n}")
        points = stations
        locations = stations
        plot = np.tile(stations, (n // len(stations), 1))
        component = "impedance"
    if not np.isfinite(points[:, 2]).all():
        raise ValueError(f"{method}: every station / electrode needs an elevation (z)")
    if not np.isfinite(values).all():
        raise ValueError(f"{method}: the data contain NaN or inf")
    noise_pct = float(spec.get("noise_pct", 0.05))
    if "std" in d:
        std = np.broadcast_to(np.asarray(d["std"], dtype=float), values.shape).copy()
        noise_floor = float(spec.get("noise_floor", 0.0))
    else:
        if "noise_floor" not in spec:
            raise ValueError(f"{method}: give 'std' in the file or a noise_floor in the dataset")
        noise_floor = float(spec["noise_floor"])
        std = noise_pct * np.abs(values) + noise_floor
    if np.any(std <= 0):
        raise ValueError(f"{method}: uncertainty is zero for {int(np.sum(std <= 0))} data")
    print(f"[Pipeline] Loading {method} data from {files[0]}: {values.size} data")
    return PipelineDataset(
        method=method, component=component, locations=locations, observed=values, std=std,
        method_kwargs=kwargs, files=files, noise_pct=noise_pct, noise_floor=noise_floor,
        spacing=_nearest_spacing(points), spacing_kind="points",
        model=spec.get("model") or None,
        regularization=dict(spec["regularization"]) if spec.get("regularization") else None,
        extent_points=points, plot_locations=plot,
    )


def _grid_surface(grid):
    """Elevation function from a gridded DEM (bilinear, edge-clamped)."""
    from scipy import ndimage
    from scipy.interpolate import RegularGridInterpolator

    x = np.asarray(grid.x, dtype=np.float64)
    y = np.asarray(grid.y, dtype=np.float64)
    z = np.array(grid.values, dtype=np.float64)
    if x[0] > x[-1]:
        x, z = x[::-1], z[:, ::-1]
    if y[0] > y[-1]:
        y, z = y[::-1], z[::-1, :]
    holes = np.isnan(z)
    if holes.all():
        raise ValueError("The DEM contains no valid elevations")
    if holes.any():
        # Fill no-data cells with the nearest valid elevation
        idx = ndimage.distance_transform_edt(holes, return_distances=False,
                                             return_indices=True)
        z = z[tuple(idx)]
    interp = RegularGridInterpolator((y, x), z, bounds_error=False, fill_value=None)

    def surface(qx, qy):
        qx = np.clip(np.asarray(qx, dtype=np.float64), x[0], x[-1])
        qy = np.clip(np.asarray(qy, dtype=np.float64), y[0], y[-1])
        return interp(np.column_stack([qy.ravel(), qx.ravel()])).reshape(qx.shape)

    return surface


def _point_surface(xyz: np.ndarray):
    """Elevation function from scattered (x, y, z) points."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from scipy.spatial import QhullError

    nearest = NearestNDInterpolator(xyz[:, :2], xyz[:, 2])
    try:
        linear = LinearNDInterpolator(xyz[:, :2], xyz[:, 2])
    except (QhullError, ValueError):
        linear = None  # too few or collinear points

    def surface(qx, qy):
        qx = np.asarray(qx, dtype=np.float64)
        q = np.column_stack([qx.ravel(), np.asarray(qy, dtype=np.float64).ravel()])
        z = linear(q) if linear is not None else np.full(len(q), np.nan)
        outside = np.isnan(z)
        if outside.any():
            z[outside] = nearest(q[outside])
        return z.reshape(qx.shape)

    return surface


def _load_topography(params: dict, data_dir: str, crs: str | None = None,
                     datasets: list | None = None) -> tuple:
    """Return (surface(x, y) -> z, info dict, has_dem).

    ``topography``: {"file": DEM grid or x, y, z points}, {"from_data": true} (the
    elevations of the data's stations, e.g. a gravity station table with an elevation
    column), or {"flat_elevation": z}.  A DEM in another CRS than the job's is sampled in
    its own CRS; longitude/latitude points are projected to ``crs``.
    """
    from ..io.crs import GEOGRAPHIC, looks_geographic, project
    from ..io.readers import GridData, detect_format, read_auto, read_station_table

    topo = params.get("topography") or {}
    fname = topo.get("file")
    if fname:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Topography file '{fname}' not found in {data_dir}")
        fmt = detect_format(path)
        print(f"[Pipeline] Loading topography from {fname}")
        if fmt in _GRID_FORMATS:
            grid = read_auto(path)
            surface = _grid_surface(grid)
            if crs and grid.crs and grid.crs != crs:
                native = surface

                def surface(qx, qy, native=native, src=grid.crs):
                    qx, qy = np.asarray(qx, dtype=float), np.asarray(qy, dtype=float)
                    gx, gy = project(qx.ravel(), qy.ravel(), crs, src)
                    return native(gx.reshape(qx.shape), gy.reshape(qy.shape))
                print(f"[Pipeline] Topography is in {grid.crs}: sampled in its own coordinates")
        else:
            if fmt == "csv":
                points = read_station_table(path)
            elif fmt in ("npy", "npz"):
                points = read_auto(path)
            else:
                raise ValueError(f"Unsupported topography file '{fname}'")
            if isinstance(points, GridData):
                raise ValueError(f"Topography '{fname}' has no coordinates")
            xyz = np.array(points.locations, dtype=np.float64)
            if np.isnan(xyz[:, 2]).all() and points.values is not None:
                xyz[:, 2] = points.values  # x, y, z table read as x, y, value
            xyz = xyz[np.isfinite(xyz).all(axis=1)]
            if len(xyz) == 0:
                raise ValueError(f"Topography '{fname}' has no valid x, y, z rows")
            if looks_geographic(xyz[:, 0], xyz[:, 1]):
                if not crs:
                    raise ValueError(f"Topography '{fname}' is in longitude/latitude; give the "
                                     "job's 'crs'")
                xyz[:, 0], xyz[:, 1] = project(xyz[:, 0], xyz[:, 1], GEOGRAPHIC, crs)
            surface = _point_surface(xyz)
        return surface, {"source": "dem", "file": fname}, True

    if topo.get("from_data"):
        pts = [ds.locations for ds in (datasets or []) if ds.extent_points is None
               and np.isfinite(ds.locations[:, 2]).any()]
        xyz = np.vstack(pts) if pts else np.empty((0, 3))
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if len(xyz) < 3:
            raise ValueError("topography from_data: none of the data files gives station "
                             "elevations (a z / elevation column)")
        print(f"[Pipeline] Topography from the elevations of {len(xyz)} stations "
              f"({xyz[:, 2].min():.0f}–{xyz[:, 2].max():.0f} m)")
        return _point_surface(xyz), {"source": "stations", "n_points": int(len(xyz))}, True

    elevation = float(topo.get("flat_elevation", params.get("flat_elevation", 0.0)))

    def flat(qx, qy):
        return np.full(np.shape(qx), elevation, dtype=np.float64)

    return flat, {"source": "flat", "flat_elevation": elevation}, False


def _padded_widths(n_core: int, dh: float, n_pad: int, factor: float) -> np.ndarray:
    core = np.full(n_core, dh)
    pad = dh * factor ** np.arange(1, n_pad + 1)
    return np.concatenate([pad[::-1], core, pad])


def _build_tensor_mesh(extent, surface, has_dem, dx, dz, depth_core, pad_distance):
    """Padded tensor mesh over `extent` whose top follows the ground surface."""
    from ..datamodel.mesh import Mesh3D
    from .meshing import PAD_FACTOR, padding_cells

    xmin, xmax, ymin, ymax = extent
    nx_core = max(4, int(np.ceil((xmax - xmin) / dx)))
    ny_core = max(4, int(np.ceil((ymax - ymin) / dx)))
    nz_depth = max(4, int(np.ceil(depth_core / dz)))
    # Enough expanding cells (dx*1.3, dx*1.3^2, ...) to reach pad_distance.
    # (Counting pad_distance / dx cells ignored the expansion: 100 m cells with
    # 2 km padding gave 20 cells reaching ~80 km.)
    n_pad = padding_cells(dx, pad_distance, PAD_FACTOR)
    pad_factor = PAD_FACTOR

    hx = _padded_widths(nx_core, dx, n_pad, pad_factor)
    hy = _padded_widths(ny_core, dx, n_pad, pad_factor)

    # Ground elevation over the core columns sets the top of the mesh
    cx = xmin + dx * (np.arange(nx_core) + 0.5)
    cy = ymin + dx * (np.arange(ny_core) + 0.5)
    cxx, cyy = np.meshgrid(cx, cy)
    ground = surface(cxx, cyy)
    z_low, z_high = float(np.min(ground)), float(np.max(ground))
    n_relief = int(np.ceil((z_high - z_low) / dz - 1e-9)) if has_dem else 0
    z_top = z_low + n_relief * dz if has_dem else z_high

    core_z = np.full(nz_depth + n_relief, dz)
    pad_z = dz * pad_factor ** np.arange(1, n_pad + 1)
    hz = np.concatenate([pad_z[::-1], core_z])

    origin = (xmin - np.sum(hx[:n_pad]), ymin - np.sum(hy[:n_pad]), z_top - np.sum(hz))
    return Mesh3D(hx=hx, hy=hy, hz=hz, origin=origin)


def _build_octree_mesh(extent, surface, dx, dz, depth_core, pad_distance, levels):
    """Octree mesh refined along the ground surface over the survey area."""
    from discretize.utils import mesh_builder_xyz
    from ..datamodel.mesh import DiscretizeMesh

    xmin, xmax, ymin, ymax = extent
    nxs = max(4, int(np.ceil((xmax - xmin) / dx)) + 1)
    nys = max(4, int(np.ceil((ymax - ymin) / dx)) + 1)
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, nxs), np.linspace(ymin, ymax, nys))
    ground = np.column_stack([gx.ravel(), gy.ravel(), surface(gx, gy).ravel()])

    tree = mesh_builder_xyz(
        ground, [dx, dx, dz],
        depth_core=depth_core,
        padding_distance=[[pad_distance] * 2, [pad_distance] * 2, [pad_distance, 0.0]],
        mesh_type="tree",
        tree_diagonal_balance=True,
    )
    tree.refine_surface(ground, padding_cells_by_level=list(levels), finalize=False)
    tree.finalize()
    return DiscretizeMesh(tree, name="octree")


# The property geology constraints describe for each method's model
GEOLOGY_PROPERTY = {"gravity": "density", "magnetics": "susceptibility",
                    "mt": "resistivity", "dc_resistivity": "resistivity"}


def _face_breaks(term, labels, factor: float) -> np.ndarray:
    """Face weights of a first-order smoothness term: ``factor`` on the faces between cells
    of different labels, 1 elsewhere."""
    grad = getattr(term.regularization_mesh, f"cell_gradient_{term.orientation}").tocoo()
    n_faces = grad.shape[0]
    lo = np.full(n_faces, np.iinfo(np.int64).max)
    hi = np.full(n_faces, np.iinfo(np.int64).min)
    lab = np.asarray(labels, dtype=np.int64)[grad.col]
    np.minimum.at(lo, grad.row, lab)
    np.maximum.at(hi, grad.row, lab)
    return np.where(hi > lo, float(factor), 1.0)


def _apply_geology(spec: dict, task: InversionTask, dmesh, active, surface, data_dir: str, crs):
    """Geology constraints (methods/geology.py) on the task: reference model, per-cell
    bounds and smallness weights, and the start at the reference."""
    from ..methods.geology import build_constraints

    method = canonical_method(task.method_type)
    prop = GEOLOGY_PROPERTY[method]
    if spec.get("property", "density") != prop:
        raise ValueError(f"The geology constraints are for {spec.get('property', 'density')}, "
                         f"the {method} data for {prop}")
    act = active if active is not None else np.ones(dmesh.n_cells, dtype=bool)
    # cells no unit covers keep the method's background (0, or log sigma for MT / DC)
    geo = build_constraints(spec, dmesh.cell_centers[act], dmesh.h_gridded[act] / 2.0, surface,
                            data_dir, crs, (task.bounds_lower, task.bounds_upper),
                            default_reference=float(_get_method(task).default_model_value))
    task.reference_model = geo.reference
    task.cell_lower, task.cell_upper = geo.lower, geo.upper
    task.smallness_weights = geo.weights
    task.smoothness_labels = geo.smoothness_labels()
    task.smoothness_break_factor = float(spec.get("sharp_factor", 0.01))
    span = geo.upper - geo.lower
    nudge = np.where(np.isfinite(span), 1e-4 * span, 1e-4)
    # the start at the reference: for MT / DC (non-linear) this is the starting model proper
    task.initial_model = np.clip(geo.reference, geo.lower + nudge, geo.upper - nudge)
    s = geo.summary()
    show = (lambda u: f"{u['value_ohm_m']:.3g} ohm m [{u['lower_ohm_m']:.3g}, {u['upper_ohm_m']:.3g}]")         if prop == "resistivity" else         (lambda u: f"{u['value']:+.3g} [{u['lower']:+.3g}, {u['upper']:+.3g}]")
    print(f"[Pipeline] Geology: {s['n_constrained']} of {s['n_cells']} cells constrained, "
          f"{s['n_partial']} partly — "
          + ", ".join(f"{u['name']} {show(u)} x{u['weight']:g} on {u['n_cells']}"
                      + (" (sharp)" if u["sharp"] else "") for u in s["units"]))
    return geo


def _active_below_surface(dmesh, surface) -> np.ndarray:
    """Cells whose centres lie below the ground surface."""
    cc = dmesh.cell_centers
    return cc[:, 2] < surface(cc[:, 0], cc[:, 1])


def _lift_buried_stations(dmesh, active, locations: np.ndarray) -> dict | None:
    """Move stations that lie inside active (ground) cells to the top of their column.

    The ground of the mesh is a staircase of cells: a station at its true elevation can
    sit inside the top cell of its column, where part of that cell's rock is above it and
    pulls the wrong way.  ``locations`` (n, 3) is changed in place; returns
    {"n": lifted, "max_m", "mean_m"} or None when no station was inside the ground.
    """
    lift = np.zeros(len(locations))
    todo = np.arange(len(locations))
    for _ in range(64):                      # one cell per pass: never more than the relief
        try:
            idx = np.atleast_1d(dmesh.point2index(locations[todo]))
        except Exception:                    # stations outside the mesh: leave them
            break
        inside = active[idx]
        if not inside.any():
            break
        todo, idx = todo[inside], idx[inside]
        top = dmesh.cell_centers[idx, 2] + 0.5 * dmesh.h_gridded[idx, 2]
        new = top + 1e-3 * dmesh.h_gridded[idx, 2]
        lift[todo] += new - locations[todo, 2]
        locations[todo, 2] = new
    moved = lift > 0
    if not moved.any():
        return None
    return {"n": int(moved.sum()), "max_m": round(float(lift.max()), 1),
            "mean_m": round(float(lift[moved].mean()), 1)}


def _jsonable(obj):
    """Convert numpy containers to plain Python for json.dumps."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _outside_core_share(dmesh, active, model, extent, margin, z_bottom) -> float:
    """Share of the recovered anomaly (|m| x cell volume) outside the core region.

    The core is the survey extent (plus ``margin``) down to ``z_bottom``.  Mass
    or moment in the padding cells beyond it is poorly constrained: smooth,
    depth-weighted regularizations tend to spread sources into the large deep
    and lateral padding cells, which fit the data as well as the true body.
    """
    cc, vol = dmesh.cell_centers, dmesh.cell_volumes
    if active is not None:
        cc, vol = cc[active], vol[active]
    xmin, xmax, ymin, ymax = extent
    core = ((cc[:, 0] >= xmin - margin) & (cc[:, 0] <= xmax + margin)
            & (cc[:, 1] >= ymin - margin) & (cc[:, 1] <= ymax + margin)
            & (cc[:, 2] >= z_bottom))
    moment = np.abs(np.asarray(model, dtype=float)) * vol
    total = moment.sum()
    return float(moment[~core].sum() / total) if total > 0 else 0.0


def _outside_core_parts(dmesh, active, model, extent, margin, z_bottom) -> tuple[float, float]:
    """(beside, below): the shares of the recovered anomaly (|m| x cell volume) beside the
    core (outside the survey extent plus ``margin``, above ``z_bottom``) and below it."""
    cc, vol = dmesh.cell_centers, dmesh.cell_volumes
    if active is not None:
        cc, vol = cc[active], vol[active]
    xmin, xmax, ymin, ymax = extent
    inside = ((cc[:, 0] >= xmin - margin) & (cc[:, 0] <= xmax + margin)
              & (cc[:, 1] >= ymin - margin) & (cc[:, 1] <= ymax + margin))
    deep = cc[:, 2] < z_bottom
    moment = np.abs(np.asarray(model, dtype=float)) * vol
    total = moment.sum()
    if not total > 0:
        return 0.0, 0.0
    return float(moment[~inside & ~deep].sum() / total), float(moment[deep].sum() / total)


def _padding_note(name: str, share: float, beside: float, below: float, kind: str) -> str:
    """What to make of a model largely in the padding, by where it sits."""
    text = (f"{share:.0%} of the recovered {name} anomaly lies outside the core mesh "
            f"(padding cells), where it is poorly constrained")
    if beside >= below:
        text += (f": {beside:.0%} beside the data area. Anomalies running across the edges of "
                 f"the data come from sources outside them, which the padding holds (counted "
                 f"by volume, its large cells weigh much); a window wider than the area you "
                 f"interpret moves those edges away from it")
    else:
        text += (f": {below:.0%} below the core. A deeper core (depth_core_m), or bounds on the "
                 f"property, keep it where the data constrain it")
    if kind not in ("l1l2", "sparse", "mgs", "tv"):
        text += "; a compact regularization (L1–L2, or Lp norms) also helps"
    return text + "."


MESH_KEYS = ("core_cell_m", "core_cell_z_m", "depth_core_m", "pad_distance_m")
# Used only when neither the params nor the data give the mesh settings
_FALLBACK_MESH = {"core_cell_m": 500.0, "core_cell_z_m": 250.0,
                  "depth_core_m": 3000.0, "pad_distance_m": 2000.0}


def _mesh_design(params: dict, datasets, extent, n_data: int) -> dict:
    """Mesh settings: those given in params, the rest recommended from the data.

    Returns {"source": "user" | "auto" | "mixed" | "fallback",
             "data_spacing": [{method, spacing_m, kind}], "recommended": dict | None,
             "used": {core_cell_m, core_cell_z_m, depth_core_m, pad_distance_m}}.
    """
    from .meshing import recommend_mesh

    given = {k: float(params[k]) for k in MESH_KEYS if params.get(k) is not None}
    spacing = [{"method": ds.method, "spacing_m": ds.spacing, "kind": ds.spacing_kind}
               for ds in datasets]
    known = [ds.spacing for ds in datasets if ds.spacing]
    recommended = recommend_mesh(extent, min(known), n_data) if known else None
    base = {k: recommended[k] for k in MESH_KEYS} if recommended else _FALLBACK_MESH
    if len(given) == len(MESH_KEYS):
        source = "user"
    elif not given:
        source = "auto" if recommended else "fallback"
    else:
        source = "mixed"
    return {"source": source, "data_spacing": spacing, "recommended": recommended,
            "used": {**base, **given}, "client": params.get("mesh_design")}


def run_data_pipeline(params: dict, data_dir: str, progress=None) -> dict:
    """Full pipeline: raw data files -> mesh -> inversion -> result.

    This is the 'data' mode worker: it reads raw survey files from
    `data_dir`, builds the mesh and survey objects, then runs the
    inversion according to `params`.

    Args:
        params: Pipeline parameters (the upload page's ``params_json``):
            inversion_mode: "single" | "joint".
            datasets: [{method, files, noise_pct, noise_floor, component,
                method_kwargs}], one per data type.  Files may be grids
                (.tif/.grd/.asc) or station data (.csv/.xyz/.txt/.dat/.obs/
                .npy/.npz); several files of one dataset are concatenated.
                Legacy params without `datasets` use data_file / method_type /
                noise_pct / noise_floor / method_kwargs.
            gz_convention: "positive_down" (default: Bouguer/field data,
                negated for SimPEG) or "simpeg" (z-up, as SimPEG models it);
                also per dataset.
            regional: regional field removed from each dataset before inverting
                (also per dataset): "mean" or {"method": "polynomial",
                "order": k} (trend surface); default none.
            topography: {file: DEM name} or {flat_elevation: metres}.
                Stations without an elevation are placed on the surface
                (+ station_height, default 0 m; a dataset's own
                station_height overrides it); with a DEM, cells above
                the surface are inactive, and stations inside the staircase
                of ground cells are moved up to the top of their column.
            param_mode: "auto" (defaults; manual-only keys are ignored) or
                "manual" (norms, alpha_*, beta0_ratio, cooling_factor,
                max_iter, max_irls_iterations, use_preconditioner, bounds_*).
            regularization_type: "sparse" (lp-norm IRLS), "l1l2" (L1–L2
                elastic net, Utsugi 2019; mixing set by l1_ratio), "mgs"
                (minimum gradient support focusing), "tv" (total variation)
                or "l2" (smooth L2 with depth weighting).  focusing_percentile / focusing_scale set the
                MGS/TV focusing parameter.  Joint gravity + magnetics may use
                "group_lasso" (Utsugi 2025; settings gl_*, see InversionTask and
                run_group_lasso_joint).
            depth_weighting: "sensitivity" (default) or "depth" (Li &
                Oldenburg, exponent depth_weighting_exponent, default 2);
                not used by L1–L2, whose weighting is l1l2_weighting.
            beta_selection: "auto" (default), "discrepancy", "lcurve" or
                "gcv" (manual mode, single inversions); beta_sweep optionally
                lists the betas to try.
            l1l2_solver: "cda" (default; Utsugi 2019 lambda path with the
                L-curve) or "irls"; l1l2_weighting "S2" (default) or "S1";
                lambda_decades, lambda_step set the lambda path.
            joint_weights, cross_gradient_weight: joint data weights and
                structural coupling (auto: all 1).
            mesh_type: "tensor" (default) or "octree"; plus core_cell_m,
                core_cell_z_m, depth_core_m, pad_distance_m, octree_levels.
            aoi: [west, east, south, north] (m) crops every dataset.
            decimate_spacing_m: thin the data to about this spacing (m),
                per dataset or for the whole job: grids by whole strides per
                axis, points to one per cell.  decimate_stride is the older
                grid-only stride.
        data_dir: Local directory containing the downloaded data files.
        progress: Optional callable ``progress(stage, message="", **fields)``
            told about the stages and each inversion iteration (the cloud
            worker passes an S3Progress).

    Returns:
        Result dict compatible with pack_result().
    """
    from ..methods.directives import IterationCollector

    report = progress or (lambda stage, message="", **fields: None)
    report("loading_data")
    params = dict(params)
    param_mode = params.get("param_mode")
    if param_mode == "auto":
        ignored = sorted(k for k in _MANUAL_KEYS if k in params)
        if ignored:
            print(f"[Pipeline] Auto mode: ignoring manual settings {ignored}")
        params = {k: v for k, v in params.items() if k not in _MANUAL_KEYS}
        if "group_lasso" in (params.get("regularization_type"), params.get("coupling")):
            params.update(GROUP_LASSO_AUTO)

    topo_file = (params.get("topography") or {}).get("file")
    specs = _dataset_specs(params, data_dir, exclude={topo_file} if topo_file else set())

    mode = params.get("inversion_mode") or (
        "joint" if params.get("method_type") == "joint" or len(specs) > 1 else "single"
    )
    if mode == "single" and len(specs) != 1:
        raise ValueError(
            f"Single inversion needs exactly one dataset, got {len(specs)}; "
            f"use inversion_mode='joint' or submit one job per dataset"
        )
    if mode == "joint" and len(specs) < 2:
        raise ValueError("Joint inversion needs at least two datasets")

    crs = _job_crs(params, data_dir, specs, topo_file)
    datasets = [_load_dataset(s, params, data_dir, single=(mode == "single"), crs=crs)
                for s in specs]

    # ── Topography and station elevations ──
    surface, topo_info, has_dem = _load_topography(params, data_dir, crs, datasets)
    for ds, spec in zip(datasets, specs):
        if ds.extent_points is not None:   # MT / DC: elevations are part of the data
            continue
        # above the ground: per dataset (a flight height), else the job's value; data
        # continued upwards are at the height they were continued to
        station_height = float(spec.get("continue_to_m") or params.get("continue_to_m") or 0.0) \
            if ds.continuation else float(spec.get("station_height", params.get("station_height", 0.0)))
        missing = np.isnan(ds.locations[:, 2])
        if missing.any():
            ds.locations[missing, 2] = (
                surface(ds.locations[missing, 0], ds.locations[missing, 1]) + station_height
            )

    report("building_mesh")

    # ── Mesh over the union of all survey extents ──
    all_locs = np.vstack([ds.mesh_points for ds in datasets])
    extent = (float(all_locs[:, 0].min()), float(all_locs[:, 0].max()),
              float(all_locs[:, 1].min()), float(all_locs[:, 1].max()))
    n_data = int(sum(ds.observed.size for ds in datasets))
    mesh_design = _mesh_design(params, datasets, extent, n_data)
    core_cell_m = mesh_design["used"]["core_cell_m"]
    core_cell_z_m = mesh_design["used"]["core_cell_z_m"]
    depth_core_m = mesh_design["used"]["depth_core_m"]
    pad_distance_m = mesh_design["used"]["pad_distance_m"]
    print(f"[Pipeline] Mesh design ({mesh_design['source']}): "
          f"{core_cell_m:g} m x {core_cell_z_m:g} m cells, core {depth_core_m:g} m deep, "
          f"padding {pad_distance_m:g} m")

    mesh_type = str(params.get("mesh_type") or "tensor").lower()
    if mesh_type == "octree":
        mesh = _build_octree_mesh(
            extent, surface, core_cell_m, core_cell_z_m, depth_core_m, pad_distance_m,
            params.get("octree_levels", [4, 4, 4]),
        )
    elif mesh_type == "tensor":
        mesh = _build_tensor_mesh(
            extent, surface, has_dem, core_cell_m, core_cell_z_m, depth_core_m,
            pad_distance_m,
        )
    else:
        raise ValueError(f"Unknown mesh_type '{mesh_type}' (expected 'tensor' or 'octree')")
    dmesh = mesh.to_discretize()

    # Octree meshes extend above the ground even for flat topography
    active = None
    if has_dem or mesh_type == "octree":
        active = _active_below_surface(dmesh, surface)
        if not active.any():
            raise ValueError("No mesh cells lie below the topography surface")
    n_params = int(active.sum()) if active is not None else mesh.n_cells
    # Stations cannot be underground: those inside the staircase of ground cells go to
    # the top of their column (MT / DC stations are part of their own geometry)
    lifts = {}
    if active is not None:
        for ds in datasets:
            if ds.extent_points is None:
                lifted = _lift_buried_stations(dmesh, active, ds.locations)
                if lifted:
                    lifts[ds.method] = lifted
                    print(f"[Pipeline] {ds.method}: {lifted['n']} of {len(ds.locations)} stations "
                          f"were inside ground cells and moved up to the top of their column "
                          f"(by {lifted['mean_m']:g} m on average, at most {lifted['max_m']:g} m)")
    if lifts:
        topo_info["stations_lifted"] = lifts
    # No sources deeper than this below the ground (gravity and magnetics): those cells
    # leave the model like the air, so they stay 0 and the data are fitted above them
    max_src = params.get("max_source_depth_m")
    max_src = float(max_src) if max_src not in (None, "") else None
    if max_src is not None:
        if not max_src > 0:
            raise ValueError("max_source_depth_m must be positive")
        if any(canonical_method(ds.method) not in ("gravity", "magnetics") for ds in datasets):
            raise ValueError("A maximum source depth is for gravity and magnetic data (for DC "
                             "or MT the cells taken out would be air)")
        cc = dmesh.cell_centers
        deep = surface(cc[:, 0], cc[:, 1]) - cc[:, 2] > max_src
        active = (np.ones(dmesh.n_cells, bool) if active is None else active) & ~deep
        if not active.any():
            raise ValueError(f"No mesh cells lie within {max_src:g} m of the ground")
        n_params = int(active.sum())
        print(f"[Pipeline] No sources below {max_src:g} m: {int(deep.sum())} deeper cells "
              f"taken out of the model ({n_params} left)")
    print(f"[Pipeline] {mesh_type} mesh: {mesh.n_cells} cells "
          f"({n_params} active), {n_data} observations, "
          f"topography: {topo_info['source']}")
    # The ground over the survey, for the record and for the viewer's 3D view
    gx, gy = np.meshgrid(np.linspace(extent[0], extent[1], 101),
                         np.linspace(extent[2], extent[3], 101))
    ground = surface(gx, gy)
    topo_info.update(elevation_min=round(float(np.min(ground)), 1),
                     elevation_max=round(float(np.max(ground)), 1))
    topography_grid = {"x": gx[0], "y": gy[:, 0], "z": ground} if has_dem else None

    # ── Inversion settings ──
    task_kwargs = {k: params[k] for k in _REG_PARAM_KEYS if params.get(k) is not None}
    if params.get("norms") is not None:
        task_kwargs["norms"] = tuple(float(v) for v in params["norms"])
    tensor_fields = {}
    if mesh_type == "tensor":
        tensor_fields = dict(hx=np.asarray(mesh.hx), hy=np.asarray(mesh.hy),
                             hz=np.asarray(mesh.hz))
    common = dict(
        task_id=params.get("task_id", "pipeline"),
        origin=tuple(float(v) for v in dmesh.origin),
        regularization_type=params.get("regularization_type", "sparse"),
        active_cells=active,
        **tensor_fields,
        **task_kwargs,
    )

    notes = []
    if mode == "joint":
        weights = params.get("joint_weights")
        weights = [float(w) for w in weights] if weights is not None else [1.0] * len(datasets)
        if len(weights) != len(datasets):
            raise ValueError(
                f"joint_weights has {len(weights)} entries for {len(datasets)} datasets"
            )
        from ..methods.coupling import resolve
        # the coupling: its own key, else as before (regularization_type "group_lasso", or
        # the cross-gradient at cross_gradient_weight, 1 by default)
        coupling = resolve(params.get("coupling"), common["regularization_type"],
                           float(params.get("cross_gradient_weight", 1.0)))
        if coupling == "group_lasso":
            common["regularization_type"] = "group_lasso"
        weight = params.get("coupling_weight", params.get("cross_gradient_weight", 1.0))
        task = InversionTask(
            method_type="joint",
            joint_methods=[ds.method for ds in datasets],
            joint_kwargs_list=[ds.method_kwargs for ds in datasets],
            joint_weights=weights,
            joint_coupling=coupling,
            coupling_weight=float(weight),
            coupling_options=dict(params.get("coupling_options") or {}),
            cross_gradient_weight=float(weight) if coupling == "cross_gradient" else 0.0,
            joint_models=[ds.model for ds in datasets] if any(ds.model for ds in datasets)
            else None,
            joint_regularizations=[ds.regularization for ds in datasets]
            if param_mode != "auto" and any(ds.regularization for ds in datasets) else None,
            joint_surveys=[
                {"locations": ds.locations, "observed": ds.sign * ds.observed, "std": ds.std}
                for ds in datasets
            ],
            **common,
        )
        if task.coupling == "group_lasso":
            if any(w != 1.0 for w in weights):
                notes.append("The group lasso balances the two datasets by its data scaling "
                             f"(gl_data_scaling='{task.gl_data_scaling}'); joint_weights are not "
                             "applied")
            if task.bounds_lower is not None or task.bounds_upper is not None:
                notes.append("The bounds apply to every model of the group lasso "
                             "(per dataset: its regularization's bounds_lower / bounds_upper)")
            used = {"bounds_lower", "bounds_upper"} | (
                {"depth_weighting_exponent"} if task.gl_weighting == "depth" else set())
            unused = sorted({k for r in (task.joint_regularizations or []) if r for k in r}
                            - used)
            if unused:
                notes.append("The group lasso uses only the bounds of a dataset's regularization"
                             + (" and its depth_weighting_exponent (gl_weighting 'depth')"
                                if task.gl_weighting == "depth" else "")
                             + f"; {unused} are not applied")
            if task.beta_selection not in ("auto", "discrepancy"):
                notes.append("The group lasso chooses lambda1 by gl_lambda1_selection="
                             f"'{task.gl_lambda1_selection}'; beta_selection is not applied")
        elif task.regularization_type == "l1l2":
            if task.l1l2_solver == "cda":
                notes.append("Joint inversion runs the L1–L2 regularization by IRLS; the "
                             "coordinate-descent lambda path (l1l2_solver='cda') is "
                             "single-method")
            notes.append("L1–L2 regularizes the model values only (no smoothness "
                         "term); norms and alpha_x/y/z are not used")
        if task.coupling == "pgi":
            notes.append("PGI regularizes the models by its rock units and their smoothness; "
                         f"regularization_type='{task.regularization_type}' and the coupling "
                         "weight are not used")
        if task.coupling != "group_lasso":
            if task.bounds_lower is not None or task.bounds_upper is not None:
                notes.append("The bounds apply to every model of the joint inversion "
                             "(joint_regularizations sets them per dataset)")
            if task.beta_selection not in ("auto", "discrepancy"):
                notes.append("Joint inversion chooses beta by the discrepancy principle; "
                             f"beta_selection='{task.beta_selection}' is not applied")
    else:
        ds = datasets[0]
        task = InversionTask(
            method_type=ds.method,
            method_kwargs=ds.method_kwargs,
            noise_pct=ds.noise_pct,
            noise_floor=ds.noise_floor,
            initial_model=np.full(n_params, float(
                _make_method(ds.method, ds.method_kwargs).default_model_value)),
            observed_data=ds.sign * ds.observed,
            data_std=ds.std,
            station_locations=ds.locations,
            **common,
        )
        if task.regularization_type == "l1l2":
            notes.append("L1–L2 regularizes the model values only (no smoothness "
                         "term); norms and alpha_x/y/z are not used")
    geology = None
    if params.get("geology"):
        if mode != "single":
            notes.append("Geology constraints apply to single inversions in this version; "
                         "they were not applied")
        else:
            geology = _apply_geology(params["geology"], task, dmesh, active, surface, data_dir, crs)
            notes += geology.notes
    for note in notes:
        print(f"[Pipeline] Note: {note}")

    group_lasso = mode == "joint" and task.coupling == "group_lasso"
    report("inverting", max_iter=task.gl_n_lambda1 if group_lasso else task.max_iter,
           phi_d_target=n_data, n_data=n_data, n_cells=n_params,
           regularization=task.regularization_type, history=[])
    history = []   # every iteration so far, for the page's live convergence curves

    def on_iteration(snap):
        # (the λ paths of the L1–L2 coordinate descent and the group lasso report no phi_m)
        history.append({"i": snap.iteration, "phi_d": snap.phi_d,
                        "phi_m": getattr(snap, "phi_m", None), "beta": snap.beta})
        report("inverting", iteration=snap.iteration, phi_d=snap.phi_d, beta=snap.beta,
               history=history[-HISTORY_POINTS:], **getattr(snap, "extra", {}))

    IterationCollector.on_iteration = on_iteration
    try:
        result = execute_task(task, mesh=mesh)
    finally:
        IterationCollector.on_iteration = None
    convergence = assess_convergence(result, n_data, task.max_iter)
    if convergence is not None:
        result["convergence"] = convergence

    result.update({
        "inversion_mode": mode,
        "param_mode": param_mode or "legacy",
        "regularization_type": task.regularization_type,
        "mesh_type": mesh_type,
        "mesh_design": mesh_design,
        "mesh_shape": tuple(mesh.shape),
        "mesh": _jsonable(dmesh.to_dict()),
        "n_cells": mesh.n_cells,
        "n_active_cells": n_params,
        "n_data": n_data,
        "cell_centers_x": extent[:2],
        "cell_centers_y": extent[2:],
        "topography": topo_info,
        "crs": crs,
        "datasets": [
            {"method": ds.method, "component": ds.component, "files": ds.files,
             "n_data": int(ds.observed.size), "noise_pct": ds.noise_pct,
             "noise_floor": ds.noise_floor, "noise_auto": ds.noise_auto, "regional": ds.regional,
             "gz_convention": ("positive_down" if ds.sign < 0 else "simpeg")
             if ds.method == "gravity" and ds.component == "gz" else None,
             "decimation": ds.decimation,
             **({"rtp": ds.rtp} if ds.rtp else {}),
             **({"continuation": ds.continuation} if ds.continuation else {}),
             **({"igrf": ds.igrf} if ds.igrf else {}),
             **({"bouguer_check": ds.bouguer_check} if ds.bouguer_check else {})}
            for ds in datasets
        ],
    })
    if active is not None:
        result["active_cells"] = active
    if topography_grid is not None:
        result["topography_grid"] = topography_grid
    if geology is not None:   # what was assumed, and the reference model itself (shown in 3D)
        result["geology"] = geology.summary()
        result["reference_model"] = geology.reference
    if mode != "joint":
        ds = datasets[0]
        result["data"] = {"locations": ds.data_locations, "observed": ds.observed,
                          "std": ds.std}
        if ds.trend is not None:
            result["data"]["regional"] = ds.trend   # observed + regional = the data as loaded
        if result.get("predicted") is not None:
            # back to the file's sign convention, like "observed"
            result["data"]["predicted"] = ds.sign * np.asarray(result.pop("predicted"), dtype=float)
    # per-dataset data of a joint run, in the files' convention too
    labels = result.get("dataset_labels") or [ds.method for ds in datasets]
    for ds, label in zip(datasets, labels):
        jd = (result.get("joint_data") or {}).get(label)
        if jd is not None:
            jd.update(locations=ds.data_locations, observed=ds.observed, std=ds.std,
                      predicted=ds.sign * np.asarray(jd["predicted"], dtype=float))
            if ds.trend is not None:
                jd["regional"] = ds.trend
    if group_lasso:
        result["settings"] = {"regularization_type": "group_lasso",
                              **{k: getattr(task, k) for k in GROUP_LASSO_KEYS}}
        # the hybrid with a cross-gradient is a coupling of its own (not Utsugi's method)
        result["settings"]["coupling"] = _group_lasso_coupling(task, [])["kind"]
        # how its cells were weighed (the workflow tree's weighting column)
        if task.gl_weighting == "depth":
            result["settings"]["depth_weighting"] = "depth"
            result["settings"]["depth_weighting_exponent"] = _common_setting(
                task, datasets, "depth_weighting_exponent")
        else:
            result["settings"]["depth_weighting"] = task.gl_weighting
    else:
        result["settings"] = {k: getattr(task, k) for k in (
            "regularization_type", "max_iter", "max_irls_iterations", "beta_selection",
            "l1_ratio", "alpha_s", "alpha_x", "alpha_y", "alpha_z", "bounds_lower", "bounds_upper",
            "depth_weighting", "depth_weighting_exponent",
        ) if hasattr(task, k)}
        if task.regularization_type == "l1l2":   # the viewer names the solver
            for k in ("l1l2_solver", "l1l2_weighting"):
                if hasattr(task, k):
                    result["settings"][k] = getattr(task, k)
        result["settings"]["alpha_s"] = effective_alpha_s(
            task, length_scales=task.regularization_type in ("l2", "sparse", "l1l2", "mgs", "tv")
            or mode == "joint")
        result["settings"]["norms"] = list(task.norms)
        if mode == "joint":
            # the models' own regularizations (joint_regularizations), where they agree
            overridden = {k for ds in datasets for k in (ds.regularization or {})}
            for key in ("regularization_type", "alpha_s", "norms", "depth_weighting",
                        "depth_weighting_exponent", "l1_ratio"):
                if key in overridden:
                    result["settings"][key] = _common_setting(task, datasets, key)
        else:
            kwargs = datasets[0].method_kwargs or {}
            if kwargs.get("magnetization") == "vector":
                result["settings"]["magnetization"] = "vector"
        if mode == "joint":   # the first column of the workflow tree
            result["settings"]["coupling"] = task.coupling
            if task.coupling == "pgi":
                result["settings"]["regularization_type"] = "pgi"
            elif task.coupling != "none":
                result["settings"]["coupling_weight"] = task.effective_coupling_weight
    if geology is not None:   # a column of the workflow tree: runs with and without it
        result["settings"]["geology"] = str(params["geology"].get("name") or "constrained")
    if max_src is not None:   # likewise: no sources below this depth
        result["settings"]["max_source_depth_m"] = max_src

    # How much of the recovered anomaly sits outside the core (in padding)?
    gx, gy = np.meshgrid(np.linspace(extent[0], extent[1], 20),
                         np.linspace(extent[2], extent[3], 20))
    z_bottom = float(np.min(surface(gx, gy))) - depth_core_m
    models = result.get("recovered_models") or (
        {task.method_type: result["recovered_model"]} if "recovered_model" in result else {})
    # the anomaly: the model minus its background (log sigma_background for MT / DC)
    background = {}
    labels = result.get("dataset_models") or [ds.method for ds in datasets]
    for ds, label in zip(datasets, labels if mode == "joint" else [task.method_type]):
        background.setdefault(label, float(
            _make_method(ds.method, ds.method_kwargs).default_model_value))
    shares = {
        name: _outside_core_share(dmesh, active, np.asarray(m) - background.get(name, 0.0),
                                  extent, core_cell_m, z_bottom)
        for name, m in models.items() if np.size(m) == n_params
    }
    if shares:
        result["outside_core_share"] = shares
        for name, share in shares.items():
            if share > PADDING_WARNING_SHARE:
                beside, below = _outside_core_parts(
                    dmesh, active, np.asarray(models[name]) - background.get(name, 0.0),
                    extent, core_cell_m, z_bottom)
                notes.append(_padding_note(name, share, beside, below,
                                           str(getattr(task, "regularization_type", ""))))
    if notes:
        result["notes"] = notes
    return result


def _common_setting(task, datasets, key):
    """A regularization setting of a joint job's models: the datasets' own values
    (joint_regularizations), else the task's; "per model" when they differ."""
    vals = []
    for ds in datasets:
        v = (ds.regularization or {}).get(key, getattr(task, key, None))
        vals.append(list(v) if isinstance(v, (tuple, list)) else v)
    return vals[0] if all(v == vals[0] for v in vals) else "per model"


def result_metadata_json(result: dict) -> str:
    """result.json: everything in a result except the model arrays."""
    meta = {k: v for k, v in result.items()
            if k not in ("recovered_model", "recovered_models", "active_cells", "data",
                         "joint_data", "predicted", "topography_grid", "reference_model",
                         "magnetization_vector", "posterior_std", "prob_body", "prior_std",
                         "samples", "bayes_spectrum")}
    return json.dumps(_jsonable(meta), indent=2, default=str)


class ProgressReport:
    """Progress report the worker keeps for the API (``progress.json``).

    Called as ``progress(stage, message="", **fields)``; fields persist
    between calls.  Stages: starting, downloading, loading_data,
    building_mesh, inverting, uploading, done, failed.  The report is written
    on every stage change and otherwise at most every ``min_interval``
    seconds, so per-iteration updates stay cheap.  Failures to write are
    printed, never raised.  Subclasses define where it goes (``_write``).
    """

    def __init__(self, min_interval: float = 10.0, clock=time.time) -> None:
        self.min_interval, self.clock = min_interval, clock
        self.state: dict = {"stage": None, "started_at": clock()}
        self._last_write = -float("inf")

    def __call__(self, stage: str, message: str = "", **fields) -> None:
        now = self.clock()
        changed = stage != self.state["stage"]
        self.state.update(fields)
        self.state.update(stage=stage, message=message, updated_at=now)
        if changed or now - self._last_write >= self.min_interval:
            self._last_write = now
            try:
                self._write(json.dumps(_jsonable(self.state)).encode())
            except Exception as e:
                print(f"[Worker] Could not update progress: {e}")

    def _write(self, body: bytes) -> None:
        raise NotImplementedError


class S3Progress(ProgressReport):
    """Progress report in S3 (AWS Batch backend)."""

    def __init__(self, s3, bucket: str, key: str, min_interval: float = 10.0,
                 clock=time.time) -> None:
        super().__init__(min_interval, clock)
        self.s3, self.bucket, self.key = s3, bucket, key

    def _write(self, body: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=self.key, Body=body,
                           ContentType="application/json")


class FileProgress(ProgressReport):
    """Progress report in a local file, replaced atomically (EC2 backend)."""

    def __init__(self, path, min_interval: float = 10.0, clock=time.time) -> None:
        super().__init__(min_interval, clock)
        self.path = Path(path)

    def _write(self, body: bytes) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(body)
        os.replace(tmp, self.path)


def run_local_job(params_path: str, data_dir: str, out_dir: str) -> int:
    """Run a data-pipeline job from local files; returns the exit status.

    Writes out_dir/progress.json, then out_dir/result.zip and result.json
    (with ``error`` and ``traceback`` if it failed).
    """
    from ..methods.directives import IterationCollector

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # the local backend asks for frequent reports (the page's live curves); EC2 reads
    # the file over SSH and keeps the default
    progress = FileProgress(out / "progress.json",
                            min_interval=float(os.environ.get("GEOINV3D_PROGRESS_INTERVAL", 10)))
    progress("starting")
    # out_dir/STOP (written by the API's "stop and keep the result") ends the
    # inversion after its current iteration; the result is then kept as usual
    stop_flag = out / "STOP"
    IterationCollector.stop_check = stop_flag.exists
    try:
        with open(params_path, encoding="utf-8") as f:
            params = json.load(f)
        result = run_data_pipeline(params, data_dir, progress=progress)
        failed = False
        print(f"[Worker] Inversion complete: {result.get('n_iterations', 0)} iterations")
    except Exception as e:
        failed = True
        result = {"task_id": os.environ.get("TASK_ID", "local"), "error": str(e),
                  "traceback": traceback.format_exc()}
        print(f"[Worker] ERROR: {e}")
        progress("failed", message=str(e))
    finally:
        IterationCollector.stop_check = None
    if not failed:
        progress("uploading")
    pack_result(result, str(out / "result.zip"))
    (out / "result.json").write_text(result_metadata_json(result), encoding="utf-8")
    if failed:
        return 1
    progress("done", message=f"{result.get('n_iterations', 0)} iterations"
             + (" (stopped early, result kept)" if result.get("stopped_early") else ""))
    return 0


def pack_result(result: dict, output_path: str) -> str:
    """Pack inversion result into a .zip archive."""
    import io
    import zipfile

    path = Path(output_path)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("result.json", result_metadata_json(result))

        if result.get("active_cells") is not None:
            buf = io.BytesIO()
            np.save(buf, np.asarray(result["active_cells"]))
            zf.writestr("active_cells.npy", buf.getvalue())

        if result.get("data"):
            # stations (n, 3), observed, std and predicted data of a single-dataset run
            buf = io.BytesIO()
            np.savez(buf, **{k: np.asarray(v, dtype=float) for k, v in result["data"].items()})
            zf.writestr("data.npz", buf.getvalue())

        if result.get("reference_model") is not None:   # geology constraints (active cells)
            buf = io.BytesIO()
            np.save(buf, np.asarray(result["reference_model"], dtype=float))
            zf.writestr("reference_model.npy", buf.getvalue())

        if result.get("magnetization_vector") is not None:   # MVI: (n, 3) east, north, up
            buf = io.BytesIO()
            np.save(buf, np.asarray(result["magnetization_vector"], dtype=float))
            zf.writestr("magnetization_vector.npy", buf.getvalue())

        if result.get("bayes_spectrum"):   # the whitened data in the eigenbasis (evidence)
            buf = io.BytesIO()
            np.savez(buf, **{k: np.asarray(v, dtype=float) for k, v in result["bayes_spectrum"].items()})
            zf.writestr("bayes_spectrum.npz", buf.getvalue())
        for key in ("posterior_std", "prob_body", "prior_std", "samples"):   # Bayesian runs
            if result.get(key) is not None:
                buf = io.BytesIO()
                np.save(buf, np.asarray(result[key], dtype=np.float32))
                zf.writestr(f"{key}.npy", buf.getvalue())

        if result.get("topography_grid"):
            # the ground over the survey (x, y axes and z on their grid), for the 3D view
            buf = io.BytesIO()
            np.savez(buf, **{k: np.asarray(v, dtype=float)
                             for k, v in result["topography_grid"].items()})
            zf.writestr("topography.npz", buf.getvalue())

        for name, data in (result.get("joint_data") or {}).items():
            # the same for each dataset of a joint run
            buf = io.BytesIO()
            np.savez(buf, **{k: np.asarray(v, dtype=float) for k, v in data.items()})
            zf.writestr(f"data_{name}.npz", buf.getvalue())

        if "recovered_model" in result:
            buf = io.BytesIO()
            np.save(buf, result["recovered_model"])
            zf.writestr("recovered_model.npy", buf.getvalue())

        if "recovered_models" in result:
            for name, model in result["recovered_models"].items():
                buf = io.BytesIO()
                np.save(buf, model)
                zf.writestr(f"recovered_{name}.npy", buf.getvalue())

    return str(path)


def main():
    """Entry point for AWS Batch container (or ``--local`` on an EC2 instance)."""
    if len(sys.argv) > 1 and sys.argv[1] == "--local":
        if len(sys.argv) != 5:
            print("usage: python -m geoinv3d.cloud.worker --local params.json DATA_DIR OUT_DIR")
            sys.exit(2)
        sys.exit(run_local_job(*sys.argv[2:5]))

    bucket = os.environ.get("TASK_BUCKET")
    task_key = os.environ.get("TASK_KEY")
    task_id = os.environ.get("TASK_ID", "unknown")
    result_prefix = os.environ.get("RESULT_PREFIX", "")
    pipeline_mode = os.environ.get("PIPELINE_MODE", "task")

    if not bucket:
        print("ERROR: TASK_BUCKET environment variable required")
        sys.exit(1)

    import boto3
    s3 = boto3.client("s3")
    progress = S3Progress(s3, bucket, f"{result_prefix}/progress.json")
    progress("starting")

    print(f"[Worker] Starting task {task_id} (mode={pipeline_mode})")

    failed = False
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            progress("downloading")
            if pipeline_mode == "data":
                params_key = os.environ.get("PIPELINE_PARAMS", "")
                data_prefix = os.environ.get("DATA_PREFIX", "")

                print(f"[Worker] Downloading pipeline params from s3://{bucket}/{params_key}")
                params_path = os.path.join(tmpdir, "params.json")
                s3.download_file(bucket, params_key, params_path)
                with open(params_path) as f:
                    params = json.load(f)

                data_dir = os.path.join(tmpdir, "data")
                os.makedirs(data_dir, exist_ok=True)

                paginator = s3.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=bucket, Prefix=data_prefix):
                    for obj in page.get("Contents", []):
                        key = obj["Key"]
                        fname = os.path.basename(key)
                        if fname:
                            local = os.path.join(data_dir, fname)
                            print(f"[Worker] Downloading {key}")
                            s3.download_file(bucket, key, local)

                result = run_data_pipeline(params, data_dir, progress=progress)

            else:
                if not task_key:
                    print("ERROR: TASK_KEY required in task mode")
                    sys.exit(1)

                input_path = os.path.join(tmpdir, "input.zip")
                print(f"[Worker] Downloading s3://{bucket}/{task_key}")
                s3.download_file(bucket, task_key, input_path)

                task = unpack_task(input_path)
                print(f"[Worker] Task: method={task.method_type}, "
                      f"reg={task.regularization_type}, "
                      f"norms={task.norms}, "
                      f"max_iter={task.max_iter}")

                from ..methods.directives import IterationCollector
                progress("inverting", max_iter=task.max_iter)
                IterationCollector.on_iteration = lambda snap: progress(
                    "inverting", iteration=snap.iteration, phi_d=snap.phi_d, beta=snap.beta)
                try:
                    result = execute_task(task)
                finally:
                    IterationCollector.on_iteration = None

            print(f"[Worker] Inversion complete: "
                  f"{result.get('n_iterations', 0)} iterations")

        except Exception as e:
            failed = True
            result = {
                "task_id": task_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            print(f"[Worker] ERROR: {e}")
            progress("failed", message=str(e))

        if not failed:
            progress("uploading")
        result_path = os.path.join(tmpdir, "result.zip")
        pack_result(result, result_path)

        result_key = f"{result_prefix}/result.zip"
        s3.upload_file(result_path, bucket, result_key)
        s3.put_object(Bucket=bucket, Key=f"{result_prefix}/result.json",
                      Body=result_metadata_json(result).encode(),
                      ContentType="application/json")
        print(f"[Worker] Uploaded result to s3://{bucket}/{result_key}")

    if failed:
        # The error is in result.json; a non-zero exit marks the Batch job FAILED
        sys.exit(1)
    progress("done", message=f"{result.get('n_iterations', 0)} iterations")
    print(f"[Worker] Done")


if __name__ == "__main__":
    main()
