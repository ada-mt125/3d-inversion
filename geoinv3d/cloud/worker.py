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

Usage (on the cloud instance):
    python -m geoinv3d.cloud.worker
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .task import InversionTask, unpack_task, pack_task


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
            m0 = np.zeros(n_active)

    if kind == "smooth":
        reg_kwargs = dict(
            alpha_s=task.alpha_s,
            alpha_x=task.alpha_x,
            alpha_y=task.alpha_y,
            alpha_z=task.alpha_z,
        )
    elif kind == "l1l2":
        reg_kwargs = dict(l1_ratio=task.l1_ratio, reference_model=np.zeros_like(m0))
    else:
        reg_kwargs = dict(
            alpha_s=task.alpha_s,
            length_scale_x=task.alpha_x,
            length_scale_y=task.alpha_y,
            length_scale_z=task.alpha_z,
            reference_model=np.zeros_like(m0),
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
    reg = {
        "smooth": regularization.WeightedLeastSquares,
        "l2": regularization.WeightedLeastSquares,
        "l1l2": ElasticNet,
        "sparse": regularization.Sparse,
        "mgs": Focusing,
        "tv": Focusing,
    }[kind](dmesh, **reg_kwargs)

    lo = hi = None
    if kind == "smooth":
        opt = optimization.InexactGaussNewton(maxIter=task.max_iter, cg_maxiter=20)
    elif task.bounds_lower is not None or task.bounds_upper is not None:
        lo = task.bounds_lower if task.bounds_lower is not None else -np.inf
        hi = task.bounds_upper if task.bounds_upper is not None else np.inf
        # A start exactly on a bound (e.g. m0 = 0 with lower = 0) leaves every
        # cell in ProjectedGNCG's active set and the model never moves, so
        # start just inside the bounds.
        span = hi - lo
        nudge = 1e-4 * span if np.isfinite(span) else 1e-4
        m0 = np.clip(m0, lo + nudge, hi - nudge)
        opt = optimization.ProjectedGNCG(
            maxIter=task.max_iter, lower=lo, upper=hi,
            maxIterLS=20, maxIterCG=30, tolCG=1e-4,
        )
    else:
        opt = optimization.InexactGaussNewton(
            maxIter=task.max_iter, maxIterLS=20, maxIterCG=30, tolCG=1e-4,
        )
    return SimpleNamespace(kind=kind, sim=sim, dmis=dmis, reg=reg, opt=opt, m0=m0,
                           lo=lo, hi=hi)


def _regularization_label(kind: str) -> str:
    return {"smooth": "smooth_L2", "l2": "smooth_L2", "sparse": "sparse_IRLS",
            "l1l2": "elastic_net_IRLS",
            "mgs": "focusing_MGS", "tv": "total_variation"}[kind]


def _finish_result(task, p, collector, m_recovered, inv_prob) -> dict:
    result = _collect_result(task, collector, m_recovered, inv_prob,
                             _regularization_label(p.kind))
    if p.kind == "l1l2":
        result["l1_ratio"] = task.l1_ratio
        result.pop("norms")
    elif p.kind == "l2":
        result.pop("norms")
        result["depth_weighting"] = "sensitivity"
    elif p.kind in ("mgs", "tv"):
        result.pop("norms")
        result["focusing_threshold"] = p.reg.focusing_threshold
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
        directive_list = [directives.UpdateSensitivityWeights()]
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
                   else directives.UpdateSensitivityWeights())
        if beta is None:
            directive_list = [
                weights,
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
                weights,
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
        if p.lo is not None:
            span = p.hi - p.lo if np.isfinite(p.hi - p.lo) else 1.0
            tol = 1e-8 * span
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

    return {
        "task_id": task.task_id,
        "method": task.method_type,
        "regularization": method_label,
        "converged": True,
        "n_iterations": len(iterations),
        "iterations": iterations,
        "recovered_model": m_recovered,
        "norms": list(task.norms),
    }


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
    res, scale = invert_l1l2(
        np.asarray(G), task.observed_data, task.l1_ratio,
        weighting=task.l1l2_weighting, model_unit=model_unit, std=task.data_std,
        criterion=criterion, lower=task.bounds_lower, upper=task.bounds_upper,
        n_decades=task.lambda_decades, step=task.lambda_step,
        fallback=task.beta_selection == "auto",
    )
    path = res.path
    chi2 = res.chi2
    iterations = []
    for i, (lam, beta) in enumerate(zip(path.lambdas, path.betas)):
        m = beta / (scale * model_unit)
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
    resid = (np.asarray(G) @ res.model - task.observed_data) / task.data_std
    for w in res.warnings:
        print(f"[L1–L2 CDA] Warning: {w}")
    return {
        "task_id": task.task_id,
        "method": task.method_type,
        "regularization": "elastic_net_CDA",
        "converged": True,
        "n_iterations": len(iterations),
        "iterations": iterations,
        "recovered_model": res.model,
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


def run_joint_inversion(task: InversionTask, mesh=None) -> dict:
    """Execute a joint inversion from a task specification.

    Each method gets an L2 (WeightedLeastSquares) regularization using the
    task's alpha_s and, as in run_sparse_inversion, alpha_x/y/z as length
    scales.  task.cross_gradient_weight > 0 couples the models structurally.
    """
    from ..datamodel.survey import SurveyData
    from ..methods.joint import JointInversion, MethodSetup

    if mesh is None:
        mesh = _build_mesh(task)
    active = task.active_cells
    n_params = int(active.sum()) if active is not None else mesh.n_cells

    setups = []
    for i, method_name in enumerate(task.joint_methods):
        method_name = canonical_method(method_name)
        kwargs = (task.joint_kwargs_list or [{}])[i] if task.joint_kwargs_list else {}
        method = _make_method(method_name, kwargs)

        survey_data = task.joint_surveys[i]
        survey = SurveyData(
            locations=survey_data["locations"],
            observed=survey_data["observed"],
            std=survey_data["std"],
            method=method_name,
        )

        weight = (task.joint_weights or [1.0])[i] if task.joint_weights else 1.0
        initial = task.joint_initial_models[i] if task.joint_initial_models else None
        if initial is None or len(initial) != n_params:
            initial = np.zeros(n_params)

        setups.append(MethodSetup(
            method=method,
            survey=survey,
            mesh=mesh,
            initial_model=initial,
            weight=weight,
            active_cells=active,
        ))

    joint = JointInversion(
        setups=setups,
        max_iter=task.max_iter,
        beta0_ratio=task.beta0_ratio,
        cooling_factor=task.cooling_factor,
        cross_gradient_weight=task.cross_gradient_weight,
        reg_kwargs=dict(
            alpha_s=task.alpha_s,
            length_scale_x=task.alpha_x,
            length_scale_y=task.alpha_y,
            length_scale_z=task.alpha_z,
        ),
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

    return {
        "task_id": task.task_id,
        "methods": [canonical_method(m) for m in task.joint_methods],
        "regularization": "joint_L2",
        "cross_gradient_weight": task.cross_gradient_weight,
        "converged": result.converged,
        "n_iterations": len(iterations),
        "iterations": iterations,
        "recovered_models": {
            k: v for k, v in result.recovered_models.items()
        },
    }


def execute_task(task: InversionTask, mesh=None) -> dict:
    """Dispatch to single or joint inversion based on task spec.

    Args:
        mesh: Optional prebuilt mesh (Mesh3D or DiscretizeMesh); by default
            it is built from the task's mesh fields.
    """
    if task.joint_methods:
        return run_joint_inversion(task, mesh)
    return run_single_inversion(task, mesh)


# ── Raw-data pipeline ───────────────────────────────────────────────────

_GRID_FORMATS = ("geotiff", "surfer_grd", "esri_ascii")
_DATA_EXTENSIONS = (".tif", ".tiff", ".grd", ".asc", ".csv", ".txt", ".dat", ".xyz",
                    ".obs", ".npy", ".npz")
# Methods whose surveys can be built from station/grid files alone
_PIPELINE_METHODS = ("gravity", "magnetics")
_DEFAULT_COMPONENT = {"gravity": "gz", "magnetics": "tmi"}
# Keys the upload page only sends in manual mode; auto mode ignores them.
# (Iteration limits are editable in both modes.)
_MANUAL_KEYS = (
    "norms", "alpha_s", "alpha_x", "alpha_y", "alpha_z", "beta0_ratio",
    "cooling_factor", "use_preconditioner",
    "bounds_lower", "bounds_upper", "joint_weights", "cross_gradient_weight",
    "l1_ratio", "beta_selection", "beta_sweep",
    "l1l2_solver", "l1l2_weighting", "lambda_decades", "lambda_step",
    "focusing_percentile", "focusing_scale",
)
# Warn when more than this share of the recovered anomaly lies in padding cells
PADDING_WARNING_SHARE = 0.5

# params key -> InversionTask field for the regularization/optimizer settings
_REG_PARAM_KEYS = (
    "max_iter", "beta0_ratio", "cooling_factor",
    "alpha_s", "alpha_x", "alpha_y", "alpha_z",
    "irls_cooling_factor", "max_irls_iterations", "use_preconditioner",
    "bounds_lower", "bounds_upper", "l1_ratio", "beta_selection", "beta_sweep",
    "l1l2_solver", "l1l2_weighting", "lambda_decades", "lambda_step",
    "focusing_percentile", "focusing_scale",
)


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
    spacing: float | None = None      # data spacing (m) of the finest file
    spacing_kind: str = ""            # "grid" or "points"


def _read_observations(path: str, component: str, stride: int, aoi) -> tuple:
    """Read one data file -> (locations (n, 3), values (n,), metadata).

    Gridded files give stations at the grid nodes (decimated by `stride`)
    with unknown elevation (z = NaN).  Point files keep their z if present.
    """
    from ..io.readers import GridData, detect_format, read_auto, read_station_table

    name = os.path.basename(path)
    fmt = detect_format(path)
    if fmt in _GRID_FORMATS:
        grid = read_auto(path)
        x, y, values = grid.x, grid.y, grid.values
        if aoi:
            west, east, south, north = aoi
            mask_x = (x >= west) & (x <= east)
            mask_y = (y >= south) & (y <= north)
            values = values[np.ix_(mask_y, mask_x)]
            x, y = x[mask_x], y[mask_y]
        x, y, values = x[::stride], y[::stride], values[::stride, ::stride]
        xx, yy = np.meshgrid(x, y)
        locs = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, np.nan)])
        steps = [abs(float(a[1] - a[0])) for a in (x, y) if len(a) > 1]
        meta = dict(grid.metadata)
        if steps:
            meta["grid_spacing"] = max(steps)
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
    if aoi:
        west, east, south, north = aoi
        keep = ((locs[:, 0] >= west) & (locs[:, 0] <= east)
                & (locs[:, 1] >= south) & (locs[:, 1] <= north))
        locs, values = locs[keep], values[keep]
    return locs, values, points.metadata


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


def _load_dataset(spec: dict, params: dict, data_dir: str, single: bool) -> PipelineDataset:
    """Load all files of one dataset spec and attach its noise model."""
    method = canonical_method(spec.get("method") or spec.get("type")
                              or params.get("method_type", "gravity"))
    if method not in _PIPELINE_METHODS:
        raise NotImplementedError(
            f"The raw-data pipeline supports {', '.join(_PIPELINE_METHODS)}; "
            f"'{method}' data also needs electrode/frequency metadata that the upload "
            f"format does not carry yet"
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
    locs_all, values_all, spacings = [], [], []
    for fname in files:
        path = os.path.join(data_dir, fname)
        print(f"[Pipeline] Loading {method} ({component}) data from {fname}")
        locs, values, meta = _read_observations(path, component, stride, aoi)
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

    noise_pct = float(spec.get("noise_pct", params.get("noise_pct", 0.05)))
    noise_floor = float(spec.get("noise_floor", params.get("noise_floor", 0.5)))
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
        spacing=spacing, spacing_kind=spacing_kind,
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


def _load_topography(params: dict, data_dir: str) -> tuple:
    """Return (surface(x, y) -> z, info dict, has_dem)."""
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
            surface = _grid_surface(read_auto(path))
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
            surface = _point_surface(xyz)
        return surface, {"source": "dem", "file": fname}, True

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


def _active_below_surface(dmesh, surface) -> np.ndarray:
    """Cells whose centres lie below the ground surface."""
    cc = dmesh.cell_centers
    return cc[:, 2] < surface(cc[:, 0], cc[:, 1])


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
            topography: {file: DEM name} or {flat_elevation: metres}.
                Stations without an elevation are placed on the surface
                (+ station_height, default 0 m); with a DEM, cells above
                the surface are inactive.
            param_mode: "auto" (defaults; manual-only keys are ignored) or
                "manual" (norms, alpha_*, beta0_ratio, cooling_factor,
                max_iter, max_irls_iterations, use_preconditioner, bounds_*).
            regularization_type: "sparse" (lp-norm IRLS), "l1l2" (L1–L2
                elastic net, Utsugi 2019; mixing set by l1_ratio), "mgs"
                (minimum gradient support focusing), "tv" (total variation)
                or "l2" (smooth L2 with depth weighting).  focusing_percentile / focusing_scale set the
                MGS/TV focusing parameter.
            beta_selection: "auto" (default), "discrepancy", "lcurve" or
                "gcv" (manual mode, single inversions); beta_sweep optionally
                lists the betas to try.
            l1l2_solver: "cda" (default; Utsugi 2019 lambda path with the
                L-curve) or "irls"; l1l2_weighting "S2" (default) or "S1";
                lambda_decades, lambda_step set the lambda path.
            joint_weights, cross_gradient_weight: joint data weights and
                structural coupling (auto: all 1).
            mesh_type: "tensor" (default) or "octree"; plus core_cell_m,
                core_cell_z_m, depth_core_m, pad_distance_m, octree_levels,
                decimate_stride, aoi.
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

    datasets = [_load_dataset(s, params, data_dir, single=(mode == "single")) for s in specs]

    # ── Topography and station elevations ──
    surface, topo_info, has_dem = _load_topography(params, data_dir)
    station_height = float(params.get("station_height", 0.0))
    for ds in datasets:
        missing = np.isnan(ds.locations[:, 2])
        if missing.any():
            ds.locations[missing, 2] = (
                surface(ds.locations[missing, 0], ds.locations[missing, 1]) + station_height
            )

    report("building_mesh")

    # ── Mesh over the union of all survey extents ──
    all_locs = np.vstack([ds.locations for ds in datasets])
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
    print(f"[Pipeline] {mesh_type} mesh: {mesh.n_cells} cells "
          f"({n_params} active), {n_data} observations, "
          f"topography: {topo_info['source']}")

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
        xgrad = float(params.get("cross_gradient_weight", 1.0))
        task = InversionTask(
            method_type="joint",
            joint_methods=[ds.method for ds in datasets],
            joint_kwargs_list=[ds.method_kwargs for ds in datasets],
            joint_weights=weights,
            cross_gradient_weight=xgrad,
            joint_surveys=[
                {"locations": ds.locations, "observed": ds.observed, "std": ds.std}
                for ds in datasets
            ],
            **common,
        )
        if task.regularization_type == "l1l2":
            notes.append("Joint inversion uses L2 regularization; the L1–L2 "
                         "regularization is not applied")
        elif task.regularization_type in ("mgs", "tv"):
            notes.append("Joint inversion uses L2 regularization; MGS/TV focusing is "
                         "not applied")
        elif task.regularization_type == "sparse" and tuple(task.norms) != (2.0,) * 4:
            notes.append("Joint inversion uses L2 regularization; norms/IRLS settings "
                         "are not applied")
        if task.bounds_lower is not None or task.bounds_upper is not None:
            notes.append("Joint inversion does not apply bounds")
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
            initial_model=np.zeros(n_params),
            observed_data=ds.observed,
            data_std=ds.std,
            station_locations=ds.locations,
            **common,
        )
        if task.regularization_type == "l1l2":
            notes.append("L1–L2 regularizes the model values only (no smoothness "
                         "term); norms and alpha_x/y/z are not used")
    for note in notes:
        print(f"[Pipeline] Note: {note}")

    report("inverting", max_iter=task.max_iter, phi_d_target=n_data, n_data=n_data,
           n_cells=n_params, regularization=task.regularization_type)
    IterationCollector.on_iteration = lambda snap: report(
        "inverting", iteration=snap.iteration, phi_d=snap.phi_d, beta=snap.beta)
    try:
        result = execute_task(task, mesh=mesh)
    finally:
        IterationCollector.on_iteration = None

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
        "datasets": [
            {"method": ds.method, "component": ds.component, "files": ds.files,
             "n_data": int(ds.observed.size), "noise_pct": ds.noise_pct,
             "noise_floor": ds.noise_floor}
            for ds in datasets
        ],
    })
    if active is not None:
        result["active_cells"] = active

    # How much of the recovered anomaly sits outside the core (in padding)?
    gx, gy = np.meshgrid(np.linspace(extent[0], extent[1], 20),
                         np.linspace(extent[2], extent[3], 20))
    z_bottom = float(np.min(surface(gx, gy))) - depth_core_m
    models = result.get("recovered_models") or (
        {task.method_type: result["recovered_model"]} if "recovered_model" in result else {})
    shares = {
        name: _outside_core_share(dmesh, active, m, extent, core_cell_m, z_bottom)
        for name, m in models.items() if np.size(m) == n_params
    }
    if shares:
        result["outside_core_share"] = shares
        for name, share in shares.items():
            if share > PADDING_WARNING_SHARE:
                notes.append(
                    f"{share:.0%} of the recovered {name} anomaly lies outside the core "
                    f"mesh (padding cells), where it is poorly constrained. Consider a "
                    f"compact regularization (L1–L2), property bounds, or a larger "
                    f"core (depth_core_m)."
                )
    if notes:
        result["notes"] = notes
    return result


def result_metadata_json(result: dict) -> str:
    """result.json: everything in a result except the model arrays."""
    meta = {k: v for k, v in result.items()
            if k not in ("recovered_model", "recovered_models", "active_cells")}
    return json.dumps(_jsonable(meta), indent=2, default=str)


class S3Progress:
    """Progress report the worker keeps in S3 (``progress.json``) for the API.

    Called as ``progress(stage, message="", **fields)``; fields persist
    between calls.  Stages: starting, downloading, loading_data,
    building_mesh, inverting, uploading, done, failed.  The report is written
    on every stage change and otherwise at most every ``min_interval``
    seconds, so per-iteration updates stay cheap.  Failures to write are
    printed, never raised.
    """

    def __init__(self, s3, bucket: str, key: str, min_interval: float = 10.0,
                 clock=time.time) -> None:
        self.s3, self.bucket, self.key = s3, bucket, key
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
                self.s3.put_object(Bucket=self.bucket, Key=self.key,
                                   Body=json.dumps(_jsonable(self.state)).encode(),
                                   ContentType="application/json")
            except Exception as e:
                print(f"[Worker] Could not update progress: {e}")


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
    """Entry point for AWS Batch container."""
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
