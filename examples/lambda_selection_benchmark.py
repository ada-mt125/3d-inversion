"""Benchmark of regularization-parameter choices: discrepancy, L-curve, GCV.

Survey and mesh of ``l1l2_paper_synthetic.py`` (TMI, B0 = 42000 nT, I = 45,
D = 30 deg, 26 x 26 stations at 2 km height, 2 x 2 x 1 km cells to 20 km) with
three models:

  prism     chi = 0.03 SI, 10 x 10 km, 2-10 km deep
  two       a small shallow block (chi 0.03, 1-4 km deep) and a large deep
            block (chi 0.02, 6-14 km deep)
  slab      chi = 0.03 slab dipping 45 deg from 2 to 12 km depth

and noise sigma = 2, 5, 10, 20 nT, for three regularizations:

  L1–L2     Utsugi (2019) CDA path, wS1 weighting, alpha = 0.8, no bounds
  Sparse    worker lp-norm IRLS (norms 0,2,2,1), chi in [0, 1]
  Smooth    worker smooth L2

Every case sweeps the regularization parameter (the CDA lambda path, or 13
fixed-beta IRLS inversions over beta_disc * 10^[-2, 2]).  The "oracle" is the
swept model closest to the truth; each criterion is scored by its model error
divided by the oracle's (1 = as good as possible).  For L1–L2 the discrepancy
principle is also run with sigma misjudged by x0.5 and x2, from the same path.

Run with:
    python examples/lambda_selection_benchmark.py [--out DIR] [--procs 4] [--quick]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

import l1l2_paper_synthetic as syn  # noqa: E402
from geoinv3d.cloud.worker import (  # noqa: E402
    _run_problem, _selection_point, _single_problem, run_fixed_beta,
)
from geoinv3d.methods.l1l2_cda import (  # noqa: E402
    _discrepancy_lambda, column_scaling, coordinate_descent, elastic_net_path,
)
from geoinv3d.methods.regparam import (  # noqa: E402
    gcv_minimum, gcv_score, lcurve_corner, sparse_terms,
)

MODELS = ("prism", "two", "slab")
NOISE_LEVELS = (2.0, 5.0, 10.0, 20.0)
METHODS = ("L1–L2", "Sparse", "Smooth")
ALPHA, WEIGHTING = 0.8, "S1"
SIGMA_MISJUDGED = (0.5, 2.0)
SWEEP = tuple(np.logspace(-2, 2, 13))
SEED = 7


def model(name: str, cc: np.ndarray) -> np.ndarray:
    x, y, depth = cc[:, 0] / 1e3, cc[:, 1] / 1e3, -cc[:, 2] / 1e3
    if name == "prism":
        return syn.true_model(cc)
    if name == "two":
        shallow = (x > 10) & (x < 16) & (y > 22) & (y < 28) & (depth > 1) & (depth < 4)
        deep = (x > 32) & (x < 42) & (y > 20) & (y < 30) & (depth > 6) & (depth < 14)
        return shallow * 0.03 + deep * 0.02
    if name == "slab":
        axis = 18.0 + (depth - 2.0)  # 45 deg dip towards +x
        return ((abs(x - axis) < 2.0) & (y > 15) & (y < 35) & (depth > 2) & (depth < 12)) * 0.03
    raise ValueError(name)


def setup():
    """Per-process problem: mesh, survey, G, true models, noise pattern."""
    syn.setup_problem()
    syn._P["models"] = {name: model(name, syn._P["cc"]) for name in MODELS}
    syn._P["noise"] = np.random.default_rng(SEED).standard_normal(len(syn._P["locs"]))


def _error(m, m_true):
    return float(np.linalg.norm(m - m_true) / np.linalg.norm(m_true))


def _score(m, m_true, d, sigma):
    G = syn._P["G"]
    inside = m_true > 0
    moment = np.clip(m, 0, None) * syn._P["vol"]
    return {"error": _error(m, m_true),
            "chi2_per_datum": float(np.sum(((G @ m - d) / sigma) ** 2) / len(d)),
            "moment_in_body": float(moment[inside].sum() / max(moment.sum(), 1e-300)),
            "peak_chi": float(m.max())}


def run_cda(m_true, d, sigma):
    """L1–L2 path: all criteria from one path, solved at each chosen lambda."""
    G, H0 = syn._P["G"], syn.H0
    K = G / H0  # uniform sigma: no row weighting, as in the paper
    s = column_scaling(K, WEIGHTING)
    X = np.asfortranarray(K / s)
    to_b = s * H0
    path = elastic_net_path(X, d, ALPHA, with_dof=True)
    n = len(d)
    ok = path.penalty > 1e-6 * path.penalty.max()
    chi2 = (path.residual_norm / sigma) ** 2
    lams = {
        "L-curve": lcurve_corner(path.lambdas[ok], path.residual_norm[ok], path.penalty[ok]),
        "GCV": gcv_minimum(path.lambdas, [gcv_score(r**2, df, n) for r, df in
                                          zip(path.residual_norm, path.dof)]),
        "Discrepancy": _discrepancy_lambda(path.lambdas, chi2, n),
    }
    for c in SIGMA_MISJUDGED:  # assumed sigma = c * true sigma
        lams[f"Discrepancy σ×{c:g}"] = _discrepancy_lambda(path.lambdas, chi2 / c**2, n)
    errors = [_error(b / to_b, m_true) for b in path.betas]
    k_best = int(np.argmin(errors))
    out = {"Oracle": {"lambda": float(path.lambdas[k_best]), "ratio": 1.0,
                      **_score(path.betas[k_best] / to_b, m_true, d, sigma)}}
    for name, lam in lams.items():
        if lam is None:  # chi^2 target not reached on the path
            out[name] = None
            continue
        k = max(int(np.searchsorted(-path.lambdas, -lam, side="right")) - 1, 0)
        b, _ = coordinate_descent(X, d, lam, ALPHA, beta0=path.betas[k])
        sc = _score(b / to_b, m_true, d, sigma)
        out[name] = {"lambda": float(lam), "ratio": sc["error"] / out["Oracle"]["error"], **sc}
    return out


def run_irls(reg_type, m_true, d, sigma):
    """Worker IRLS: discrepancy run, fixed-beta sweep (common eps), L-curve, GCV."""
    task = syn.make_task(reg_type, syn._P["locs"], d, sigma)
    mesh = syn._P["mesh"]
    with contextlib.redirect_stdout(io.StringIO()):
        p = _single_problem(task, mesh)
        base, _ = _run_problem(task, p)
        beta_disc = base["iterations"][-1]["beta"]
        eps = [float(o.irls_threshold) for o in sparse_terms(p.reg)] \
            if p.kind not in ("smooth", "l2") else None
        points, models = [], []
        for f in SWEEP:
            res, pf, inv_prob = run_fixed_beta(task, beta_disc * f, mesh, sim=p.sim,
                                               irls_thresholds=eps)
            m = np.asarray(res["recovered_model"])
            points.append(_selection_point(pf, inv_prob, m))
            models.append(m)
        betas = np.array([pt["beta"] for pt in points])
        phi_d = np.array([pt["phi_d"] for pt in points])
        phi_m = np.array([pt["phi_m"] for pt in points])
        lams = {"L-curve": lcurve_corner(betas, phi_d, phi_m),
                "GCV": gcv_minimum(betas, [pt["gcv"] for pt in points])}
        chosen = {name: np.asarray(run_fixed_beta(task, lam, mesh, sim=p.sim,
                                                  irls_thresholds=eps)[0]["recovered_model"])
                  for name, lam in lams.items()}
    errors = [_error(m, m_true) for m in models + [np.asarray(base["recovered_model"])]]
    k_best = int(np.argmin(errors))
    best_m = (models + [np.asarray(base["recovered_model"])])[k_best]
    best_beta = float(betas[k_best]) if k_best < len(betas) else float(beta_disc)
    out = {"Oracle": {"lambda": best_beta, "ratio": 1.0,
                      **_score(best_m, m_true, d, sigma)}}
    sc = _score(np.asarray(base["recovered_model"]), m_true, d, sigma)
    out["Discrepancy"] = {"lambda": float(beta_disc),
                          "ratio": sc["error"] / out["Oracle"]["error"], **sc}
    for name, lam in lams.items():
        sc = _score(chosen[name], m_true, d, sigma)
        out[name] = {"lambda": float(lam), "ratio": sc["error"] / out["Oracle"]["error"], **sc}
    return out


def run_case(job):
    method, name, sigma = job
    m_true = syn._P["models"][name]
    d = syn._P["G"] @ m_true + sigma * syn._P["noise"]
    t0 = time.time()
    if method == "L1–L2":
        out = run_cda(m_true, d, sigma)
    else:
        out = run_irls({"Sparse": "sparse", "Smooth": "l2"}[method], m_true, d, sigma)
    return {"method": method, "model": name, "sigma": sigma,
            "runtime_s": time.time() - t0, "criteria": out}


CRITERIA = ("Discrepancy", "L-curve", "GCV", "Discrepancy σ×0.5", "Discrepancy σ×2")
# Categorical slots 1-3 of the reference palette, fixed order.
METHOD_COLORS = {"L1–L2": "#2a78d6", "Sparse": "#eb6834", "Smooth": "#1baf7a"}
METHOD_MARKERS = {"L1–L2": "o", "Sparse": "s", "Smooth": "^"}


def plot(rows, path):
    """Error ratio to the oracle per criterion; one dot per model and noise level."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.2))
    offsets = {"L1–L2": -0.22, "Sparse": 0.0, "Smooth": 0.22}
    jitter = np.linspace(-0.07, 0.07, len(MODELS) * len(NOISE_LEVELS))
    for method in METHODS:
        for i, crit in enumerate(CRITERIA):
            vals = [r["criteria"].get(crit) for r in rows if r["method"] == method]
            vals = [v["ratio"] for v in vals if v]
            if not vals:
                continue
            x = i + offsets[method] + jitter[:len(vals)]
            ax.plot(x, vals, METHOD_MARKERS[method], color=METHOD_COLORS[method], ms=6,
                    alpha=0.8, mec="#fcfcfb", mew=0.5,
                    label=method if i == 0 else None)
            ax.plot([i + offsets[method] - 0.09, i + offsets[method] + 0.09],
                    [np.median(vals)] * 2, color="#0b0b0b", lw=2)
    ax.axhline(1.0, color="#52514e", lw=1, ls=":")
    ax.set_xticks(range(len(CRITERIA)))
    ax.set_xticklabels(CRITERIA, fontsize=9)
    ax.set_ylabel("model error ÷ best achievable")
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("Regularization-parameter choice: 3 models × 4 noise levels per method\n"
                 "(bars: medians; σ-misjudged runs for L1–L2 only; cases where χ² = N is "
                 "never reached are left out)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_tables(rows, out_dir):
    lines = ["## Summary: model error ÷ best achievable (median, worst; "
             "cases > 1.25)", "",
             "| Method | " + " | ".join(CRITERIA) + " |", "|---|" + "---|" * len(CRITERIA)]
    summary = {}
    for method in METHODS:
        cells = []
        for crit in CRITERIA:
            vals = [r["criteria"].get(crit) for r in rows if r["method"] == method]
            vals = [v["ratio"] for v in vals if v]
            if not vals:
                cells.append("–")
                continue
            summary[(method, crit)] = vals
            cells.append(f"{np.median(vals):.2f}, {max(vals):.2f}; "
                         f"{sum(v > 1.25 for v in vals)}/{len(vals)}")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    for key, title in (("moment_in_body", "share of χ·V inside the true bodies"),
                       ("chi2_per_datum", "χ²/N")):
        lines += ["", f"## Median {title}", "",
                  "| Method | Oracle | " + " | ".join(CRITERIA) + " |",
                  "|---|---|" + "---|" * len(CRITERIA)]
        for method in METHODS:
            cells = []
            for crit in ("Oracle",) + CRITERIA:
                vals = [r["criteria"].get(crit) for r in rows if r["method"] == method]
                vals = [v[key] for v in vals if v]
                cells.append(f"{np.median(vals):.2f}" if vals else "–")
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += ["", "## Cases", "",
              "| Method | Model | σ (nT) | Criterion | λ or β | error ratio | "
              "‖m−m*‖/‖m*‖ | χ²/N | χ·V in bodies | peak χ |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for crit in ("Oracle",) + CRITERIA:
            v = r["criteria"].get(crit)
            if crit not in r["criteria"]:
                continue
            if v is None:
                lines.append(f"| {r['method']} | {r['model']} | {r['sigma']:g} | {crit} | "
                             "χ²=N not reached | | | | | |")
                continue
            lines.append(
                f"| {r['method']} | {r['model']} | {r['sigma']:g} | {crit} | "
                f"{v['lambda']:.3g} | {v['ratio']:.2f} | {v['error']:.3f} | "
                f"{v['chi2_per_datum']:.2f} | {v['moment_in_body']:.2f} | "
                f"{v['peak_chi']:.4f} |")
    table = "\n".join(lines)
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(table + "\n")
    return "\n".join(lines[:lines.index("## Cases")])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output",
                                                      "lambda_selection_benchmark"))
    parser.add_argument("--procs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--quick", action="store_true", help="prism, sigma = 2 and 10")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")

    models = ("prism",) if args.quick else MODELS
    sigmas = (2.0, 10.0) if args.quick else NOISE_LEVELS
    jobs = [(meth, name, s) for meth in METHODS for name in models for s in sigmas]
    jobs.sort(key=lambda j: j[0] == "L1–L2")  # the slower IRLS sweeps first
    rows = []
    t0 = time.time()
    # forkserver: thread pools started while computing G are not fork-safe
    with get_context("forkserver").Pool(args.procs, initializer=setup) as pool:
        for row in pool.imap_unordered(run_case, jobs):
            rows.append(row)
            c = row["criteria"]
            summary = "  ".join(
                f"{k}={c[k]['ratio']:.2f}" if c.get(k) else f"{k}=n/a"
                for k in CRITERIA if k in c)
            print(f"[{len(rows):>2}/{len(jobs)}] {row['method']:<6} {row['model']:<5} "
                  f"σ={row['sigma']:<4g} {summary} ({row['runtime_s']:.0f} s)", flush=True)
    print(f"total {time.time() - t0:.0f} s")
    order = {m: i for i, m in enumerate(METHODS)}
    rows.sort(key=lambda r: (order[r["method"]], MODELS.index(r["model"]), r["sigma"]))
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    plot(rows, os.path.join(args.out, "error_ratio.png"))
    print(write_tables(rows, args.out))
    print(f"outputs in {args.out}")


if __name__ == "__main__":
    main()
