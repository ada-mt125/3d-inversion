"""Choosing beta (lambda): discrepancy principle vs L-curve vs GCV.

Uses the magnetic synthetic of ``l1l2_paper_synthetic.py`` (Nwosu & Becken
2025 survey, chi = 0.03 SI prism 2-10 km deep) and, for L1–L2 (a = 0.5),
the lp-norm sparse and the smooth L2 regularizations at sigma = 2 and 10 nT:

  1. the worker's default inversion (discrepancy principle, chi^2 = N),
  2. a sweep of fixed-beta inversions over beta_disc * 10^[-2, 2]
     (``beta_selection="gcv"`` through ``run_single_inversion``), which gives
     the L-curve corner and the GCV minimum,
  3. the models at the L-curve and GCV betas,

and reports peak chi, compactness, depth range and misfit for each choice.

Run with:
    python examples/lambda_selection.py [--out DIR]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import l1l2_paper_synthetic as syn
from geoinv3d.cloud.worker import run_fixed_beta, run_single_inversion
from geoinv3d.datamodel.survey import SurveyData
from geoinv3d.methods.magnetics import MagneticsMethod
from geoinv3d.methods.regparam import lcurve_curvature

CASES = [("l1l2", 0.5), ("sparse", None), ("l2", None)]
NOISE_LEVELS = (2.0, 10.0)
CHOICES = ("discrepancy", "lcurve", "gcv")
CHOICE_LABELS = {"discrepancy": "Discrepancy (χ²=N)", "lcurve": "L-curve corner",
                 "gcv": "GCV minimum"}
# Categorical slots 1-3 of the reference palette, fixed order.
CHOICE_COLORS = {"discrepancy": "#2a78d6", "lcurve": "#eb6834", "gcv": "#1baf7a"}
CHOICE_MARKERS = {"discrepancy": "o", "lcurve": "s", "gcv": "^"}


def run(out_dir):
    mesh = syn.build_mesh()
    dmesh = mesh.to_discretize()
    cc, vol = dmesh.cell_centers, dmesh.cell_volumes
    m_true = syn.true_model(cc)
    locs = syn.build_stations()
    survey = SurveyData(locations=locs, observed=np.zeros(len(locs)), std=np.ones(len(locs)))
    sim = MagneticsMethod(inducing_field=(syn.B0, syn.INC, syn.DEC)).make_simulation_full(
        mesh, survey)
    G = np.asarray(sim.G, dtype=float)
    d_clean = G @ m_true
    noise_unit = np.random.default_rng(syn.SEED).standard_normal(len(locs))

    rows, curves, models = [], {}, {}
    for sigma in NOISE_LEVELS:
        dobs = d_clean + sigma * noise_unit
        for reg_type, alpha in CASES:
            label = syn.method_label(reg_type, alpha)
            task = syn.make_task(reg_type, locs, dobs, sigma, alpha)
            task.beta_selection = "gcv"
            t0 = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                res_gcv = run_single_inversion(task, mesh)
                sel = res_gcv["beta_selection"]
                res_lc, _, _ = run_fixed_beta(task, sel["beta_lcurve"], mesh)
            elapsed = time.time() - t0
            curves[(label, sigma)] = sel
            ms = {"discrepancy": np.asarray(res_gcv["discrepancy_model"]),
                  "lcurve": np.asarray(res_lc["recovered_model"]),
                  "gcv": np.asarray(res_gcv["recovered_model"])}
            for choice, m in ms.items():
                models[(label, sigma, choice)] = m
                row = {"method": label, "sigma": sigma, "choice": choice,
                       "beta": sel[f"beta_{choice}"],
                       "beta_over_disc": sel[f"beta_{choice}"] / sel["beta_discrepancy"]}
                row.update(syn.metrics(m, m_true, cc, vol, G, dobs, sigma))
                rows.append(row)
            print(f"σ={sigma:>4g}  {label:<12} β_disc={sel['beta_discrepancy']:.3g}  "
                  f"L-curve ×{sel['beta_lcurve'] / sel['beta_discrepancy']:.3g}  "
                  f"GCV ×{sel['beta_gcv'] / sel['beta_discrepancy']:.3g}  ({elapsed:.0f} s)")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"rows": rows,
                   "curves": {f"{k[0]}|{k[1]:g}": v for k, v in curves.items()}},
                  f, indent=2)
    return {"rows": rows, "curves": curves, "models": models, "m_true": m_true,
            "dmesh": dmesh}


def plot_curves(res, path):
    import matplotlib.pyplot as plt

    labels = [syn.method_label(r, a) for r, a in CASES]
    fig, axes = plt.subplots(3, len(labels), figsize=(4.2 * len(labels), 10))
    styles = {NOISE_LEVELS[0]: "-", NOISE_LEVELS[1]: "--"}
    for j, lab in enumerate(labels):
        for sigma in NOISE_LEVELS:
            sel = res["curves"][(lab, sigma)]
            b = np.asarray(sel["beta"])
            pd, pm, g = (np.asarray(sel[k]) for k in ("phi_d", "phi_m", "gcv"))
            n = sel["n_data"]
            axes[0, j].loglog(pd / n, pm, color="#52514e", ls=styles[sigma], lw=1.5,
                              marker=".", ms=5, label=f"σ = {sigma:g} nT")
            tt, kappa = lcurve_curvature(b, pd, pm)
            axes[1, j].semilogx(np.exp(tt) / sel["beta_discrepancy"], kappa,
                                color="#52514e", ls=styles[sigma], lw=1.5)
            axes[2, j].loglog(b / sel["beta_discrepancy"], g, color="#52514e",
                              ls=styles[sigma], lw=1.5, marker=".", ms=5)
            for choice in CHOICES:
                beta = sel[f"beta_{choice}"]
                i = int(np.argmin(abs(np.log(b / beta))))
                kw = dict(color=CHOICE_COLORS[choice], marker=CHOICE_MARKERS[choice],
                          ms=8, ls="none", mec="#fcfcfb", mew=1)
                axes[0, j].plot(pd[i] / n, pm[i], **kw,
                                label=CHOICE_LABELS[choice] if sigma == NOISE_LEVELS[0]
                                else None)
                axes[2, j].plot(b[i] / sel["beta_discrepancy"], g[i], **kw)
                axes[1, j].axvline(beta / sel["beta_discrepancy"],
                                   color=CHOICE_COLORS[choice], lw=1, ls=styles[sigma],
                                   alpha=0.8)
        axes[0, j].set_title(lab, fontsize=10)
        axes[0, j].set_xlabel("φ_d / N")
        axes[1, j].set_xlabel("β / β_discrepancy")
        axes[2, j].set_xlabel("β / β_discrepancy (nearest sweep point marked)")
        axes[0, j].axvline(1.0, color="#c3c2b7", lw=1, ls=":")
    axes[0, 0].set_ylabel("φ_m")
    axes[1, 0].set_ylabel("L-curve curvature")
    axes[2, 0].set_ylabel("GCV")
    for ax in axes.ravel():
        ax.grid(alpha=0.25, lw=0.5, which="both")
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("β selection on the magnetic synthetic (solid: σ = 2 nT, dashed: 10 nT)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sections(res, sigma, path):
    import matplotlib.pyplot as plt

    labels = [syn.method_label(r, a) for r, a in CASES]
    dmesh = res["dmesh"]
    xe, ze = dmesh.nodes_x / 1e3, dmesh.nodes_z / 1e3
    fig, axes = plt.subplots(len(labels), len(CHOICES), figsize=(9, 2.5 * len(labels) + 0.5),
                             sharex=True, sharey=True, squeeze=False)
    cmap = syn._chi_cmap()
    rows = {(r["method"], r["sigma"], r["choice"]): r for r in res["rows"]}
    for i, lab in enumerate(labels):
        for j, choice in enumerate(CHOICES):
            ax = axes[i, j]
            m = res["models"][(lab, sigma, choice)]
            row = rows[(lab, sigma, choice)]
            vmax = max(float(m.max()), 1e-6)
            im = ax.pcolormesh(xe, ze, syn._section(m, dmesh), cmap=cmap, vmin=0, vmax=vmax)
            syn._draw_body(ax)
            ax.set_title(f"{lab} · {CHOICE_LABELS[choice]}\n"
                         f"β×{row['beta_over_disc']:.2g}, χ²/N {row['chi2_per_datum']:.2f}, "
                         f"peak {vmax:.3f}", fontsize=8)
            ax.set_xlim(0, 50)
            ax.set_ylim(-20, 0)
            ax.tick_params(labelsize=7)
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cb.ax.tick_params(labelsize=6)
        axes[i, 0].set_ylabel("z (km)", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("x (km)", fontsize=8)
    fig.suptitle(f"E–W sections at y = 25 km, σ = {sigma:g} nT (dashed: true prism)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_table(res, out_dir):
    head = ("| σ (nT) | Method | β choice | β/β_disc | Peak χ | V≥10% (km³) | χ·V in prism | "
            "Depth ≥50% (km) | χ²/N | ‖m−m*‖/‖m*‖ |")
    lines = [head, "|" + "---|" * 10]
    for r in res["rows"]:
        lines.append(
            f"| {r['sigma']:g} | {r['method']} | {CHOICE_LABELS[r['choice']]} | "
            f"{r['beta_over_disc']:.2g} | {r['peak_chi']:.4f} | "
            f"{r['anomaly_volume_km3']:.0f} | {r['moment_in_body']:.2f} | "
            f"{r['top_km']:.0f}–{r['bottom_km']:.0f} | {r['chi2_per_datum']:.2f} | "
            f"{r['model_error']:.2f} |")
    table = "\n".join(lines)
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(table + "\n")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output",
                                                      "lambda_selection"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")

    res = run(args.out)
    plot_curves(res, os.path.join(args.out, "curves.png"))
    for sigma in NOISE_LEVELS:
        plot_sections(res, sigma, os.path.join(args.out, f"sections_sigma{sigma:g}.png"))
    print(write_table(res, args.out))
    print(f"outputs in {args.out}")


if __name__ == "__main__":
    main()
