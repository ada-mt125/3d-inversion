"""Focusing (MGS), total variation, L1–L2, sparse and L2 on the magnetic synthetic.

Same survey and prism as ``l1l2_paper_synthetic.py`` (Nwosu & Becken 2025
survey, chi = 0.03 SI prism 2-10 km deep).  Runs, through the worker
(``run_single_inversion``, discrepancy principle chi^2 = N, positivity):

  * MGS: minimum gradient support focusing (Portniaguine & Zhdanov 1999),
    with the loose bound chi <= 1 and with an a-priori bound chi <= 0.04
  * TV: total variation of the isotropic gradient
  * Sparse lp (worker defaults) and smooth L2

and adds the best-alpha L1–L2 wS1 model (Utsugi 2019 CDA, lambda from the
L-curve, no bounds) from ``l1l2_paper_synthetic.py``'s output (run that
first).

Run with:
    python examples/focusing_comparison.py [--out DIR] [--l1l2 DIR]
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
from geoinv3d.cloud.worker import run_single_inversion

# (regularization_type, label, upper bound on chi; None = unbounded)
METHODS = (("mgs", "MGS", 1.0), ("mgs", "MGS χ≤0.04", 0.04), ("tv", "Total variation", 1.0),
           ("sparse", "Sparse lp", 1.0), ("l2", "Smooth L2", None))


def run_method(reg_type, sigma, upper):
    d = syn.observed(sigma)
    task = syn.make_task(reg_type, syn._P["locs"], d, sigma)
    if upper is not None:
        task.bounds_lower, task.bounds_upper = 0.0, upper
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        result = run_single_inversion(task, syn._P["mesh"])
    m = np.asarray(result["recovered_model"], dtype=float)
    row = {"sigma": sigma, "runtime_s": time.time() - t0,
           "n_iterations": result["n_iterations"],
           "focusing_threshold": result.get("focusing_threshold")}
    row.update(syn.metrics(m, syn._P["m_true"], syn._P["cc"], syn._P["vol"], syn._P["G"], d,
                           sigma))
    return row, m


def main():
    here = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=os.path.join(here, "output", "focusing_comparison"))
    parser.add_argument("--l1l2", default=os.path.join(here, "output", "l1l2_paper_synthetic"),
                        help="output directory of l1l2_paper_synthetic.py")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")

    syn.setup_problem()
    rows, models = [], {}
    for sigma in syn.NOISE_LEVELS:
        for reg_type, label, upper in METHODS:
            row, m = run_method(reg_type, sigma, upper)
            row["label"] = label
            rows.append(row)
            models[(label, sigma)] = m
            print(f"σ={sigma:<4g} {label:<16} peak={row['peak_chi']:.4f} "
                  f"V10={row['anomaly_volume_km3']:6.0f} in-body={row['moment_in_body']:.2f} "
                  f"depth {row['top_km']:.0f}-{row['bottom_km']:.0f} km "
                  f"χ²/N={row['chi2_per_datum']:.2f} ε={row['eps_Am']:.2f} "
                  f"({row['runtime_s']:.0f} s)", flush=True)

    # Best-alpha L1–L2 models from the paper-protocol run
    l1l2_labels = []
    with open(os.path.join(args.l1l2, "results.json")) as f:
        l1l2 = json.load(f)
    stored = np.load(os.path.join(args.l1l2, "models.npz"))
    for w, alpha in l1l2["best_alpha"].items():
        if w != "S1":  # wS2 is discussed in the L1–L2 report
            continue
        label = f"L1–L2 w{w} α={alpha:g}"
        l1l2_labels.append(label)
        for sigma in syn.NOISE_LEVELS:
            m = stored[f"{w}|{alpha:g}|{sigma:g}"]
            row = {"label": label, "sigma": sigma}
            row.update(syn.metrics(m, syn._P["m_true"], syn._P["cc"], syn._P["vol"],
                                   syn._P["G"], syn.observed(sigma), sigma))
            rows.append(row)
            models[(label, sigma)] = m
    labels = l1l2_labels + [label for _, label, _ in METHODS]

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    head = ("| σ (nT) | Method | Peak χ | V≥10% (km³) | χ·V in prism | Depth ≥50% (km) | "
            "χ²/N | ε (A/m) | neg. share |")
    lines = [head, "|" + "---|" * 9]
    for sigma in syn.NOISE_LEVELS:
        for lab in labels:
            r = next(r for r in rows if r["label"] == lab and r["sigma"] == sigma)
            lines.append(syn._row_md(r, lab))
    table = "\n".join(lines)
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(table + "\n")

    dmesh = syn._P["dmesh"]
    panels = [(f"{lab}\nσ={s:g} nT", models[(lab, s)]) for s in syn.NOISE_LEVELS
              for lab in labels]
    syn.plot_sections(panels, dmesh, os.path.join(args.out, "sections.png"),
                      "E–W sections at y = 25 km (dashed: true prism)", ncol=len(labels))
    # Categorical slots 1-6 of the reference palette, fixed order
    syn.SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    syn.MARKERS = ["o", "s", "^", "D", "v", "P"]
    syn.plot_metrics(rows, labels, os.path.join(args.out, "metrics.png"))
    print(table)
    print(f"outputs in {args.out}")


if __name__ == "__main__":
    main()
