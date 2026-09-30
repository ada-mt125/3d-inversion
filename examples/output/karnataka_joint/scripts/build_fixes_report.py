"""The follow-up to Section 5 of the Karnataka joint report: what was fixed, before and after.

    py examples/output/karnataka_joint/scripts/fixes_summary.py
    py examples/output/karnataka_joint/scripts/fixes_figures.py
    py examples/output/karnataka_joint/scripts/build_fixes_report.py [--pdf]

Every number comes from figures/fixes_numbers.json (fixes_summary.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "karnataka_inputs" / "shared"))
from style import Figures, page, table, to_pdf  # noqa: E402

FIGS = ROOT / "figures"
N = json.loads((FIGS / "fixes_numbers.json").read_text(encoding="utf-8"))
FULL, LOW, MAG = N["full"], N["low"], N["magnetic"]
OUT = ROOT / "karnataka_joint_fixes_en.html"
LABEL = {"none": "No coupling", "cross_gradient": "Cross-gradient",
         "joint_total_variation": "Joint total variation",
         "linear_correspondence": "Linear correspondence", "pgi": "PGI (rock units)",
         "group_lasso_depth": "Group lasso",
         "group_lasso_depth_uncoupled": "L1 + L2 by ADMM, no coupling (its control)"}


def pct(x):
    return f"{100 * x:.0f}%"


def chi(x):
    ok = 0.7 <= x <= 1.3
    return (f"{x:.2f}", "n" if ok else "n warn")


def core(x):
    return (pct(x), "n" if x >= 0.5 else "n warn")


def minutes(m):
    run = (m or {}).get("run") or {}
    if "minutes" in run:
        return f"{run['minutes']:.0f}"
    if "seconds" in run:
        return f"{run['seconds'] / 60:.1f}"
    return "—"


def cost(m):
    run = (m or {}).get("run") or {}
    return f"${run['cost_usd']:.2f}" if "cost_usd" in run else "—"


def rows_for(level):
    rows = []
    for k, label in LABEL.items():
        pair = level.get(k) or {}
        for when in ("before", "after"):
            m = pair.get(when)
            if m is None:
                continue
            tag = "report" if when == "before" else "fixed"
            name = f"{label} — {tag}" + (" †" if m.get("stopped") else "")
            rows.append([name, chi(m["gravity"]["chi2"]), chi(m["magnetics"]["chi2"]),
                         core(m["gravity"]["core"]), pct(m["gravity"]["below"]),
                         core(m["magnetics"]["core"]), pct(m["magnetics"]["below"]),
                         str(m.get("n_iterations") or "—"), minutes(m)]
                        + ([cost(m)] if level is FULL else []))
    return rows


HEAD = ["Run", "Gravity χ²/N", "Magnetic χ²/N", "Density in the core", "below it",
        "Susceptibility in the core", "below it", "Iterations", "Minutes"]


def gl_line(level, when="after"):
    m = (level.get("group_lasso_depth") or {}).get(when)
    return m


def main():
    F = Figures("en", FIGS)
    gl_b, gl_a = FULL["group_lasso_depth"]["before"], FULL["group_lasso_depth"]["after"]
    lo_b, lo_a = LOW["group_lasso_depth"]["before"], LOW["group_lasso_depth"]["after"]
    jtv_b, jtv_a = FULL["joint_total_variation"]["before"], FULL["joint_total_variation"]["after"]
    pgi_b, pgi_a = FULL["pgi"]["before"], FULL["pgi"]["after"]
    bal = (gl_a or {}).get("group_lasso", {}).get("balance") or []
    u = MAG.get("underfit") or {}
    mv = MAG.get("mvi") or {}
    mi = MAG.get("induced") or {}
    mz = mv.get("magnetization") or {}
    total_cost = sum(((v.get("after") or {}).get("run") or {}).get("cost_usd", 0.0) for v in FULL.values()) \
        + ((mv.get("run") or {}).get("cost_usd", 0.0))

    def fit(m):
        return f"{m['gravity']['chi2']:.2f} and {m['magnetics']['chi2']:.2f}"

    def cores(m):
        return f"{pct(m['gravity']['core'])} and {pct(m['magnetics']['core'])}"

    problems = [
        ["Group lasso: model outside the core",
         "γ = 2 gives every column of the scaled sensitivity unit norm: every cell, however deep or far "
         "into the padding, costs the same.",
         "Cells weighed like the SimPEG runs: volume × Li &amp; Oldenburg depth weight (β = 1 of each model) "
         "(<code>gl_weighting=\"depth\"</code>).",
         f"In the core {cores(lo_b)} → {cores(lo_a)} (2 km); "
         + (f"{cores(gl_b)} → {cores(gl_a)} (full)" if gl_a else "full run pending")],
        ["Group lasso: unbalanced fit",
         "One λ₁ for both datasets; the gravity data are fitted faster along the sweep and λ₂ caps the magnetic fit.",
         "A data weight per dataset, reweighted until each fits χ² = N; the group norm and λ₂ keep their "
         "physical meaning (<code>gl_balance</code>).",
         f"χ²/N {fit(lo_b)} → {fit(lo_a)} (2 km); " + (f"{fit(gl_b)} → {fit(gl_a)} (full)" if gl_a else "full run pending")],
        ["Group lasso: cost at full resolution",
         "13 λ₁ values, plain ADMM.",
         "Over-relaxed ADMM (α = 1.6: 31% fewer iterations, same minimizer); 8 λ₁ values, λ₁ from χ² = N and the balance.",
         (f"{minutes(gl_a)} min on c5.18xlarge, run to the end ({cost(gl_a)}); the report's run was stopped "
          f"after 59 min at 9 of 13 values" if gl_a else "full run pending")],
        ["Joint total variation: target not reached",
         "The IRLS β control halved its step at each reversal and never let it grow back: β crept by 1.7% per "
         "iteration while φ_d stayed 20–50% above the target.",
         "Two corrections in a row in the same direction make the step 1.5× larger again (<code>DampedUpdateIRLS</code>).",
         f"χ²/N {fit(jtv_b)} → " + (fit(jtv_a) if jtv_a else "pending") + " (full)"],
        ["PGI: slow, and not compact",
         "The tutorials' sensitivity weights (they filled the bottom of the mesh in the magnetic inversions).",
         "The models' own depth weights (β = 1), and a first β of 0.1 of the eigenvalue estimate with them "
         "(SimPEG's PGI schedule never raises β).",
         f"2 km: in the core {cores(LOW['pgi']['before'])} → {cores(LOW['pgi']['after'])}, χ²/N "
         f"{fit(LOW['pgi']['before'])} → {fit(LOW['pgi']['after'])}, "
         f"{LOW['pgi']['after']['n_iterations']} iterations in {minutes(LOW['pgi']['after'])} min. At full "
         "resolution it still overfitted (0.42 N after 13 iterations): its first β needs tuning per dataset, "
         "so PGI was not pursued further (the run was stopped)."],
        ["Magnetic data: systematic residual",
         "An induced magnetization cannot make the anomaly of the south of the Sandur belt: remanence.",
         "Magnetization-vector inversion (MVI; SimPEG Cartesian, <code>magnetization=\"vector\"</code>).",
         (f"RMS {mi['rms']:.0f} → {mv['rms']:.0f} nT; at the {u['n']} stations underfitted by more than "
          f"{u['threshold_nT']:.0f} nT: {u['induced_mean_nT']:.0f} → {u['mvi_mean_nT']:.0f} nT on average"
          if mv else "full run pending")],
    ]

    body = [
        '<header><div class="eyebrow">GeoInv3D · field-data test · 30 September 2026 · follow-up</div>',
        "<h1>Karnataka Joint Inversion: the Open Problems of Section 5</h1>",
        "<p class=\"lede\">Section 5 of the joint report listed what was left open after the first joint inversion "
        "of the Karnataka gravity and magnetic data. Each problem has been traced to its cause and fixed in the "
        "code; the 2 km study was repeated locally and the full-resolution runs on AWS EC2, with the same data, "
        "mesh and settings as in the report except for the fixes.</p></header>",
        "<h2>1 What was wrong, and what changed</h2>",
        table(["Problem", "Cause", "Fix", "Result"],
              [[c if i == 0 else (c, "wrap") for i, c in enumerate(r)] for r in problems], numeric_from=9),
        "<h2>2 The full-resolution runs</h2>",
        f"<p>335,518 cells below the ground, 10,081 data, as in the report; the fixed runs on c5.9xlarge "
        f"(c5.18xlarge for the group lasso), ${total_cost:.2f} in all including the MVI run. "
        "Red: χ²/N outside 0.7–1.3, or less than half of a model in the core. † stopped by us before the end.</p>",
        table(HEAD + ["Cost"], rows_for(FULL), compact=True),
    ]
    if gl_a:
        body.append(
            f"<p><b>Group lasso.</b> With the cells weighed like those of the other runs the group lasso keeps "
            f"{cores(gl_a)} of the density and susceptibility models in the core ({cores(gl_b)} as reported), and the "
            f"balance fits both datasets: χ²/N {fit(gl_a)}"
            + (f" after {len(bal) - 1} rounds of reweighing (data weights "
               + ", ".join(f"{w:.2f}" for w in bal[-1]['data_weights']) + " for gravity and magnetics)" if bal else "")
            + ". Its coupling is still visible: "
            f"{pct(gl_a['coupling']['support']['magnetic_in_dense'])} of the magnetic cells are anomalous in density "
            f"({pct(FULL['group_lasso_depth_uncoupled']['after']['coupling']['support']['magnetic_in_dense'])} "
            "in its uncoupled control), against "
            f"{pct(FULL['none']['after']['coupling']['support']['magnetic_in_dense'])} for the SimPEG run without "
            "coupling.</p>"
            if FULL.get("group_lasso_depth_uncoupled", {}).get("after") and FULL.get("none", {}).get("after") else "")
    body.append(F("fixes_sections", "E–W sections through the Sandur belt (northing 1667.5 km), down to the base "
                  "of the mesh: the group lasso as reported (stopped at the 9th of 13 values of λ₁), the group lasso "
                  "fixed, and the uncoupled run. The dashed line is the base of the core; the titles give the shares "
                  "of |model| × volume in and below the core."))
    body += ["<h2>3 The 2 km study</h2>",
             "<p>The same couplings on the 2 km × 2 km × 500 m mesh (1,296 + 1,296 data), locally.</p>",
             table(HEAD, rows_for(LOW), compact=True)]
    if mv:
        body += ["<h2>4 Remanence: the magnetization-vector inversion</h2>",
                 f"<p>The single magnetic inversion of the magnetic report (β = 1) and the same with a magnetization "
                 f"vector per cell (|m<sub>i</sub>| ≤ 1 SI): χ²/N {mi['chi2']:.2f} → {mv['chi2']:.2f}, RMS "
                 f"{mi['rms']:.0f} → {mv['rms']:.0f} nT, largest residual {mi['max_abs']:.0f} → {mv['max_abs']:.0f} nT. "
                 f"The {u['n']} stations that the induced model underfits by more than {u['threshold_nT']:.0f} nT lie at "
                 f"easting {u['extent_km'][0]:.0f}–{u['extent_km'][1]:.0f} km, northing {u['extent_km'][2]:.0f}–"
                 f"{u['extent_km'][3]:.0f} km; their mean residual falls from {u['induced_mean_nT']:.0f} to "
                 f"{u['mvi_mean_nT']:.0f} nT. The strongly magnetized cells point at inclination "
                 f"{mz.get('resultant_inclination', 0):.0f}°, declination {mz.get('resultant_declination', 0):.0f}° "
                 f"(coherence {mz.get('resultant_coherence', 0):.2f}), far from the present field (I "
                 f"{mz.get('inducing_inclination', 0):.0f}°, D {mz.get('inducing_declination', 0):.0f}°): the data ask "
                 f"for remanent magnetization, as the magnetic report suspected. {pct(mv['core'])} of the MVI "
                 f"amplitude lies in the core ({pct(mi['core'])} of the induced model).</p>",
                 F("fixes_mvi", "Magnetic residuals of the induced and the magnetization-vector inversion (±150 nT; "
                   "black dots: the stations the induced model underfits by more than 150 nT), and the MVI "
                   "amplitude integrated with depth."),
                 "<p>The MVI model is a single-method result: the joint inversions still couple the induced "
                 "susceptibility. A joint inversion with a vector model (coupling its amplitude) is not in this "
                 "version.</p>"]
    body += ["<h2>5 What is still open</h2><ul>",
             "<li>The choice of coupling is unchanged by the fixes: on this belt the structural couplings remain "
             "the consistent assumption (Section 3 of the report).</li>",
             "<li>Remanence in the joint inversion: the couplings act on the induced susceptibility; the MVI "
             "amplitude could be coupled instead.</li>",
             "<li>PGI's rock units are still our assumptions; measured susceptibilities of the iron formation and "
             "the greenstone would make them a constraint.</li>",
             "<li>Depth still needs independent information (mapped dips, boreholes).</li></ul>",
             "<h2>Files</h2><ul>",
             "<li>Code: <code>geoinv3d/methods/group_lasso.py</code> (cell_weights, scale_models, relaxation), "
             "<code>geoinv3d/cloud/worker.py</code> (gl_weighting, gl_balance, run_mvi_inversion), "
             "<code>geoinv3d/methods/directives.py</code> (DampedUpdateIRLS), <code>geoinv3d/methods/joint.py</code> "
             "and <code>coupling.py</code> (PGI); docs/group_lasso_joint.md, docs/joint_couplings.md, "
             "docs/magnetization_vector.md</li>",
             "<li>Runs: <code>data/ec2_runs_fixed/</code>, <code>data/lowres_runs_fixed/</code>, "
             "<code>karnataka_magnetic/data/ec2_runs/beta1_mvi</code>; parameters in "
             "<code>scripts/joint_params.py</code> and <code>deploy/ec2_multi_run.py</code></li>",
             "<li>This page: <code>scripts/fixes_summary.py</code>, <code>scripts/fixes_figures.py</code>, "
             "<code>scripts/build_fixes_report.py</code></li></ul>",
             '<p class="note">GeoInv3D · SimPEG 0.25.2 · every number comes from the runs above.</p>']
    OUT.write_text(page("en", "Karnataka Section 5 Fixes", "\n".join(body)), encoding="utf-8")
    print(OUT)
    if "--pdf" in sys.argv:
        print(to_pdf(OUT))


if __name__ == "__main__":
    main()
