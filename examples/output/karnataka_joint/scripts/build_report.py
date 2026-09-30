"""Build the Karnataka joint gravity-magnetic report: English HTML, optionally the PDF.

    py examples/output/karnataka_joint/scripts/make_figures.py
    py examples/output/karnataka_joint/scripts/build_report.py [--pdf]

Every number in the text comes from figures/numbers.json; the interpretive text is in
report_text.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "karnataka_inputs" / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import Figures, page, table, to_pdf  # noqa: E402

FIGS = ROOT / "figures"
N = json.loads((FIGS / "numbers.json").read_text(encoding="utf-8"))
F, LOW, S, RUNS, M = N["full"], N["low"], N["single"], N["runs"], N["mesh"]
KEYS = N["keys"]
LABEL = {"none": "No coupling", "cross_gradient": "Cross-gradient", "joint_total_variation": "Joint total variation",
         "linear_correspondence": "Linear correspondence", "pgi": "PGI (rock units)",
         "group_lasso": "Group lasso",
         "group_lasso_uncoupled": "L1 + L2 by ADMM, no coupling (the group lasso's control)",
         "group_lasso_std": "Group lasso, error-weighted",
         "group_lasso_std_uncoupled": "L1 + L2 by ADMM, error-weighted, no coupling"}
# runs stopped before their end are marked in every table
LABEL = {k: v + (" †" if (F.get(k) or {}).get("stopped") else "") for k, v in LABEL.items()}
DAGGER = "† Stopped by us before the end, to limit the cost: " + "; ".join(
    f"{LABEL[k].rstrip(' †')} after {F[k]['stopped']['at_iteration']} of "
    f"{F[k]['stopped'].get('of') or 60} {'values of λ₁' if k.startswith('group_lasso') else 'iterations'}"
    for k in KEYS if F[k].get("stopped")) + ". Their models are intermediate."
FAMILY = {"none": "—", "cross_gradient": "structural", "joint_total_variation": "structural",
          "linear_correspondence": "petrophysical", "pgi": "petrophysical", "group_lasso": "joint sparsity",
          "group_lasso_uncoupled": "control", "group_lasso_std": "joint sparsity", "group_lasso_std_uncoupled": "control"}
ASSUMES = {
    "none": "Nothing: the two models only share the mesh.",
    "cross_gradient": "The gradients of the two models are parallel, or one of them is zero (Gallardo &amp; Meju 2003).",
    "joint_total_variation": "The two models change in the same places (Haber &amp; Holtzman Gazit 2013).",
    "linear_correspondence": "One linear relation in every cell: density contrast = 0.5 × susceptibility, the ratio of the two upper bounds.",
    "pgi": "Every cell belongs to one of a few rock units with a density and a susceptibility (Astic &amp; Oldenburg 2019).",
    "group_lasso": "Few cells are anomalous, and a cell is anomalous in both models or in neither (Utsugi 2025).",
    "group_lasso_uncoupled": "Few cells are anomalous in each model, no pairing: what the group lasso's solver does without the coupling.",
    "group_lasso_std": "The group lasso with the data weighted by their errors and λ₁ chosen for χ² = N, like the other runs.",
    "group_lasso_std_uncoupled": "The control of the row above: no pairing.",
}
OUT = ROOT / "karnataka_joint_report_en.html"


def pct(x):
    return f"{100 * x:.0f}%"


def rng(b):
    return f"{b['top_km']:.1f}–{b['bottom_km']:.1f}"


def has(k):
    return k in F


def warn(text, bad):
    return (text, "n warn" if bad else "n")


def good(text, ok):
    return (text, "n good" if ok else "n")


def g(k):
    return F[k]["gravity"]


def m(k):
    return F[k]["magnetics"]


def c(k):
    return F[k]["coupling"]


# ---------------------------------------------------------------- tables

def couplings_table():
    head = ["Coupling", "Kind", "What it assumes", "Iterations", "EC2 machine", "Time (min)"]
    rows = []
    for k in KEYS:
        r = RUNS.get(k, {})
        rows.append([LABEL[k], (FAMILY[k], ""), (ASSUMES[k], "wrap"), str(F[k]["n_iterations"]),
                     (r.get("instance", "—"), ""), f"{r['minutes']:.0f}" if r else "—"])
    return table(head, rows, numeric_from=3)


def fit_table():
    head = ["Coupling", "Gravity χ²/N", "RMS (mGal)", "Max |residual| (mGal)", "Magnetic χ²/N", "RMS (nT)",
            "Max |residual| (nT)"]
    rows = [["Single inversions (same settings)", f"{S['gravity_chi2']:.2f}", "", "", f"{S['magnetics_chi2']:.2f}", "", ""]]
    for k in KEYS:
        rows.append([LABEL[k], warn(f"{g(k)['chi2']:.2f}", not 0.7 <= g(k)["chi2"] <= 1.3), f"{g(k)['rms']:.2f}",
                     f"{g(k)['max_abs']:.1f}", warn(f"{m(k)['chi2']:.2f}", not 0.7 <= m(k)["chi2"] <= 1.3),
                     f"{m(k)['rms']:.0f}", f"{m(k)['max_abs']:,.0f}"])
    return table(head, rows, compact=True)


def models_table():
    head = ["Coupling", "Density: core", "below the core", "main high, half-max km", "centroid km",
            "correlation with the uncoupled model", "Susceptibility: core", "below the core",
            "Sandur belt, centroid km", "SI·km³", "correlation with the uncoupled model"]
    rows = []
    for k in KEYS:
        rows.append([LABEL[k], warn(pct(g(k)["core"]), g(k)["core"] < 0.5), warn(pct(g(k)["below"]), g(k)["below"] > 0.3),
                     rng(g(k)["main"]), f"{g(k)['main']['centroid_km']:.1f}", f"{g(k)['corr_with_none']:.2f}",
                     warn(pct(m(k)["core"]), m(k)["core"] < 0.5), warn(pct(m(k)["below"]), m(k)["below"] > 0.3),
                     f"{m(k)['box_sandur']['centroid_km']:.1f}", f"{m(k)['box_sandur']['total']:.0f}",
                     f"{m(k)['corr_with_none']:.2f}"])
    return table(head, rows)


def alike_table():
    head = ["Coupling", "Cross-gradient (0 = parallel gradients)", "Susceptibility edges on a density edge",
            "Magnetic cells that are dense or light", "Cell-by-cell correlation of |density| and susceptibility",
            "Correlation of the integrated maps"]
    base = c("none")
    rows = []
    for k in KEYS:
        x = c(k)
        rows.append([LABEL[k], good(f"{x['cross_gradient']:.2f}", x["cross_gradient"] < 0.5 * base["cross_gradient"]),
                     good(pct(x["edges_shared"]), x["edges_shared"] > 1.3 * base["edges_shared"]),
                     good(pct(x["support"]["magnetic_in_dense"]), x["support"]["magnetic_in_dense"] > 2 * base["support"]["magnetic_in_dense"]),
                     good(f"{x['corr_abs_cells']:.2f}", x["corr_abs_cells"] > 0.4),
                     good(f"{x['corr_integrated']:.2f}", x["corr_integrated"] > 0.5)])
    return table(head, rows, compact=True)


def low_table():
    order = ["none", "cross_gradient", "joint_total_variation", "linear_correspondence", "pgi", "group_lasso_unbounded",
             "group_lasso", "group_lasso_uncoupled", "group_lasso_std", "group_lasso_std_uncoupled"]
    names = {**{k: v.rstrip(" †") for k, v in LABEL.items()}, "group_lasso_unbounded": "Group lasso without bounds"}
    head = ["Run (2 km mesh, local)", "Gravity χ²/N", "Magnetic χ²/N", "Density range (g/cc)", "Susceptibility range (SI)",
            "Density in the core", "Susceptibility in the core", "Time (min)"]
    rows = []
    for k in order:
        if k not in LOW:
            continue
        f = LOW[k]
        gg, mm = f["gravity"], f["magnetics"]
        rows.append([names[k], warn(f"{gg['chi2']:.2f}", not 0.7 <= gg["chi2"] <= 1.3),
                     warn(f"{mm['chi2']:.2f}", not 0.7 <= mm["chi2"] <= 1.3),
                     warn(f"{gg['min']:.2f} to {gg['max']:.2f}".replace("-", "−"), gg["max"] > 1 or gg["min"] < -1),
                     warn(f"{mm['min']:.2f} to {mm['max']:.2f}".replace("-", "−"), mm["min"] < -0.01),
                     pct(gg["core"]), pct(mm["core"]), f"{f['seconds'] / 60:.0f}" if f.get("seconds") else "—"])
    return table(head, rows)


def body(fig, notes):
    n = len(KEYS)
    cost = sum(r["cost_usd"] for r in RUNS.values())
    minutes = [r["minutes"] for r in RUNS.values()]
    return f"""
<header>
  <div class="eyebrow">GeoInv3D · field-data test · 30 September 2026</div>
  <h1>Karnataka Joint Gravity–Magnetic Inversion: Comparing Couplings</h1>
  <p class="lede">The gravity and the magnetic data of the same area inverted together on one mesh with terrain, once for each coupling of the joint inversion. Each model keeps the regularization of its single inversion; only the coupling differs. The report shows what each coupling does to the two models, what it costs in data fit, and which assumption suits this belt.</p>
  <dl class="meta">
    <div><dt>Area</dt><dd>70 × 70 km, easting 641–711 km, northing 1634–1704 km (UTM 43N, EPSG:32643)</dd></div>
    <div><dt>Data</dt><dd>Complete Bouguer anomaly, {N['n_data']['gravity']:,} points, error 0.5 mGal; magnetic anomaly continued to 1 km above the ground, {N['n_data']['magnetics']:,} points, error 5% + 10 nT; second-order trends removed</dd></div>
    <div><dt>Models</dt><dd>Density contrast (−0.2 to +0.5 g/cc) and susceptibility (0 to 1 SI) on {M['n_active']:,} cells below the ground (1 km × 1 km × 250 m); each sparse with α<sub>s</sub> = 1 and depth weighting β = 1</dd></div>
    <div><dt>Runs</dt><dd>{n} joint inversions on AWS EC2 (ap-south-1, c5.9xlarge and c5.18xlarge), {min(minutes):.0f}–{max(minutes):.0f} minutes each, ${cost:.2f} in total; the same couplings on a 2 km mesh locally</dd></div>
  </dl>
</header>

<section class="summary" aria-labelledby="sum">
  <h2 id="sum">Summary</h2>
  <ul>
@@SUMMARY@@
  </ul>
</section>

<h2><span class="no">1</span>Set-up</h2>
<div class="prose">
<p>The two datasets are those of the single-method reports: the Bouguer anomaly with the terrain correction at 1 km, and the magnetic anomaly continued upwards to 1 km above the ground. They share one mesh that follows the ground ({M['shape'][0]} × {M['shape'][1]} × {M['shape'][2]} cells, {M['n_active']:,} below the ground), so that a cell of the density model and a cell of the susceptibility model are the same rock.</p>
<p>In every run each model is regularized as in the reference run of its own report: sparse (IRLS) with α<sub>s</sub> = 1, p = [0, 2, 2, 2] and Li &amp; Oldenburg depth weighting with β = 1, within its bounds. One trade-off parameter serves both models; the pipeline balances the two regularizations against their data and cools it until χ² = N for the {N['n_data']['gravity'] + N['n_data']['magnetics']:,} data together, with at most 60 iterations and 40 IRLS iterations. PGI and the group lasso replace this regularization by their own. The coupling weight is 1 in the pipeline's unit-free scale (as strong as the regularization).</p>
</div>
{couplings_table()}
<p class="note">{DAGGER}</p>
<p class="note">The couplings are those of <code>geoinv3d/methods/coupling.py</code> (docs/joint_couplings.md). PGI's units: iron formation (+0.5 g/cc, 0.5 SI), greenstone (+0.3 g/cc, 0.005 SI), light granite (−0.15 g/cc, 0.002 SI) and the background; the densities follow the rock samples, the susceptibility of the iron formation is what the magnetic data need in a 1 km cell, not a measurement. The group lasso uses the settings of its paper (datasets balanced by their largest amplitudes, λ₁ from the L-curve over 13 values, λ₂ = 0.3); it had no bounds before this test, and they were added for it (Section 4).</p>

<h2><span class="no">2</span>Results</h2>

<h3><span class="no">2.1</span>Data fit</h3>
{fit_table()}
<p class="note">Red: χ²/N outside 0.7–1.3. The first row gives the single inversions with the same settings. † stopped early (Section 1).</p>
{fig('residuals', "Residuals (observed − predicted) of the gravity data (top, ±3 mGal) and the magnetic data (bottom, ±150 nT) for each coupling.")}
<div class="prose">
<p>Inverted together without a coupling, the two models are those of the single inversions (cell-by-cell correlation {S['gravity_corr']:.2f} for density and {S['magnetics_corr']:.2f} for susceptibility): the joint set-up itself changes nothing. The residual of the magnetic data south of the Sandur belt, described in the magnetic report, is there in every run: no coupling repairs what the induced-magnetization model cannot explain.</p>
</div>

<h3><span class="no">2.2</span>What each coupling does to the models</h3>
{fig('sections', "E–W sections through the Sandur belt (northing 1667.5 km): density contrast (left, ±0.3 g/cc) and susceptibility (right, square-root scale to 1 SI) for each coupling. The black line is the ground of the mesh, the dashed line the base of the core.")}
{models_table()}
<p class="note">Depths below the ground. Main high: the density profile under the main Bouguer high (671.5, 1664.5 km). Sandur belt: susceptibility × volume in the box of the magnetic report. Correlations are cell by cell in the core columns. Red: less than half in the core, or more than 30% below it.</p>
{fig('slices', "Layers at about 1.5 km and 4 km below the mean ground: density contrast and susceptibility for each coupling (core area only).")}
<div class="prose">
<ul class="plain">
  {notes}
</ul>
</div>

<h3><span class="no">2.3</span>How alike the two models are</h3>
<div class="prose">
<p>Measured in the core of the mesh. The cross-gradient measure is the sum of |∇ρ × ∇χ|² over the sum of |∇ρ|²|∇χ|²: 0 when the gradients are parallel wherever both exist, 1 when they are perpendicular. An edge is a cell whose squared gradient exceeds 1% of the model's largest; a cell is anomalous when its value exceeds a tenth of the model's largest.</p>
</div>
{alike_table()}
<p class="note">Green: clearly more alike than the uncoupled models (the first row).</p>
{fig('crossplot', "Density contrast against susceptibility, cell by cell in the core (logarithmic counts). Each coupling leaves its own signature.")}
{fig('integrated', "Vertically integrated density (top) and susceptibility (bottom) for each coupling.")}
{fig('profiles', "Dense rock (left: positive density contrast × volume) and magnetic rock (right: susceptibility × volume) in the Sandur belt, per kilometre of depth below the ground, for each coupling.")}

<h2><span class="no">3</span>Which coupling suits this belt</h2>
<div class="prose">
@@DISCUSSION@@
</div>

<h2><span class="no">4</span>The 2 km study, and bounds for the group lasso</h2>
<div class="prose">
<p>Before the EC2 runs every coupling was run locally on a 2 km × 2 km × 500 m mesh (1,296 + 1,296 data):</p>
</div>
{low_table()}
<p class="note">Red: χ²/N outside 0.7–1.3, or values outside the bounds of the other runs.</p>
<div class="prose">
@@LOWTEXT@@
</div>

<h2><span class="no">5</span>Problems found, and what is unfinished</h2>
<div class="prose">
<p>This test was the first use of the joint inversion on field data at this size. It was stopped before every run had finished; the table lists what went wrong, what was fixed on the way and what is left.</p>
</div>
@@PROBLEMS@@
<p class="note">AWS cost of the three reports (gravity with terrain, magnetics, joint): about $17 on EC2, of which $6 for the two group lasso runs of this report and $3.6 for two that were stopped without a result.</p>

<h2><span class="no">6</span>Recommendations</h2>
<div class="prose">
<ul class="plain">
@@RECOMMEND@@
</ul>
</div>

<h2>Files</h2>
<div class="prose">
<ul class="plain">
  <li>Parameters of every run: <code>examples/output/karnataka_joint/scripts/joint_params.py</code>; launched with <code>deploy/ec2_multi_run.py karnataka-joint</code> and <code>scripts/run_lowres.py</code></li>
  <li>This report and its figures: <code>scripts/make_figures.py</code>, <code>scripts/build_report.py</code>, <code>scripts/report_text.py</code>; results in <code>data/ec2_runs/</code> and <code>data/lowres_runs/</code></li>
  <li>All runs in one interactive workflow (DAG viewer): <code>karnataka_joint.geoinv3d_viewer.html</code></li>
  <li>The single-method reports: <code>examples/output/karnataka_gravity_terrain/</code>, <code>examples/output/karnataka_magnetic/</code></li>
</ul>
</div>
<footer>GeoInv3D · SimPEG 0.25.2 · Generated by build_report.py; every number comes from the runs above.</footer>
"""


def main():
    from report_text import coupling_notes, discussion, low_text, problems, recommend, summary
    fig = Figures("en", FIGS)
    ctx = dict(F=F, LOW=LOW, S=S, KEYS=KEYS, g=g, m=m, c=c, has=has, pct=pct, rng=rng)
    html = body(fig, coupling_notes(**ctx))
    for tag, fn in (("@@SUMMARY@@", summary), ("@@DISCUSSION@@", discussion), ("@@LOWTEXT@@", low_text),
                    ("@@PROBLEMS@@", problems), ("@@RECOMMEND@@", recommend)):
        html = html.replace(tag, fn(**ctx))
    OUT.write_text(page("en", "Karnataka Joint Inversion", html), encoding="utf-8")
    print(OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")
    if "--pdf" in sys.argv:
        print(to_pdf(OUT))


if __name__ == "__main__":
    main()
