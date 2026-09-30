"""The Karnataka joint gravity-magnetic report, second series of runs: English HTML (and PDF).

    py examples/output/karnataka_joint/scripts/make_figures.py --set v2
    py examples/output/karnataka_joint/scripts/build_report_v2.py [--pdf]

Every number in the text comes from figures_v2/numbers.json; the interpretive text is in
report_text_v2.py.  Runs: data/ec2_runs_fixed (full resolution), data/lowres_runs_fixed (2 km),
karnataka_magnetic/data/ec2_runs/beta1 and beta1_mvi (the magnetic data alone).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "karnataka_inputs" / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import Figures, page, table, to_pdf  # noqa: E402

FIGS = ROOT / "figures_v2"
N = json.loads((FIGS / "numbers.json").read_text(encoding="utf-8"))
F, LOW, S, RUNS, M, MAG = N["full"], N["low"], N["single"], N["runs"], N["mesh"], N.get("magnetic", {})
GL = N.get("gl_info", {})
KEYS = N["keys"]
GLK, CTRL = "group_lasso_depth", "group_lasso_depth_uncoupled"
LABEL = {"none": "No coupling", "cross_gradient": "Cross-gradient", "joint_total_variation": "Joint total variation",
         "linear_correspondence": "Linear correspondence", "pgi": "PGI (rock units)",
         GLK: "Group lasso", CTRL: "L1 + L2 by ADMM, no coupling (the group lasso's control)"}
FAMILY = {"none": "—", "cross_gradient": "structural", "joint_total_variation": "structural",
          "linear_correspondence": "petrophysical", "pgi": "petrophysical", GLK: "joint sparsity", CTRL: "control"}
ASSUMES = {
    "none": "Nothing: the two models only share the mesh.",
    "cross_gradient": "The gradients of the two models are parallel, or one of them is zero (Gallardo &amp; Meju 2003).",
    "joint_total_variation": "The two models change in the same places (Haber &amp; Holtzman Gazit 2013).",
    "linear_correspondence": "One linear relation in every cell: density contrast = 0.5 × susceptibility, the ratio of the two upper bounds.",
    GLK: "Few cells are anomalous, and a cell is anomalous in both models or in neither (Utsugi 2025).",
    CTRL: "Few cells are anomalous in each model, no pairing: what the group lasso's solver does without the coupling.",
}
OUT = ROOT / "karnataka_joint_report_v2_en.html"


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
        its = F[k]["n_iterations"]
        unit = " values of λ₁" if k.startswith("group_lasso") else ""
        rows.append([LABEL[k], (FAMILY[k], ""), (ASSUMES[k], "wrap"), f"{its}{unit}",
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
    names = {k: v for k, v in LABEL.items()}
    head = ["Run (2 km mesh, local)", "Gravity χ²/N", "Magnetic χ²/N", "Density range (g/cc)", "Susceptibility range (SI)",
            "Density in the core", "Susceptibility in the core", "Time (min)"]
    rows = []
    for k in KEYS:
        if k not in LOW:
            continue
        f = LOW[k]
        gg, mm = f["gravity"], f["magnetics"]
        rows.append([names[k], warn(f"{gg['chi2']:.2f}", not 0.7 <= gg["chi2"] <= 1.3),
                     warn(f"{mm['chi2']:.2f}", not 0.7 <= mm["chi2"] <= 1.3),
                     f"{gg['min']:.2f} to {gg['max']:.2f}".replace("-", "−"),
                     f"{mm['min']:.2f} to {mm['max']:.2f}".replace("-", "−"),
                     pct(gg["core"]), pct(mm["core"]), f"{f['seconds'] / 60:.1f}" if f.get("seconds") else "—"])
    return table(head, rows)


def mvi_table():
    a, b = MAG["induced"], MAG["mvi"]
    head = ["Magnetic inversion (β = 1)", "χ²/N", "RMS (nT)", "Max |residual| (nT)", "In the core", "Below the core",
            "Sandur belt, centroid km", "EC2 machine", "Time (min)"]
    rows = []
    for label, x in (("Susceptibility along the present field", a), ("Magnetization vector (MVI)", b)):
        run = x.get("run") or {}
        rows.append([label, f"{x['chi2']:.2f}", f"{x['rms']:.0f}", f"{x['max_abs']:.0f}", pct(x["core"]),
                     pct(x["below"]), f"{x['box_sandur']['centroid_km']:.1f}", (run.get("instance", "—"), ""),
                     f"{run['minutes']:.0f}" if "minutes" in run else "—"])
    return table(head, rows, numeric_from=1)


def body(fig, notes):
    n = len(KEYS)
    cost = sum(r["cost_usd"] for r in RUNS.values())
    minutes = [r["minutes"] for r in RUNS.values()]
    mag_cost = ((MAG.get("mvi") or {}).get("run") or {}).get("cost_usd", 0.0)
    gl = GL.get(GLK) or {}
    return f"""
<header>
  <div class="eyebrow">GeoInv3D · field-data test · 30 September 2026 · second series</div>
  <h1>Karnataka Joint Gravity–Magnetic Inversion: Couplings and Remanence</h1>
  <p class="lede">The gravity and the magnetic data of the same area inverted together on one mesh with terrain, once for each coupling of the joint inversion, and the magnetic data alone with a magnetization vector per cell. The report shows what each coupling does to the two models, what it costs in data fit, which assumption suits this belt, and what the magnetic data say about the direction of magnetization.</p>
  <dl class="meta">
    <div><dt>Area</dt><dd>70 × 70 km, easting 641–711 km, northing 1634–1704 km (UTM 43N, EPSG:32643)</dd></div>
    <div><dt>Data</dt><dd>Complete Bouguer anomaly, {N['n_data']['gravity']:,} points, error 0.5 mGal; magnetic anomaly continued to 1 km above the ground, {N['n_data']['magnetics']:,} points, error 5% + 10 nT; second-order trends removed</dd></div>
    <div><dt>Models</dt><dd>Density contrast (−0.2 to +0.5 g/cc) and susceptibility (0 to 1 SI) on {M['n_active']:,} cells below the ground (1 km × 1 km × 250 m); each sparse with α<sub>s</sub> = 1 and depth weighting β = 1</dd></div>
    <div><dt>Runs</dt><dd>{n} joint inversions and one magnetization-vector inversion on AWS EC2 (ap-south-1, c5.9xlarge and c5.18xlarge), {min(minutes):.0f}–{max(minutes):.0f} minutes each, ${cost + mag_cost:.2f} in total; the same couplings on a 2 km mesh locally</dd></div>
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
<p>In the runs of the SimPEG couplings each model is regularized as in the reference run of its own report: sparse (IRLS) with α<sub>s</sub> = 1, p = [0, 2, 2, 2] and Li &amp; Oldenburg depth weighting with β = 1, within its bounds. One trade-off parameter serves both models; the pipeline balances the two regularizations against their data and cools it until χ² = N for the {N['n_data']['gravity'] + N['n_data']['magnetics']:,} data together, with at most 60 iterations and 40 IRLS iterations. The coupling weight is 1 in the pipeline's unit-free scale (as strong as the regularization).</p>
<p>The group lasso has its own regularization: an L2 term (λ₂ = {gl.get('lambda2', 0.3):g}) and the group norm of each cell's (density, susceptibility) pair, solved by ADMM within the same bounds. Its cells are weighed like those of the other runs (cell volume × the depth weight with β = 1), each datum by its error, λ₁ is chosen on a sweep of {F[GLK]['n_iterations'] if has(GLK) else 8} values for χ² = N, and the two datasets are then reweighted until each fits its own χ² = N. Its control runs the same solver without the pairing: each model soft-thresholded on its own.</p>
</div>
{couplings_table()}
<p class="note">The couplings are those of <code>geoinv3d/methods/coupling.py</code> (docs/joint_couplings.md); the group lasso of <code>geoinv3d/methods/group_lasso.py</code> (docs/group_lasso_joint.md).</p>

<h2><span class="no">2</span>Results</h2>

<h3><span class="no">2.1</span>Data fit</h3>
{fit_table()}
<p class="note">Red: χ²/N outside 0.7–1.3. The first row gives the single inversions with the same settings.</p>
{fig('residuals', "Residuals (observed − predicted) of the gravity data (top, ±3 mGal) and the magnetic data (bottom, ±150 nT) for each coupling.")}
<div class="prose">
@@FITTEXT@@
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

<h2><span class="no">3</span>The magnetic residual: remanent magnetization</h2>
<div class="prose">
@@MVITEXT@@
</div>
{mvi_table()}
{fig('mvi', "Magnetic residuals of the susceptibility inversion and of the magnetization-vector inversion (±150 nT; black dots: the stations the susceptibility model underfits by more than 150 nT), and the amplitude of the magnetization vector integrated with depth.")}

<h2><span class="no">4</span>Which coupling suits this belt</h2>
<div class="prose">
@@DISCUSSION@@
</div>

<h2><span class="no">5</span>The 2 km study</h2>
<div class="prose">
<p>Every coupling was also run locally on a 2 km × 2 km × 500 m mesh (1,296 + 1,296 data):</p>
</div>
{low_table()}
<p class="note">Red: χ²/N outside 0.7–1.3.</p>
<div class="prose">
@@LOWTEXT@@
</div>

<h2><span class="no">6</span>Conclusions and recommendations</h2>
<div class="prose">
<ul class="plain">
@@RECOMMEND@@
</ul>
</div>

<h2>Files</h2>
<div class="prose">
<ul class="plain">
  <li>Parameters of every run: <code>examples/output/karnataka_joint/scripts/joint_params.py</code> (joint) and <code>deploy/ec2_multi_run.py</code> (the magnetic runs <code>beta1</code> and <code>beta1_mvi</code>); launched with <code>deploy/ec2_multi_run.py</code> and <code>scripts/run_lowres.py --out data/lowres_runs_fixed</code></li>
  <li>This report and its figures: <code>scripts/make_figures.py --set v2</code>, <code>scripts/build_report_v2.py</code>, <code>scripts/report_text_v2.py</code>; results in <code>data/ec2_runs_fixed/</code>, <code>data/lowres_runs_fixed/</code> and <code>karnataka_magnetic/data/ec2_runs/</code></li>
  <li>The single-method reports: <code>examples/output/karnataka_gravity_terrain/</code>, <code>examples/output/karnataka_magnetic/</code></li>
</ul>
</div>
<footer>GeoInv3D · SimPEG 0.25.2 · Generated by build_report_v2.py; every number comes from the runs above.</footer>
"""


def main():
    from report_text_v2 import coupling_notes, discussion, fit_text, low_text, mvi_text, recommend, summary
    fig = Figures("en", FIGS)
    ctx = dict(F=F, LOW=LOW, S=S, MAG=MAG, GL=GL, KEYS=KEYS, g=g, m=m, c=c, has=has, pct=pct, rng=rng,
               GLK=GLK, CTRL=CTRL)
    html = body(fig, coupling_notes(**ctx))
    for tag, fn in (("@@SUMMARY@@", summary), ("@@FITTEXT@@", fit_text), ("@@MVITEXT@@", mvi_text),
                    ("@@DISCUSSION@@", discussion), ("@@LOWTEXT@@", low_text), ("@@RECOMMEND@@", recommend)):
        html = html.replace(tag, fn(**ctx))
    OUT.write_text(page("en", "Karnataka Joint Inversion", html), encoding="utf-8")
    print(OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")
    if "--pdf" in sys.argv:
        print(to_pdf(OUT))


if __name__ == "__main__":
    main()
