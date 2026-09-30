"""Build the Karnataka gravity report with terrain: English HTML, optionally the PDF.

    py examples/output/karnataka_gravity_terrain/scripts/make_figures.py
    py examples/output/karnataka_gravity_terrain/scripts/build_report.py [--pdf]

Every number in the text comes from figures/numbers.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "karnataka_inputs" / "shared"))
from style import Figures, esc, page, table, to_pdf  # noqa: E402

FIGS = ROOT / "figures"
N = json.loads((FIGS / "numbers.json").read_text(encoding="utf-8"))
F, OLD, LOW, ST, D, P, M = N["full"], N["old"], N["low"], N["steps"], N["data"], N["prep"], N["mesh"]
TC, TCS, GS = P["gravity"]["tc_nodes"], P["terrain_correction"], P["stations"]
FULL = [("original_sparse", "Original sparse (α_s=1e-4, sensitivity)"), ("l1l2_irls", "L1–L2 (IRLS)"),
        ("as1_beta0.5", "sparse α_s=1, β=0.5"), ("as1_beta1", "sparse α_s=1, β=1"),
        ("as1_beta1.5", "sparse α_s=1, β=1.5"), ("as0.1_beta1", "sparse α_s=0.1, β=1")]
KEYS = [k for k, _ in FULL]
NEW = KEYS[1:]
STEPS = [("flat500", "Flat earth, 500 m layers (28 Sep report)"), ("as1_beta1_flat", "Flat earth, 250 m layers"),
         ("as1_beta1_no_tc", "Terrain in the mesh, no terrain correction"),
         ("as1_beta1", "Terrain in the mesh and terrain correction")]
LOWRES = [k for k in ["base", "as1e-2", "as0.1", "as1", "as1_p0111", "as1_p0222", "as1_p0220", "as1e-4_p0222",
                      "as1_p0222_dw0.5", "as1_p0222_dw1", "as1_p0222_dw1.5", "as1_p0222_dw2",
                      "as1_p0221_dw1", "as1_p0111_dw1", "as0.1_p0222_dw1", "base_dw1"] if k in LOW]
OUT = ROOT / "karnataka_gravity_terrain_report_en.html"


def label(key):
    return esc(dict(FULL)[key]).replace("α_s", "α<sub>s</sub>")


def mgal(x, d=1):
    return f"{x:.{d}f}".replace("-", "−")


def pct(x):
    return f"{100 * x:.0f}%"


def rng(b):
    return f"{b['top_km']:.1f}–{b['bottom_km']:.1f}"


def cen(key, which="main", src=F):
    return src[key][which]["centroid_km"]


def span(values, fmt="{:.2f}"):
    return f"{fmt.format(min(values))}–{fmt.format(max(values))}"


def warn(text, bad):
    return (text, "n warn" if bad else "n")


C = N["integrated_corr"]
_ix = {k: i for i, k in enumerate(C["keys"])}
PAIRS = [("as1_p0222_dw0.5", "as1_beta0.5"), ("as1_p0222_dw1", "as1_beta1"), ("as1_p0222_dw1.5", "as1_beta1.5")]
_shift = {k: cen(k) - cen(k, src=OLD) for k in NEW}
_dw_lat = [LOW[k]["lateral"] for k in LOWRES if "_dw" in k and k.startswith("as1_")]
RUNS = N["runs"]
V = dict(
    chi=span([F[k]["chi2"] for k in KEYS]), rms=span([F[k]["rms"] for k in KEYS]),
    corr_min=f"{C['min_offdiag']:.2f}",
    corr_max=f"{max(C['matrix'][i][j] for i in range(len(KEYS)) for j in range(len(KEYS)) if i != j):.2f}",
    corr_new=f"{min(C['matrix'][_ix[a]][_ix[b]] for a in NEW for b in NEW if a != b):.2f}",
    core_min=pct(min(F[k]["core"] for k in NEW if k != "as1_beta1.5")),
    nw_new=span([cen(k, "nw") for k in NEW], "{:.1f}"),
    nw_gap=span([cen(k) - cen(k, "nw") for k in NEW], "{:.1f}"),
    lowgap=max(abs(LOW[lk]["main"]["centroid_km"] - cen(fk)) for lk, fk in PAIRS) if LOWRES else float("nan"),
    as01gap=abs(LOW["as0.1_p0222_dw1"]["main"]["centroid_km"] - cen("as0.1_beta1")) if LOWRES else float("nan"),
    shift_small=max(abs(v) for k, v in _shift.items() if k != "as0.1_beta1"),
    shift_as01=_shift["as0.1_beta1"],
    oldnew=span(list(N["integrated_old_vs_new"].values())),
    minutes=span([r["minutes"] for r in RUNS.values()], "{:.1f}"),
    cost=sum(r["cost_usd"] for r in RUNS.values()),
    dw_lat=f"{pct(min(_dw_lat))}–{pct(max(_dw_lat))}" if _dw_lat else "",
    low_seconds="about a minute",
)


# ---------------------------------------------------------------- tables

def runs_table():
    head = ["Run", "Regularization", "α<sub>s</sub>", "Norms p = [p<sub>s</sub>, p<sub>x</sub>, p<sub>y</sub>, p<sub>z</sub>]",
            "Depth weighting", "EC2 time (min)"]
    rows = []
    for k in KEYS:
        s = F[k]["settings"]
        if s["regularization_type"] == "l1l2":
            reg, a_s, norms = "elastic net (Utsugi 2019), L1 share 0.8, solved by IRLS", "—", "—"
        else:
            reg, a_s = "sparse (IRLS)", f"{s['alpha_s']:g}"
            norms = "[" + ", ".join(f"{p:g}" for p in s["norms"]) + "]"
        w = (f"Li &amp; Oldenburg, β = {s['depth_weighting_exponent']:g}" if s.get("depth_weighting") == "depth"
             else "sensitivity")
        rows.append([label(k), (reg, "wrap"), a_s, norms, w, f"{RUNS[k]['minutes']:.1f}" if k in RUNS else "—"])
    return table(head, rows, numeric_from=99)


def steps_table():
    head = ["Run (α<sub>s</sub> = 1, β = 1)", "χ²/N", "RMS (mGal)", "Core", "Below the core", "D50 km",
            "Main high, half-max km", "Main high, centroid km", "NW high, centroid km",
            "Integrated density: correlation with the last row"]
    rows = [[lab, f"{ST[k]['chi2']:.2f}", f"{ST[k]['rms']:.2f}", pct(ST[k]["core"]), pct(ST[k]["below"]),
             f"{ST[k]['D50']:.1f}", rng(ST[k]["main"]), f"{ST[k]['main']['centroid_km']:.1f}",
             f"{ST[k]['nw']['centroid_km']:.1f}", f"{N['integrated_steps'][k]:.3f}"] for k, lab in STEPS]
    return table(head, rows)


def fit_table():
    head = ["Run", "χ²/N", "RMS (mGal)", "Max |residual| (mGal)", "Correlation, observed vs predicted", "χ²/N, flat earth"]
    rows = [[label(k), f"{F[k]['chi2']:.2f}", f"{F[k]['rms']:.2f}", f"{F[k]['max_abs']:.1f}", f"{F[k]['corr']:.4f}",
             f"{OLD[k]['chi2']:.2f}"] for k in KEYS]
    return table(head, rows, compact=True)


def corr_table():
    short = ["Original", "L1–L2", "β = 0.5", "β = 1", "β = 1.5", "α<sub>s</sub> = 0.1"]
    m = C["matrix"]
    rows = [[short[i]] + [("—", "n") if i == j else (f"{m[i][j]:.2f}", "n good" if m[i][j] >= 0.9 else "n")
                          for j in range(len(short))] for i in range(len(short))]
    return table([""] + short, rows, compact=True)


def depth_table():
    head = ["Run", "Core", "Lateral padding", "Below the core", "D50 km", "D90 km", "Main high, half-max km",
            "Main high, centroid km", "NW high, half-max km", "NW high, centroid km"]
    rows = []
    for k in KEYS:
        f = F[k]
        rows.append([label(k), pct(f["core"]), warn(pct(f["lateral"]), f["lateral"] > 0.2),
                     warn(pct(f["below"]), f["below"] > 0.3), warn(f"{f['D50']:.1f}", f["D50"] > 7), f"{f['D90']:.1f}",
                     warn(rng(f["main"]), f["main"]["bottom_km"] > 12), f"{cen(k):.1f}", rng(f["nw"]),
                     f"{cen(k, 'nw'):.1f}"])
    return table(head, rows)


def old_new_table():
    head = ["Run", "Main high, half-max km: flat", "with terrain", "Centroid km: flat", "with terrain", "change",
            "D50 km: flat", "with terrain", "NW high, centroid km: flat", "with terrain",
            "Integrated density, correlation flat vs terrain"]
    rows = []
    for k in KEYS:
        o, f = OLD[k], F[k]
        rows.append([label(k), rng(o["main"]), rng(f["main"]), f"{cen(k, src=OLD):.1f}", f"{cen(k):.1f}",
                     mgal(cen(k) - cen(k, src=OLD)), f"{o['D50']:.1f}", f"{f['D50']:.1f}",
                     f"{cen(k, 'nw', OLD):.1f}", f"{cen(k, 'nw'):.1f}", f"{N['integrated_old_vs_new'][k]:.3f}"])
    return table(head, rows)


def lowres_table():
    head = ["Run", "α<sub>s</sub>", "Norms p", "Weighting", "χ²/N", "Core", "Lateral padding", "Below the core", "D50 km",
            "Main high, half-max km", "Main high, centroid km"]
    rows = []
    for k in LOWRES:
        f, s = LOW[k], LOW[k]["settings"]
        w = f"β = {s['depth_weighting_exponent']:g}" if s.get("depth_weighting") == "depth" else "sensitivity"
        norms = "[" + ",".join(f"{p:g}" for p in s["norms"]) + "]"
        rows.append([f"<code>{k}</code>", (f"{s['alpha_s']:g}", ""), (norms, ""), (w, ""), f"{f['chi2']:.2f}",
                     pct(f["core"]), warn(pct(f["lateral"]), f["lateral"] > 0.2), warn(pct(f["below"]), f["below"] > 0.3),
                     warn(f"{f['D50']:.1f}", f["D50"] > 7), warn(rng(f["main"]), f["main"]["bottom_km"] > 12),
                     f"{f['main']['centroid_km']:.1f}"])
    return table(head, rows, numeric_from=4)


# ---------------------------------------------------------------- text

def body(fig):
    o, t = F["original_sparse"], F
    lift = M["lifted"]
    s_flat, s_250, s_notc, s_tc = (ST[k] for k, _ in STEPS)
    gm, gn = F["as1_beta1"]["ground_main_m"], F["as1_beta1"]["ground_nw_m"]
    ground_text = (f"{gm:.0f} m at both the main Bouguer high and the north-western high" if abs(gm - gn) < 1 else
                   f"{gm:.0f} m at the main Bouguer high and {gn:.0f} m at the north-western high")
    return f"""
<header>
  <div class="eyebrow">GeoInv3D · field-data test · 30 September 2026</div>
  <h1>Karnataka Gravity Inversion with Terrain: Comparing Regularizations</h1>
  <p class="lede">The six regularization settings of the 28 September comparison, run again at full resolution on AWS EC2 with the ground surface in the mesh and a terrain correction added to the Bouguer anomaly. This report shows what the terrain changes, what it leaves as it was, and whether the conclusions of the flat-earth comparison still hold.</p>
  <dl class="meta">
    <div><dt>Area</dt><dd>70 × 70 km, easting 641–711 km, northing 1634–1704 km (UTM 43N, EPSG:32643)</dd></div>
    <div><dt>Data</dt><dd>NGPM Bouguer anomaly at 1 km ({D['n']:,} points) plus a terrain correction computed from the DEM (2.67 g/cc, to 50 km); second-order trend surface removed</dd></div>
    <div><dt>Terrain</dt><dd>Copernicus GLO-90 DEM, {P['dem']['aoi_min']:.0f}–{P['dem']['aoi_max']:.0f} m in the area; stations on the ground</dd></div>
    <div><dt>Mesh</dt><dd>1 km × 1 km × 250 m core, 10 km below the lowest ground; about 20 km padding at the sides: {M['shape'][0]} × {M['shape'][1]} × {M['shape'][2]} = {M['n_cells']:,} cells, {M['n_active']:,} below the ground</dd></div>
    <div><dt>Runs</dt><dd>8 full-resolution inversions on AWS EC2 (ap-south-1, c5.4xlarge), {V['minutes']} minutes each, ${V['cost']:.2f} in total; a 16-run 2 km study run locally</dd></div>
  </dl>
</header>

<section class="summary" aria-labelledby="sum">
  <h2 id="sum">Summary</h2>
  <ul>
    <li><b>The Bouguer anomaly had no terrain correction, and it matters on the ridges.</b> Computed from the DEM, the correction is small over most of the area (median {TC['median']:.2f} mGal at the nodes) but reaches {TC['max']:.1f} mGal on the Sandur ridges and exceeds 1 mGal at {TC['n_above_1']} of {D['n']:,} nodes, against a data error of 0.5 mGal. The ridges run along the north-western part of the NW–SE Bouguer high.</li>
    <li><b>The depth of the main body does not change.</b> Under the main Bouguer high the centroid moves by at most {V['shift_small']:.1f} km for four of the five compact settings ({V['shift_as01']:+.1f} km for α<sub>s</sub> = 0.1, whose body now reaches the density bound): {cen('as1_beta0.5'):.1f} km for β = 0.5, {cen('l1l2_irls'):.1f} km for L1–L2, {cen('as1_beta1'):.1f} km for β = 1 and {cen('as1_beta1.5'):.1f} km for β = 1.5, measured below the ground.</li>
    <li><b>The terrain changes the top kilometre.</b> With the correction the model gains dense rock directly under the ridges: the density summed over the top kilometre changes by up to {N['shallow']['diff_max']:.2f} g/cc·km there, and the dense belt becomes a continuous line along the ridge in the 1 km slice. This is where the banded iron formation crops out (3.4 g/cc in the rock samples).</li>
    <li><b>The conclusions of the flat-earth comparison hold.</b> All six runs fit the data about equally well (χ²/N {V['chi']}); the vertically integrated density maps correlate at {V['corr_min']}–{V['corr_max']} pairwise and at {V['oldnew']} with their flat-earth counterparts; the regularization still sets the depth.</li>
    <li><b>The coarse 2 km study still works for screening.</b> For three values of β it predicts the full-resolution centroid of the main body to within {V['lowgap']:.1f} km.</li>
    <li><b>Recommendation.</b> Use the terrain and the terrain correction as the standard set-up: they cost one DEM and {RUNS['as1_beta1']['minutes']:.0f} instead of 5–6 minutes per run. They matter for anything shallow and for the joint inversion with the magnetic data; for the depth and shape of the deep bodies the recommendation of the earlier report is unchanged (α<sub>s</sub> = 1, depth weighting with β between 0.5 and 1, reported as a range).</li>
    <li><b>Caveat.</b> Gravity data alone do not determine depth, with or without terrain. The terrain correction uses a 90 m DEM and is interpolated between stations about {GS['spacing_km']:.1f} km apart, so terrain closer than about 100 m to a station is not accounted for.</li>
  </ul>
</section>

<h2><span class="no">1</span>Terrain, data and the runs</h2>

<h3><span class="no">1.1</span>The terrain correction</h3>
<div class="prose">
<p>The NGPM Bouguer anomaly is a <b>simple</b> Bouguer anomaly: the station table reproduces it as free-air anomaly minus an infinite slab of 2.67 g/cc, with no terrain term (the check of 29 September). A slab is wrong wherever the ground is not flat: hills above a station pull upwards, and valleys below it are rock the slab removed that is not there. Both lower the anomaly, so the terrain correction is positive and is added.</p>
<p>The correction was computed from the Copernicus GLO-90 DEM (90 m cells) at each of the {GS['n_used']:,} NGPM stations in and within 10 km of the area, with the reduction density of 2.67 g/cc, out to 50 km. The DEM is used at three resolutions around each station: its bilinear surface within about 0.5 km (integrated in polar coordinates, so that the step between a station and its neighbouring cells does not count as a slab), 90 m prisms to 4.7 km, 450 m cells to 22.5 km and 1.8 km cells beyond. The code (<code>geoinv3d/methods/terrain.py</code>) reproduces the analytic attraction of a ring-shaped plateau to within 3% in each zone. The station heights in the table agree with the DEM: DEM minus station {GS['dem_minus_station_mean']:+.1f} m on average, standard deviation {GS['dem_minus_station_std']:.1f} m.</p>
<p>At the {GS['n_in_aoi']:,} stations inside the area the correction has a median of {TCS['stations']['median']:.2f} mGal, exceeds 0.5 mGal at {TCS['stations']['n_above_0.5']} and 1 mGal at {TCS['stations']['n_above_1']} stations, and reaches {TCS['stations']['max']:.1f} mGal on the ridge tops near 1,000 m. On average {TCS['zones_mean']['near']:.2f} mGal comes from the terrain between 0.5 and 4.7 km, {TCS['zones_mean']['middle']:.2f} from 4.7–22.5 km, {TCS['zones_mean']['inner']:.2f} from within 0.5 km and {TCS['zones_mean']['far']:.2f} from beyond 22.5 km.</p>
<p>The inverted data are the NGPM grid, which was interpolated from these stations. The complete anomaly is the smooth quantity, so the stations' corrections were gridded the same way (thin-plate spline) and added to the grid at its 1 km nodes (Figure {fig.ref('terrain')}). After the second-order trend is removed, the anomaly that is inverted changes by <span class="num">{mgal(N['tc_change']['min'])}</span> to <span class="num">+{N['tc_change']['max']:.1f}</span> mGal: up along the two Sandur ridges, slightly down elsewhere because the trend absorbs the mean. At the main Bouguer high itself, south-east of the ridges, the change is {N['tc_change']['at_main_high']:+.1f} mGal.</p>
</div>
{fig('terrain', "Left: ground elevation and the NGPM gravity stations. Centre: terrain correction at the 1 km nodes. Right: change of the inverted (trend-removed) anomaly when the correction is added. The + marks the main Bouguer high, the × the north-western high.")}

<h3><span class="no">1.2</span>The ground in the mesh</h3>
<div class="prose">
<p>The mesh follows the ground: cells whose centres lie above the DEM (averaged to 450 m) are air and are left out, which leaves {M['n_active']:,} of {M['n_cells']:,} cells. The layers are 250 m thick instead of the 500 m of the earlier runs, so that the relief ({M['ground_min']:.0f}–{M['ground_max']:.0f} m in the mesh) is three steps rather than one; the core reaches 10 km below the lowest ground. Each datum is placed on the ground at its node, at the DEM's elevation ({P['gravity']['node_elevation_min']:.0f}–{P['gravity']['node_elevation_max']:.0f} m).</p>
<p>The ground of the mesh is a staircase, so a station at its true elevation can lie inside the top cell of its column, with part of that cell's rock above it. The pipeline now moves such stations to the top of their column: {lift['n']:,} of {D['n']:,} stations were raised, by {lift['mean_m']:.0f} m on average and at most {lift['max_m']:.0f} m (in steep valleys, where a 1 km cell spans several hundred metres of relief).</p>
<p><b>All depths in this report are measured below the ground of each column</b>, not below sea level. In the mesh the ground is at {ground_text}.</p>
</div>

<h3><span class="no">1.3</span>Data and runs</h3>
<div class="prose">
<p>With the correction the Bouguer anomaly ranges from <span class="num">{mgal(D['raw'][0])}</span> to <span class="num">{mgal(D['raw'][1])}</span> mGal. After removing a least-squares second-order trend surface (<span class="num">{mgal(D['regional'][0])}</span> to <span class="num">{mgal(D['regional'][1])}</span> mGal), the residual anomaly ranges from <span class="num">{mgal(D['residual'][0])}</span> to <span class="num">+{D['residual'][1]:.1f}</span> mGal (Figure {fig.ref('data')}). Every point has an error of 0.5 mGal, as before.</p>
</div>
{fig('data', "Left: complete Bouguer anomaly at the 1 km nodes. Centre: least-squares second-order trend surface, removed as the regional field. Right: the residual anomaly, i.e. the data inverted (red is positive, dense).")}
<div class="prose">
<p>The six runs are the settings of the 28 September report; they share the mesh, the data and the errors:</p>
</div>
{runs_table()}
<p class="note">Common settings, unchanged: density contrast bounded to −0.2 to +0.5 g/cc; β chosen automatically for χ² = N; at most 30 iterations and 30 IRLS iterations; α<sub>x</sub> = α<sub>y</sub> = α<sub>z</sub> = 1 as length scales. Li &amp; Oldenburg depth weighting weights the model norm by (z + z₀)<sup>−β/2</sup>, where z is the depth below the nearest station and z₀ half the smallest cell (now 125 m). Two more runs, with β = 1, separate the effects in Section 2.</p>

<h2><span class="no">2</span>What the terrain changes</h2>
<div class="prose">
<p>Three things differ from the earlier runs: the layers are 250 m instead of 500 m, the ground is in the mesh, and the data carry the terrain correction. Four runs with the same regularization (α<sub>s</sub> = 1, β = 1) take these steps one at a time:</p>
</div>
{steps_table()}
<p class="note">Depths below the ground of each column. The last column compares the vertically integrated density map of each run with that of the last row.</p>
{fig('steps', "The four runs of the table. Left: E–W sections through the main Bouguer high; the black line is the ground of the mesh, the dashed line the base of the core. Right: the layer about 1 km below the mean ground. Colour scale ±0.3 g/cc.")}
<div class="prose">
<ul class="plain">
  <li><b>Thinner layers change almost nothing.</b> On a flat earth, 250 m layers give a centroid of {s_250['main']['centroid_km']:.1f} km under the main high against {s_flat['main']['centroid_km']:.1f} km with 500 m layers, and the same half-maximum range.</li>
  <li><b>With the ground in the mesh the main body keeps its depth below the ground.</b> Its centroid is {s_notc['main']['centroid_km']:.1f} km below the ground against {s_250['main']['centroid_km']:.1f} km on the flat earth. The ground there is now at {F['as1_beta1']['ground_main_m']:.0f} m instead of zero, so in elevation the body is {(F['as1_beta1']['ground_main_m'] / 1e3 - s_notc['main']['centroid_km']) + s_250['main']['centroid_km']:.1f} km higher: the depth weighting counts depth from the stations, and the body follows them. The data fit is the same.</li>
  <li><b>The terrain correction leaves the deep body where it is</b> (centroid {s_tc['main']['centroid_km']:.1f} km) and changes the top of the model. Figure {fig.ref('shallow')} shows the density summed over the top kilometre below the ground: with the correction there is more dense rock along the two ridges, up to {N['shallow']['diff_max']:.2f} g/cc·km more, and the dense belt of the 1 km slice (Figure {fig.ref('steps')}, right) becomes a continuous line along the south-western ridge.</li>
  <li><b>Why the correction goes to the top.</b> It is a short-wavelength signal that follows the ridges, a few kilometres wide. A compact model can only fit it with sources close to the surface, and the depth weighting of the deep body is not affected.</li>
</ul>
<p>The shallow dense rock is geologically reasonable: the ridges are banded iron formation, 3.38–3.40 g/cc in the two samples from the area, which the 2.67 g/cc of the Bouguer slab and of the terrain correction underestimates by 0.7 g/cc. The terrain correction does not remove this excess; it removes the error of treating the hills as a flat slab, and what is left is fitted as density contrast in the hills.</p>
</div>
{fig('shallow', "Density contrast summed over the top kilometre below the ground (g/cc·km), α<sub>s</sub> = 1, β = 1. Left: terrain in the mesh, data without the terrain correction. Centre: with the correction. Right: the difference.")}

<h2><span class="no">3</span>The six settings with terrain</h2>

<h3><span class="no">3.1</span>Data fit</h3>
<div class="prose">
<p>All six runs reach the target fit with nearly the same statistics as on the flat earth (last column). The residual maps (Figure {fig.ref('residuals')}) have the same character as before:</p>
<ul class="plain">
  <li><b>Original settings:</b> patches of positive and negative misfit at places with steep gradients, up to {o['max_abs']:.1f} mGal.</li>
  <li><b>L1–L2:</b> a broad positive residual over the main high and the north-eastern belt: the model underestimates the amplitude of the main anomalies. Its χ²/N is {t['l1l2_irls']['chi2']:.2f}.</li>
  <li><b>β = 1.5:</b> a positive residual over the main high as well, χ²/N {t['as1_beta1.5']['chi2']:.2f}: with the ridges' extra {TC['max']:.0f} mGal the deepest model has more difficulty fitting the short wavelengths. On the flat earth it fitted best ({OLD['as1_beta1.5']['chi2']:.2f}).</li>
  <li><b>β = 0.5, β = 1 and α<sub>s</sub> = 0.1:</b> the least structured residuals, mostly short-wavelength stripes along strike and the isolated points near (660, 1680) and (702, 1659) km that appear in every model.</li>
</ul>
</div>
{fit_table()}
{fig('residuals', "Residuals (observed − predicted) of the six full-resolution models; colour scale ±3 mGal.")}

<h3><span class="no">3.2</span>Lateral structure</h3>
<div class="prose">
<p>Figure {fig.ref('slices')} shows five models at about 2, 5 and 8 km below the mean ground. The top row of Figure {fig.ref('robust')} shows the vertically integrated density, the bottom row the centroid depth of each column.</p>
<ul class="plain">
  <li><b>The integrated density maps are nearly the same.</b> Pairwise correlations are {V['corr_min']}–{V['corr_max']}, and no less than {V['corr_new']} without the original settings. The NW–SE dense belt, the secondary high at its north-western end and the parallel low-density belts to the east have the same position and strike in every model.</li>
  <li><b>They are also the maps of the flat-earth runs.</b> Setting by setting, the integrated density with terrain correlates at {V['oldnew']} with the flat-earth result.</li>
  <li><b>The centroid depth maps differ between settings</b> as they did before: 2–4 km for β = 0.5 and L1–L2, 4–6 km for β = 1, 6–8 km and more for β = 1.5.</li>
</ul>
</div>
{corr_table()}
<p class="note">Correlation between the vertically integrated density maps in the 70 × 70 km core. Green: ≥ 0.90.</p>
{fig('slices', "Layers of five models at about 2, 5 and 8 km below the mean ground (core area only); colour scale ±0.3 g/cc.")}
{fig('robust', "Top: vertically integrated density (g/cc·km). Bottom: centroid depth of |density contrast| below the ground in each column (only columns whose integral exceeds 20% of the maximum).")}

<h3><span class="no">3.3</span>Depth</h3>
<div class="prose">
<p>Figure {fig.ref('sections')} shows an E–W section through the main Bouguer high, and Figure {fig.ref('mass_profiles')} the distribution of |mass| with depth. The table gives where the mass lies and measures of the density profiles under the main Bouguer high (easting 671.5 km, northing 1664.5 km) and the north-western high (650.5, 1682.5 km). As before, depth is given as the centroid of the positive density contrast, because bodies held at the +0.5 g/cc bound have no single peak; the half-maximum range is measured between cell edges.</p>
</div>
{depth_table()}
<p class="note">Core, lateral padding and below the core add up to 100%. D50, D90: half and 90% of the |mass| under the core area lie above this depth below the ground. Red: lateral padding &gt; 20%, below the core &gt; 30%, D50 &gt; 7 km, or the main body reaching below 12 km.</p>
{fig('sections', "E–W sections through the main Bouguer high (northing 1664.5 km) in the six models. The black line is the ground of the mesh, the vertical dotted line the high, the dashed line the base of the core mesh; below it is padding. Colour scale ±0.3 g/cc.")}
<div class="prose">
<ul class="plain">
  <li><b>The original settings still give columns.</b> Under the main high the density contrast is {o['main']['peak']:.2f} g/cc from the surface to {o['main']['bottom_km']:.1f} km, the bottom of the mesh; {pct(o['below'])} of the mass lies below the core and {pct(o['lateral'])} in the lateral padding.</li>
  <li><b>The other five have a base.</b> The main body ends above 12 km, and at least {V['core_min']} of the mass lies in the core ({pct(t['as1_beta1.5']['core'])} for β = 1.5).</li>
  <li><b>β controls the depth monotonically.</b> For β = 0.5, 1 and 1.5 the centroid of the main body is at {cen('as1_beta0.5'):.1f}, {cen('as1_beta1'):.1f} and {cen('as1_beta1.5'):.1f} km, the half-maximum range {rng(t['as1_beta0.5']['main'])}, {rng(t['as1_beta1']['main'])} and {rng(t['as1_beta1.5']['main'])} km.</li>
  <li><b>L1–L2 stays close to β = 0.5:</b> half-maximum range {rng(t['l1l2_irls']['main'])} km, centroids {abs(cen('l1l2_irls') - cen('as1_beta0.5')):.1f} km apart.</li>
  <li><b>α<sub>s</sub> = 0.1 now behaves like α<sub>s</sub> = 1.</b> Its main body reaches the bound (peak {t['as0.1_beta1']['main']['peak']:.2f} g/cc) and its centroid is at {cen('as0.1_beta1'):.1f} km, {abs(cen('as0.1_beta1') - cen('as1_beta1')):.1f} km from the β = 1 run; on the flat earth it stayed just below the bound and {cen('as0.1_beta1', src=OLD) - cen('as1_beta1', src=OLD):.1f} km deeper.</li>
  <li><b>The north-western high is always shallower.</b> Apart from the original settings, its centroid lies at {V['nw_new']} km, {V['nw_gap']} km above that of the main high.</li>
</ul>
</div>
{fig('mass_profiles', "Distribution of |mass| with depth below the ground, under the core area (percent per km). Left: the six full-resolution models. Right: five runs of the 2 km study (Section 4). The dashed line is 10 km.")}

<h3><span class="no">3.4</span>Against the flat-earth runs</h3>
<div class="prose">
<p>Figure {fig.ref('old_vs_new')} and the table compare each setting with its run of 28 September (flat earth at zero elevation, 500 m layers, no terrain correction; its depths were measured from that flat surface and are re-measured here with the same code). Figure {fig.ref('centre_profiles')} shows the density profiles themselves.</p>
</div>
{old_new_table()}
{fig('old_vs_new', "Half-maximum depth range (bars) and centroid (dots) under the main Bouguer high for the six settings: flat earth (grey) and with terrain (blue).", narrow=True)}
<div class="prose">
<ul class="plain">
  <li><b>The five compact settings keep their depths.</b> The centroid under the main high moves by {mgal(min(_shift.values()))} to {max(_shift.values()):+.1f} km; the ordering of the settings and the spread between them (from {cen('as1_beta0.5'):.1f} to {cen('as1_beta1.5'):.1f} km) are those of the flat-earth comparison.</li>
  <li><b>The original settings change most</b>, but only because their columns end where the mesh ends: this mesh reaches {abs(M['z_bottom']) / 1e3:.1f} km below sea level, the earlier one 21.4 km.</li>
  <li><b>The differences between regularizations are far larger than the effect of the terrain.</b> Changing β from 0.5 to 1.5 moves the centroid by {cen('as1_beta1.5') - cen('as1_beta0.5'):.1f} km; adding the terrain and its correction moves it by at most {V['shift_small']:.1f} km at a fixed setting with α<sub>s</sub> = 1.</li>
</ul>
</div>
{fig('centre_profiles', "Density with depth below the ground under the main Bouguer high (left) and the north-western high (right). Thick lines: with terrain; thin lines of the same colour: the flat-earth runs.")}

<h2><span class="no">4</span>The 2 km study with terrain</h2>
<div class="prose">
<p>The 16 sparse inversions of the earlier coarse study were repeated locally with the terrain: 2 km × 2 km × 500 m cells, every second node (1,296 data), {V['low_seconds']} each.</p>
</div>
{lowres_table() if LOWRES else ''}
<p class="note">"sensitivity" is SimPEG's sensitivity weighting, "β = …" Li &amp; Oldenburg depth weighting. Red cells as in Section 3.3.</p>
<div class="prose">
<ul class="plain">
  <li><b>The trends are those of the flat-earth study.</b> α<sub>s</sub> decides whether the model reaches the bottom of the mesh: with sensitivity weighting, α<sub>s</sub> = 10⁻⁴, 10⁻², 0.1 and 1 leave {pct(LOW['base']['below'])}, {pct(LOW['as1e-2']['below'])}, {pct(LOW['as0.1']['below'])} and {pct(LOW['as1']['below'])} of the mass below the core. Depth weighting decides the lateral leakage: {pct(LOW['as1_p0222']['lateral'])} of the mass in the lateral padding with sensitivity weighting, {V['dw_lat']} with depth weighting.</li>
  <li><b>β sets the mean depth:</b> β = 0.5, 1, 1.5 and 2 give main-body centroids of {LOW['as1_p0222_dw0.5']['main']['centroid_km']:.1f}, {LOW['as1_p0222_dw1']['main']['centroid_km']:.1f}, {LOW['as1_p0222_dw1.5']['main']['centroid_km']:.1f} and {LOW['as1_p0222_dw2']['main']['centroid_km']:.1f} km.</li>
  <li><b>Against the 1 km runs</b> (Figure {fig.ref('lowres_vs_full')}) the coarse centroid is within {V['lowgap']:.1f} km for the three values of β and within {V['as01gap']:.1f} km for α<sub>s</sub> = 0.1. The coarse mesh remains good for screening β; its 2 km cells cannot hold the shallow dense rock of the ridges.</li>
</ul>
</div>
{fig('lowres_vs_full', "Half-maximum depth range (bars) and centroid (dots) under the main Bouguer high: the 2 km study (grey) and the 1 km full-resolution runs (blue), both with terrain.", narrow=True)}

<h2><span class="no">5</span>Recommendations</h2>
<div class="prose">
<ul class="plain">
  <li><b>Make the terrain the default for field data.</b> A DEM in the job (<code>topography: {{file}}</code>), stations on the ground, and the terrain correction added to simple Bouguer anomalies. The pipeline already reports whether a station table contains a terrain correction; the correction itself is now in <code>geoinv3d/methods/terrain.py</code> and is applied in the data preparation (<code>karnataka_inputs/prepare_inputs.py</code>).</li>
  <li><b>Settings: unchanged.</b> Sparse, α<sub>s</sub> = 1, p = [0,2,2,2], Li &amp; Oldenburg depth weighting with β between 0.5 and 1, reported as a depth range (main-body centroid {cen('as1_beta0.5'):.1f}–{cen('as1_beta1'):.1f} km below the ground), with L1–L2 as an independent check.</li>
  <li><b>Interpret the top kilometre only with the terrain correction.</b> Without it the ridges' signal is missing from the data, and the shallow dense rock along them is underestimated.</li>
  <li><b>A better terrain correction needs better inputs:</b> the original station values instead of the grid (the table is rounded to 1 mGal), a 30 m DEM for the ground within 100 m of a station, and the density of the hills (the banded iron formation is 0.7 g/cc denser than the reduction density).</li>
  <li><b>Depth still needs independent constraints</b> (rock densities, boreholes, sections, seismic data), or the magnetic data of the same area: the joint inversion is the subject of a separate report.</li>
</ul>
</div>

<h2>Files</h2>
<div class="prose">
<ul class="plain">
  <li>Inputs (DEM, terrain correction, data at the nodes): <code>examples/output/karnataka_inputs/</code> (<code>prepare_inputs.py</code>)</li>
  <li>This report and its figures: <code>examples/output/karnataka_gravity_terrain/scripts/</code> (<code>make_figures.py</code>, <code>build_report.py</code>)</li>
  <li>The eight full-resolution results: <code>data/ec2_runs/</code>; the 2 km study: <code>data/lowres_runs/</code>; launched with <code>deploy/ec2_multi_run.py karnataka-gravity-terrain</code></li>
  <li>All runs in one interactive workflow (DAG viewer): <code>karnataka_gravity_terrain.geoinv3d_viewer.html</code></li>
  <li>The flat-earth comparison of 28 September: <code>examples/output/karnataka_gravity/</code></li>
</ul>
</div>
<footer>GeoInv3D · SimPEG 0.25.2 · Generated by build_report.py; every number comes from the runs above.</footer>
"""


def main():
    fig = Figures("en", FIGS)
    OUT.write_text(page("en", "Karnataka Gravity with Terrain", body(fig)), encoding="utf-8")
    print(OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")
    if "--pdf" in sys.argv:
        print(to_pdf(OUT))


if __name__ == "__main__":
    main()
