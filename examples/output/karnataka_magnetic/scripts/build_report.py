"""Build the Karnataka magnetic report: English HTML, optionally the PDF.

    py examples/output/karnataka_magnetic/scripts/make_figures.py
    py examples/output/karnataka_magnetic/scripts/build_report.py [--pdf]

Every number in the text comes from figures/numbers.json, except the table of the synthetic
test of 28 September (LOGBOOK.md) and the published values of Section 6 (the sources cited there;
localities and references in karnataka_inputs/shared/literature.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "karnataka_inputs" / "shared"))
from style import Figures, esc, page, table, to_pdf  # noqa: E402
from literature import (GOLD, LEGEND, LIT_CSS, MINES, RIDGE_MIN_M, RIDGE_RELIEF_M, cite,  # noqa: E402
                        references_html)

FIGS = ROOT / "figures"
N = json.loads((FIGS / "numbers.json").read_text(encoding="utf-8"))
F, D, P, M, G = N["full"], N["data"], N["prep"]["magnetic"], N["mesh"], N["gravity"]
FULL = [("sens", "sparse, sensitivity weighting"), ("beta0.5", "sparse, β = 0.5"), ("beta1", "sparse, β = 1"),
        ("beta1.5", "sparse, β = 1.5"), ("beta2", "sparse, β = 2"), ("beta3", "sparse, β = 3"),
        ("l1l2_irls", "L1–L2 (IRLS)")]
TESTS = [("beta1", "Reference (β = 1)"),
         ("as0.1_beta1", "α<sub>s</sub> = 0.1"), ("beta1_ub0.3", "Upper bound 0.3 SI"),
         ("beta1_err2", "Error 2% + 5 nT"), ("beta1_h500", "Continued to 500 m above the ground"),
         ("beta1_raw80", "As flown, 80 m above the ground (not continued)"), ("beta1_flat", "Flat earth")]
KEYS = [k for k, _ in FULL]
SHALLOW = ["beta0.5", "beta1", "l1l2_irls"]
OUT = ROOT / "karnataka_magnetic_report_en.html"
RUNS = N["runs"]
LIT = N["literature"]
REF_KEYS = ["MM93", "GSI", "IJERT", "MEAI", "ROM", "IBMK", "IBMN", "GEM", "SK18", "SB14", "SIN20", "BHAT25", "NMET",
            "MGR23"]


def label(key):
    return dict(FULL)[key]


def nt(x):
    return f"{x:,.0f}".replace("-", "−")


def pct(x):
    return f"{100 * x:.0f}%"


def span(values, fmt="{:.2f}"):
    return f"{fmt.format(min(values))}–{fmt.format(max(values))}"


def km(x):
    """A distance in the text: 1 decimal, without a trailing .0."""
    return f"{x:.1f}".removesuffix(".0")


def warn(text, bad):
    return (text, "n warn" if bad else "n")


def box(key, which="sandur"):
    return F[key][f"box_{which}"]


CS = N["integrated_corr_smooth"]
CR = N["integrated_corr"]
_ix = {k: i for i, k in enumerate(CS["keys"])}


def corr(a, b, m=CS):
    return m["matrix"][_ix[a]][_ix[b]]


V = dict(
    chi=span([F[k]["chi2"] for k in KEYS]), rms=span([F[k]["rms"] for k in KEYS], "{:.0f}"),
    shallow_corr=span([corr(a, b) for a in SHALLOW for b in SHALLOW if a < b]),
    shallow_corr_raw=span([corr(a, b, CR) for a in SHALLOW for b in SHALLOW if a < b]),
    minutes=span([r["minutes"] for r in RUNS.values()], "{:.1f}"),
    cost=sum(r["cost_usd"] for r in RUNS.values()),
    moment=span([box(k)["total"] for k in KEYS if k != "beta3"], "{:.0f}"),
)


# ---------------------------------------------------------------- tables

def runs_table():
    head = ["Run", "Regularization", "Depth weighting", "χ²/N", "RMS (nT)", "Max |residual| (nT)", "Iterations",
            "EC2 time (min)"]
    rows = []
    for k in KEYS:
        s, f = F[k]["settings"], F[k]
        reg = ("elastic net (Utsugi 2019), L1 share 0.8, solved by IRLS" if s["regularization_type"] == "l1l2"
               else "sparse (IRLS), α<sub>s</sub> = 1, p = [0, 2, 2, 2]")
        w = (f"Li &amp; Oldenburg, β = {s['depth_weighting_exponent']:g}" if s.get("depth_weighting") == "depth"
             else "sensitivity")
        rows.append([label(k), (reg, "wrap"), (w, ""), f"{f['chi2']:.2f}", f"{f['rms']:.0f}", nt(f["max_abs"]),
                     str(f["n_iterations"]), f"{RUNS[k]['minutes']:.1f}"])
    return table(head, rows, numeric_from=3)


def where_table():
    head = ["Run", "Core", "Lateral padding", "Below the core", "D50 km", "D90 km",
            "Sandur belt: centroid km", "10–90% depth range km", "susceptibility × volume (SI·km³)", "largest value (SI)"]
    rows = []
    for k in KEYS:
        f, b = F[k], box(k)
        rows.append([label(k), warn(pct(f["core"]), f["core"] < 0.5), warn(pct(f["lateral"]), f["lateral"] > 0.3),
                     warn(pct(f["below"]), f["below"] > 0.3), warn(f"{f['D50']:.1f}", f["D50"] > 7), f"{f['D90']:.1f}",
                     f"{b['centroid_km']:.1f}", f"{b['d10_km']:.1f}–{b['d90_km']:.1f}", f"{b['total']:.0f}", f"{b['max']:.2f}"])
    return table(head, rows)


def corr_table():
    short = ["sens.", "β = 0.5", "β = 1", "β = 1.5", "β = 2", "β = 3", "L1–L2"]
    m = CS["matrix"]
    rows = [[short[i]] + [("—", "n") if i == j else (f"{m[i][j]:.2f}", "n good" if m[i][j] >= 0.85 else "n")
                          for j in range(len(short))] for i in range(len(short))]
    return table([""] + short, rows, compact=True)


def tests_table():
    head = ["Run (sparse, β = 1)", "χ²/N", "RMS (nT)", "Core", "Below the core", "Sandur belt: centroid km",
            "susceptibility × volume (SI·km³)", "largest value (SI)",
            "Integrated susceptibility at 5 km: correlation with the reference"]
    rows = []
    for k, lab in TESTS:
        f, b = F[k], box(k)
        rows.append([(lab, "wrap"), warn(f"{f['chi2']:.2f}", f["chi2"] > 1.2), f"{f['rms']:.0f}", pct(f["core"]),
                     warn(pct(f["below"]), f["below"] > 0.3), f"{b['centroid_km']:.1f}", f"{b['total']:.0f}",
                     f"{b['max']:.2f}", f"{N['integrated_tests_smooth'][k]:.2f}"])
    return table(head, rows)


SYNTHETIC = table(
    ["Run (synthetic test, 28 September)", "χ²/N", "Correlation with the true model", "Shallow block, true 2.0 km",
     "Deep block, true 5.5 km", "Dyke dip, true 45°"],
    [["sparse, sensitivity weighting", "1.11", "0.21", "3.9 km", "8.5 km", "64°"],
     ["sparse, β = 2", "1.07", "0.24", "3.7 km", "8.0 km", "62°"],
     ["sparse, β = 3", "1.31", "0.17", "4.9 km", "8.6 km", "66°"],
     ["L1–L2 (IRLS)", "1.00", "0.60", "3.9 km", "6.7 km", "65°"]], compact=True)


def localities_table():
    """Section 6.2: distances from the published localities to strong columns of the models."""
    head = ["Locality", "Source", "E, N (km)", "To strong susceptibility (km)", "Area as close",
            "To dense column (km)", "Area as close", "To strong MVI amplitude (km)", "Area as close"]
    what = {**{n: f"iron, {op}" if op else "iron" for n, _, _, _, op in MINES},
            "Joga": "gold in BIF quartz veins", "Taranagar": "gold in sulphidic chert"}
    source = {n: k for n, _, _, k, *_ in MINES + GOLD}
    rows = []
    for i, r in enumerate(LIT["distances"]["chi"]):
        cells = [f"{r['name']} ({what[r['name']]})", cite(source[r["name"]]), f"{r['x']:.1f}, {r['y']:.1f}"]
        for k in ("chi", "rho", "mvi"):
            q = LIT["distances"][k][i]
            cells += [f"{q['distance_km']:.1f}", pct(q["area_as_close"])]
        rows.append(cells)
    return table(head, rows, numeric_from=2)

DIST = {k: {r["name"]: r for r in LIT["distances"][k]} for k in ("chi", "rho", "mvi")}
FAR, RG, MINE_NAMES = LIT["farthest_mine"], LIT["ridges"], LIT["mines"]
GOLD_NAMES = [r["name"] for r in LIT["distances"]["chi"] if r["kind"] == "gold"]
GOLD_WHAT = {n: w for n, _, _, _, w in GOLD}
WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "all four"}


def area_range(k, names):
    """The share of the area as close to a strong column as the closest and the farthest of ``names``."""
    a = [DIST[k][n]["area_as_close"] for n in names]
    return f"{pct(min(a))} to {pct(max(a))}" if pct(min(a)) != pct(max(a)) else pct(a[0])


def geology(fig):
    """Section 6, the literature cross-check of 1 October: the models against the published geology."""
    th, W, K, C, U = LIT["thresholds"], LIT["within_dense"], LIT["kumaraswamy"], LIT["belt_contrast"], LIT["underfit"]
    on_ridge = max(LIT["mine_to_ridge_km"].values())
    on_ridge = "them" if on_ridge < 0.45 else f"or within {km(on_ridge)} km of them"     # 0.45 km: one DEM cell
    outside_far = 1 - RG["chi"]["outside_within_2km"]
    mvi_ok = [m for m in MINE_NAMES if DIST["mvi"][m]["distance_km"] <= FAR["chi"] + 1e-9]
    mvi_out = [m for m in MINE_NAMES if m not in mvi_ok]
    mvi = (f"The MVI amplitude, which has fewer strong columns ({RG['mvi']['n']} against {RG['chi']['n']}), puts "
           f"{WORDS[len(mvi_ok)]} of them within {km(FAR['chi'])} km of one ({area_range('mvi', mvi_ok)} of the area)"
           + ("" if not mvi_out else " and " + ", ".join(f"{m} {km(DIST['mvi'][m]['distance_km'])} km" for m in mvi_out)
              + " away") + ".")
    gold_far = [g for g in GOLD_NAMES if DIST["chi"][g]["distance_km"] > FAR["chi"] + 1e-9]
    gold_head = ("Gold is not where the magnetic rock is." if len(gold_far) == len(GOLD_NAMES)
                 else "The gold is only partly where the magnetic rock is." if gold_far
                 else "The gold is where the magnetic rock is, too.")
    jo, ta = (DIST["chi"][g] for g in GOLD_NAMES)
    gold_dense = max(DIST["rho"][g]["distance_km"] for g in GOLD_NAMES)
    gold_dense = "in the dense belt" if gold_dense == 0 else f"in, or within {km(gold_dense)} km of, the dense belt"
    ratio = C["chi_near"] / C["chi_between"]
    chi_word = ("higher at the ridges" if ratio >= 1.25 else "slightly higher at the ridges" if ratio > 1.02
                else "about the same at the ridges" if ratio >= 0.98 else "lower at the ridges")
    centroids = [box(k)["centroid_km"] for k in SHALLOW]
    mag = LIT["mvi"]
    return f"""
<h2><span class="no">6</span>Comparison with the published geology</h2>
<div class="added">Added 1 October 2026 · literature cross-check</div>
<div class="prose">
<p>The models were built without the geological map. This section asks whether the magnetic rock of the models is where the iron formation of the Sandur belt is mapped and mined, and what the published work says about the parts of the models that cannot be checked from the data.</p>
</div>
<h3><span class="no">6.1</span>Where the iron formation is mapped</h3>
<div class="prose">
<p>The Sandur belt is a ring of elongated hills, 900 to 1,050 m above sea level, around a lower centre: the Copper Mountain range in the east and the Sandur range in the west {cite('MEAI')}. The hills are held up by the banded iron formation, which carries the iron ore of six ranges (Donimalai, Kumaraswamy, Ramandurg, Kanavehalli, Devagiri and Thimmappanagudi) and, with it, the manganese ore of the Deogiri Formation, discontinuous over about 40 km of strike {cite('GSI', 'IJERT')}. The Donimalai iron formation is made of amphibole, hematite, magnetite and chert {cite('ROM')}.</p>
<p>In Figure {fig.ref('localities')} the narrow highs and lows of the anomaly as flown follow the ridges of the DEM, and every mine lies on {on_ridge}. The two limbs and the south-eastern closure of Section 2.4 are the western range, the eastern range and the Kumaraswamy–Donimalai hills.</p>
</div>
{fig('localities', f"Published localities on the ground elevation (left) and on the magnetic anomaly as flown, 80 m above the ground (right). {LEGEND} Sources: {cite('IBMK', 'IBMN', 'GEM', 'SB14', 'SIN20', 'NMET')}.")}
<div class="prose">
<p>Figure {fig.ref('schematic')} draws the ridges of the DEM as a schematic map and lays its outline over the anomaly and the model. The magnetic rock of the model is drawn to the ridges but not confined to them. Of its strongly magnetic columns (integrated susceptibility ≥ {th['chi']:g} SI·km, the 5 km edge band left out), {pct(RG['chi']['near'])} lie within {km(RG['near_km'])} km of a ridge, where {pct(RG['area_near'])} of the area does, and {pct(RG['chi']['in_belt'])} inside the envelope of the ridges, which covers {pct(RG['area_in_belt'])} of the area; for the MVI amplitude the two shares are {pct(RG['mvi']['near'])} and {pct(RG['mvi']['in_belt'])}. {"Most" if outside_far > 0.5 else "Part"} of the rest lies away from the belt ({pct(outside_far)} of the strongly magnetic columns outside the envelope are more than 2 km from it): the north–south sheets and the corner bodies of Section 2.4.</p>
</div>
{fig('schematic', f"(a) A schematic drawn for this report from the 450 m DEM of the inputs; it is not a geological map. The ridges (dark green) are where the ground stands more than {RIDGE_RELIEF_M:.0f} m above the median of a 12 km window, and above {RIDGE_MIN_M:.0f} m; they are read as ridges of iron formation because the iron formation holds up the hills of the belt {cite('MEAI')}. The outline of the schist belt (light green) is the envelope of the ridges, and the central metavolcanic terrane between them is labelled after the structural study of the belt {cite('MM93')}. The mapped belt continues north-west beyond the envelope. (b) The magnetic anomaly as flown, 80 m above the ground, and (c) the vertically integrated susceptibility (β = 1), with the envelope (solid line) and the ridges (dotted) of (a). {LEGEND}")}

<h3><span class="no">6.2</span>Do the models put magnetic rock under the mines?</h3>
{fig('localities_models', f"The published localities on the vertically integrated density contrast (left, gravity with terrain, β = 1) and susceptibility (right, β = 1) of Figure {fig.ref('gravity')}. Green lines: the thresholds of the table below, {th['rho']:g} g/cc·km and {th['chi']:g} SI·km. {LEGEND}")}
{localities_table()}
<p class="note">Distance from each locality to the nearest 1 km column with integrated susceptibility ≥ {th['chi']:g} SI·km (β = 1, Figure {fig.ref('gravity')}), integrated density contrast ≥ {th['rho']:g} g/cc·km (gravity with terrain, β = 1) and integrated MVI amplitude ≥ {th['mvi']:g} SI·km (the magnetization-vector run with β = 1 of the joint report, <code>data/ec2_runs/beta1_mvi</code>). "Area as close" is the share of the area ({LIT['edge_km']:.0f} km edge band excluded) that lies at least as close to such a column: a small share means the locality is not where a random point would fall. The distances are computed from the models, between the centres of the 1 km columns, so they are good to one cell; the positions are good to about 1 km.</p>
<div class="prose"><ul class="plain">
  <li><b>All four iron-ore mines lie on or within {km(FAR['chi'])} km of a strongly magnetic column</b>, a condition met by only {area_range('chi', MINE_NAMES)} of the area, and on or within {km(FAR['rho'])} km of a dense column ({area_range('rho', MINE_NAMES)} of the area). {mvi} Kumaraswamy, at the south-eastern closure, sits on a column of {K['value']:.1f} SI·km, stronger than {pct(K['share_box_below'])} of the columns of the Sandur belt box of Section 2.2; at 5 km resolution the strongest magnetic rock of the belt lies {km(K['smooth_max_km'])} km from it.</li>
  <li><b>Within the belt the test is weaker.</b> Of the cells within {km(W['dense_km'])} km of a dense column (edge band excluded), {pct(W['share'])} are also within {km(W['magnetic_km'])} km of a strongly magnetic one. The mines confirm that the models place the magnetic rock in the right belt and on its margins; at 1 km cells they cannot show that the models pick the mined ridge rather than its neighbour.</li>
  <li><b>{gold_head}</b> The Joga occurrence ({GOLD_WHAT['Joga']}) is {km(jo['distance_km'])} km from the nearest strongly magnetic column, a distance within which {pct(jo['area_as_close'])} of the area lies, and the Taranagar occurrence ({GOLD_WHAT['Taranagar']}) {km(ta['distance_km'])} km ({pct(ta['area_as_close'])}) {cite('SB14', 'SIN20')}. Both lie {gold_dense}. Sulphide-facies iron formation, which hosts this gold, carries little magnetite, so a weak magnetic response there is expected.</li>
</ul></div>

<h3><span class="no">6.3</span>What the published work says about the rest</h3>
<div class="prose"><ul class="plain">
  <li><b>The synform.</b> Section 2.3 reads the bowl of magnetic rock as a synform with iron formation on its limbs, the traditional view of the belt. A structural study disputes it: two metasedimentary belts flank a central metavolcanic terrane and do not join at either end, bedding is steep (strike 115° to 150° in the east), and the early folds are near-isoclinal and plunge about 45° to the north {cite('MM93')}. The stratigraphy points the same way. The iron formation is a layer of the greenstone succession, not a body enclosed by it: a greenstone belt is the whole Archaean supracrustal pile of mafic and ultramafic volcanics, minor felsic volcanics and sediments, and its iron formation, mostly of Algoma type (single layers a few metres to a few hundred metres thick and kilometres to tens of kilometres long), is a chemical sediment interbedded with volcanics, chert and clastics, typically laid down at the end of a volcanic cycle. It lies conformably in the upper part of the succession and hosts the ore: the supergene iron ore of Kumaraswamy and Donimalai, and the gold of Joga and Taranagar in iron formation or sulphidic chert. In the west of the Sandur belt the sequence runs from metabasalt at the base through quartzite, carbonate, greywacke and argillite to manganiferous phyllite with chert, with the iron formation at the top {cite('MM93')}, and the Geological Survey places the assemblage in the Bababudan type, iron formation over mafic volcanics and quartzite {cite('GSI')}. Stratigraphically above is not shallower, though: with steep bedding and near-isoclinal D1 folds, the iron formation appears in section as thin, near-vertical sheets on both sides of the belt, not as a layer over the greenstone. In a simple synform the youngest unit, the iron formation, would lie in the core of the belt and the older metabasalt outside it. The models show the reverse, more clearly in the dense rock than in the magnetic: inside the envelope of the Sandur belt (Figure {fig.ref('schematic')}) the integrated density contrast is higher between the ridges than within {km(RG['near_km'])} km of them ({C['rho_between']:.1f} against {C['rho_near']:.1f} g/cc·km), the integrated susceptibility is {chi_word} ({C['chi_near']:.2f} against {C['chi_between']:.2f} SI·km), and the magnetic sheets reach the surface at the ridges (Section 2.3). This fits the reading of Mukhopadhyay &amp; Matin, two iron-bearing metasedimentary belts flanking a central metavolcanic terrane, better than a simple synform: in plan the models are consistent with the stratigraphy. The base of the bowl should still not be interpreted; the steep sheets reaching the surface are the reliable part.</li>
  <li><b>Depth of the ore.</b> The ore bodies are shallow: at Kumaraswamy two E–W bands 1.2 and 3.5 km long with a mean drilled depth of 62 to 70 m {cite('IBMK')}; at Donimalai supergene enrichment reaches ore grade in the upper 170 m {cite('ROM')}. A ground magnetic survey in the western part of the belt put the tops of two iron-formation bands at about 70 and 130 m {cite('SK18')}. All of this lies within the top cell of the mesh (250 m). The centroid depths of Section 2 ({min(centroids):.1f} to {max(centroids):.1f} km for the three recommended settings) describe the belt of iron formation as a whole, not the ore.</li>
  <li><b>Remanence.</b> The underfitted high of Section 2.1 lies at and south of the Kumaraswamy leases, the richest part of the belt: of the {U['n']} nodes the β = 1 model underfits by more than {U['threshold_nT']:.0f} nT, {U['within_5km_of_kumaraswamy']} lie within 5 km of the mine, {U['south_of_kumaraswamy']} of them south of it. The ore is a hematite and magnetite assemblage altered from the iron formation {cite('ROM', 'BHAT25')}. No palaeomagnetic measurements from the belt were found, so the magnetization direction of the MVI run of the joint report (I {mag['inclination']:.0f}°, D {nt(mag['declination'])}°) is untested.</li>
  <li><b>The north-eastern belt.</b> It runs south-east into the Mincheri block, where a reconnaissance survey mapped narrow ridges of banded hematite quartzite, ferruginous chert and magnetite quartzite of low grade, with sporadic copper (up to 4,767 ppm in one sample) {cite('NMET')}. Maurya et al. report a previously unmapped arm of the schist belt from gravity and magnetic data {cite('MGR23')}; whether it is this belt cannot be told from the abstract.</li>
  <li><b>The north–south sheets and the corner bodies</b> of Section 2.4 have no counterpart in the sources found, and they lie outside the envelope of the ridges (Figure {fig.ref('schematic')}). This supports the caution given there.</li>
</ul></div>
<div class="callout"><b>For this report.</b> The magnetic models find the iron formation where it is mapped and mined: both ranges, the south-eastern closure and every iron mine. What they add beyond the map (depth, the base of the bowl, cell values) is not confirmed by the literature, and the ore itself is far below their resolution.</div>
"""


# ---------------------------------------------------------------- text

def body(fig):
    t = F
    sw = P["short_wavelength_share"]
    raw, up = P["raw_grid"], P["continued_at_nodes"]
    r80, ref = F["beta1_raw80"], F["beta1"]
    res = N["residual"]
    return f"""
<header>
  <div class="eyebrow">GeoInv3D · field-data test · 30 September 2026 · literature cross-check 1 October 2026</div>
  <h1>Karnataka Magnetic Inversion: Comparing Regularizations</h1>
  <p class="lede">The airborne magnetic data of the area of the gravity comparison, continued upwards and inverted at full resolution on AWS EC2, with the terrain in the mesh. Seven regularization settings are compared for data fit, lateral structure and depth, and six further runs test the choices made in preparing the data.</p>
  <dl class="meta">
    <div><dt>Area</dt><dd>70 × 70 km, easting 641–711 km, northing 1634–1704 km (UTM 43N, EPSG:32643)</dd></div>
    <div><dt>Data</dt><dd>Total magnetic intensity anomaly, 37.5 m grid flown 80 m above the ground; continued upwards to 1 km above the ground and sampled at 1 km: {D['n']:,} points; second-order trend surface removed</dd></div>
    <div><dt>Field</dt><dd>IGRF 2020 at the centre: 42,100 nT, inclination 19.3°, declination −1.4° to grid north; induced magnetization only, susceptibility 0 to 1 SI</dd></div>
    <div><dt>Mesh</dt><dd>As for gravity with terrain: 1 km × 1 km × 250 m core below the Copernicus DEM, {M['shape'][0]} × {M['shape'][1]} × {M['shape'][2]} = {M['n_cells']:,} cells, {M['n_active']:,} below the ground</dd></div>
    <div><dt>Runs</dt><dd>13 full-resolution inversions on AWS EC2 (ap-south-1, c5.4xlarge), {V['minutes']} minutes each, ${V['cost']:.2f} in total</dd></div>
  </dl>
</header>

<section class="summary" aria-labelledby="sum">
  <h2 id="sum">Summary</h2>
  <ul>
    <li><b>The data have to be continued upwards before they are thinned.</b> As flown, {pct(sw['80 m'])} of the grid's variance is at wavelengths shorter than 2 km, which sampling at 1 km aliases; at 1 km above the ground {100 * sw['1000 m']:.2f}% is left. Inverted as flown, the data cannot be fitted (χ²/N {r80['chi2']:.2f}, RMS {r80['rms']:.0f} nT) and the model is smeared to depth.</li>
    <li><b>The depth weighting decides where the susceptibility goes, and the textbook value fails.</b> With the Li &amp; Oldenburg exponent β = 3 usually quoted for magnetic data, {pct(t['beta3']['below'])} of the susceptibility × volume lies below the 10 km core of the mesh; with β = 2, {pct(t['beta2']['below'])}; with sensitivity weighting, {pct(t['sens']['below'])}. With β = 0.5 and 1 and with L1–L2 it is {pct(min(t[k]['below'] for k in SHALLOW))}–{pct(max(t[k]['below'] for k in SHALLOW))}, and {pct(min(t[k]['core'] for k in SHALLOW))}–{pct(max(t[k]['core'] for k in SHALLOW))} is in the core.</li>
    <li><b>All seven fit the data equally</b> (χ²/N {V['chi']}), so the fit cannot choose. The centroid of the magnetic rock of the Sandur belt lies at {box('beta0.5')['centroid_km']:.1f} km for β = 0.5, {box('beta1')['centroid_km']:.1f} km for β = 1, {box('l1l2_irls')['centroid_km']:.1f} km for L1–L2, {box('beta1.5')['centroid_km']:.1f} km for β = 1.5 and {box('beta3')['centroid_km']:.1f} km for β = 3: in every case shallower than the dense body there ({G['box_sandur_density_centroid']:.1f} km in the gravity model with β = 1).</li>
    <li><b>The residuals are not random.</b> Every model leaves the positive anomaly on the southern side of the Sandur belt underfitted by more than 150 nT. A model magnetized only along the present field does not explain the belt completely; remanent magnetization of the iron formation is the likely reason.</li>
    <li><b>Susceptibility values are not rock values.</b> The compact models hold most magnetic cells at the 1 SI bound; with a bound of 0.3 SI the same data are fitted by a body {box('beta1_ub0.3')['volume_half_km3'] / box('beta1')['volume_half_km3']:.1f} times the size. What the data constrain is the product: {V['moment']} SI·km³ in the Sandur belt for β ≤ 2 and L1–L2.</li>
    <li><b>The magnetic and the dense rock are related but not the same.</b> The magnetic sheets lie along the margins of the dense belt and inside it; column by column the two integrated models correlate at only {G['corr_integrated']:.2f}.</li>
    <li><b>Against the published geology</b> (Section 6). All four iron-ore mines of the belt lie on or within {km(FAR['chi'])} km of a strongly magnetic column, which only {area_range('chi', MINE_NAMES)} of the area does. Of the strongly magnetic columns, {pct(RG['chi']['near'])} lie within {km(RG['near_km'])} km of a ridge held up by the iron formation, against {pct(RG['area_near'])} of the area. The ore bodies themselves are less than 200 m deep and far below the resolution of the models.</li>
    <li><b>Recommendation.</b> Continue the data upwards to about one cell above the ground; sparse with α<sub>s</sub> = 1 and depth weighting β between 0.5 and 1, with L1–L2 as an independent check; report the depth as a range and the susceptibility as a product with volume.</li>
  </ul>
</section>

<h2><span class="no">1</span>Data and preparation</h2>

<h3><span class="no">1.1</span>Upward continuation</h3>
<div class="prose">
<p>The survey (NGPM Block 8) was flown 80 m above the ground on lines 300 m apart; the grid has 37.5 m cells. In the area the anomaly ranges from <span class="num">{nt(raw['min'])}</span> to <span class="num">+{nt(raw['max'])}</span> nT, with the largest values over the banded iron formation of the Sandur belt (Figure {fig.ref('data')}).</p>
<p>A mesh of 1 km cells cannot produce the short wavelengths such data contain, and sampling the grid at 1 km would alias them: {pct(sw['80 m'])} of the grid's variance in the area is at wavelengths shorter than 2 km. The grid was therefore continued upwards by {P['continued_by_m']:.0f} m, to 1 km above the ground, before sampling (<code>geoinv3d/methods/continuation.py</code>): this multiplies every wavelength λ by exp(−2π · 920 m / λ), which leaves {100 * sw['1000 m']:.2f}% of the variance below 2 km ({100 * sw['500 m']:.1f}% at 500 m). The continued field is what a survey flown 1 km above the ground would have measured; its receivers are placed there, {N['receivers'][0]:.0f}–{N['receivers'][1]:.0f} m above sea level. At the nodes it ranges from <span class="num">{nt(up['min'])}</span> to <span class="num">+{nt(up['max'])}</span> nT. Section 4 tests this choice.</p>
</div>
{fig('data', "Left: ground elevation. Second: the magnetic anomaly as flown, 80 m above the ground. Third: continued upwards to 1 km above the ground. Right: the continued field at the 1 km nodes after removing a second-order trend surface, i.e. the data inverted.")}

<h3><span class="no">1.2</span>Field, regional and errors</h3>
<div class="prose">
<p>The inducing field is the IGRF 2020 field at the centre of the area: 42,100 nT, inclination 19.3°, declination −1.4° relative to grid north. At this low inclination a magnetized body gives a low over and north of it and a high to the south, which is the pattern of the Sandur belt in Figure {fig.ref('data')}. Only induced magnetization is modelled; the susceptibility is bounded to 0–1 SI.</p>
<p>A least-squares second-order trend surface (<span class="num">{nt(D['regional'][0])}</span> to <span class="num">+{nt(D['regional'][1])}</span> nT) is removed as the regional field, as for gravity. Each datum has an error of 5% of its value plus 10 nT ({D['std'][0]:.0f}–{D['std'][2]:.0f} nT, median {D['std'][1]:.0f} nT). The continued data are nearly noise-free; the error stands for what a model of 1 km cells with induced magnetization cannot reproduce. Section 4 tests a tighter error.</p>
<p>The mesh and the terrain are those of the gravity runs with terrain: cells above the DEM are left out, and the core reaches 10 km below the lowest ground. All depths are measured below the ground of each column.</p>
</div>

<h3><span class="no">1.3</span>The runs</h3>
{runs_table()}
<p class="note">Common settings: susceptibility 0–1 SI; β chosen automatically for χ² = N; at most 60 iterations and 40 IRLS iterations; α<sub>x</sub> = α<sub>y</sub> = α<sub>z</sub> = 1 as length scales. Li &amp; Oldenburg depth weighting weights the model norm by (z + z₀)<sup>−β/2</sup>, z the depth below the nearest receiver; β = 3 is the value usually quoted for magnetic data (the kernel decays as r⁻³), β = 2 for gravity.</p>

<h2><span class="no">2</span>Comparing the seven settings</h2>

<h3><span class="no">2.1</span>Data fit</h3>
<div class="prose">
<p>All seven runs reach the target, χ²/N {V['chi']}, with a residual RMS of {V['rms']} nT (table above). The residual maps (Figure {fig.ref('residuals')}) are alike, and they are not noise:</p>
<ul class="plain">
  <li><b>The high south of the Sandur belt is underfitted in every model</b>: the observed field exceeds the predicted one by more than 150 nT at {min(res['n_above_150'].values())}–{max(res['n_above_150'].values())} nodes (1 km² each), most of them there, and falls short of it by as much at only {min(res['n_below_150'].values())}–{max(res['n_below_150'].values())}. The error there is large (5% of up to {nt(up['max'])} nT), so these residuals are only a few standard deviations: for β = 1 the normalized residuals have a 99th percentile of {res['p99']:.1f}, {100 * res['above_3']:.1f}% of them exceed 3, and the worst 1% carry {pct(res['worst_1pct_share'])} of χ².</li>
  <li><b>The same sign along the north-eastern belt</b> and at the north-western end of the Sandur belt: positive residuals on the belts, in all seven models.</li>
  <li><b>What this means.</b> A body magnetized along the present field gives a fixed ratio between its low and its high. The observed high is stronger than any of the models can make it while fitting the low, which points to a magnetization direction other than the present field: remanent magnetization, common in banded iron formation. The models are therefore a first approximation in the belts.</li>
</ul>
</div>
{fig('residuals', "Residuals (observed − predicted) of the seven models, colour scale ±150 nT, and the data error (bottom right).")}

<h3><span class="no">2.2</span>Where the susceptibility goes</h3>
<div class="prose">
<p>The table gives where the susceptibility × volume lies in each model, and measures of the magnetic rock in the Sandur belt (a box of 20 × 18 km around it, easting 658–678 km, northing 1657–1675 km). Figure {fig.ref('mass_profiles')} shows the distribution with depth.</p>
</div>
{where_table()}
<p class="note">Core, lateral padding and below the core add up to 100%. D50, D90: half and 90% of the susceptibility × volume under the core area lie above this depth below the ground. Red: core &lt; 50%, lateral padding &gt; 30%, below the core &gt; 30%, D50 &gt; 7 km.</p>
{fig('mass_profiles', "Distribution of susceptibility × volume with depth below the ground, under the core area (percent per km). The dashed line is 10 km, the depth of the core.", narrow=True)}
<div class="prose">
<ul class="plain">
  <li><b>β = 2, β = 3 and sensitivity weighting fill the bottom of the mesh.</b> They leave {pct(t['beta2']['below'])}, {pct(t['beta3']['below'])} and {pct(t['sens']['below'])} of the susceptibility × volume below the core, in the large padding cells, where the data cannot tell what is there. These weightings make deep cells cheap, and the data, fitted within 5%, do not object.</li>
  <li><b>β = 0.5, β = 1 and L1–L2 keep it in the core:</b> {pct(t['beta0.5']['core'])}, {pct(t['beta1']['core'])} and {pct(t['l1l2_irls']['core'])}, with {pct(t['beta0.5']['below'])}, {pct(t['beta1']['below'])} and {pct(t['l1l2_irls']['below'])} below it. β = 1.5 is in between ({pct(t['beta1.5']['below'])} below the core).</li>
  <li><b>About a fifth to a quarter lies in the lateral padding</b> for the shallow settings. The belts leave the area at its north-western and south-eastern edges, and their anomalies are cut there.</li>
  <li><b>For gravity the same exponents behaved differently:</b> β = 1 gave a main body centred at 5.4 km, and only β = 1.5 began to trail to depth. For both methods the usable exponents are well below the textbook values of 2 and 3.</li>
</ul>
</div>

<h3><span class="no">2.3</span>The Sandur belt in section</h3>
{fig('sections', "E–W sections through the Sandur belt (northing 1667.5 km) in the seven models. The black line is the ground of the mesh, the dashed line the base of the core. The colour scale is a square-root scale from 0 to the bound of 1 SI.")}
<div class="prose">
<ul class="plain">
  <li><b>The sparse models are made of cells at the bound.</b> The largest value in the belt is {box('beta1')['max']:.2f} SI in all six; the models are sheets and blocks one to three cells wide.</li>
  <li><b>β sets the depth monotonically.</b> The centroid of the belt's magnetic rock lies at {box('beta0.5')['centroid_km']:.1f}, {box('beta1')['centroid_km']:.1f}, {box('beta1.5')['centroid_km']:.1f}, {box('beta2')['centroid_km']:.1f} and {box('beta3')['centroid_km']:.1f} km for β = 0.5, 1, 1.5, 2 and 3; 80% of it lies between {box('beta1')['d10_km']:.1f} and {box('beta1')['d90_km']:.1f} km for β = 1 and between {box('beta3')['d10_km']:.1f} and {box('beta3')['d90_km']:.1f} km for β = 3.</li>
  <li><b>A deeper model needs more magnetic rock:</b> {box('beta0.5')['total']:.0f}, {box('beta1')['total']:.0f}, {box('beta1.5')['total']:.0f}, {box('beta2')['total']:.0f} and {box('beta3')['total']:.0f} SI·km³ in the box.</li>
  <li><b>L1–L2 is smooth and does not reach the bound</b> (largest value {box('l1l2_irls')['max']:.2f} SI): a bowl from the surface to about 5 km with its centroid at {box('l1l2_irls')['centroid_km']:.1f} km, between β = 1 and β = 1.5, and {box('l1l2_irls')['total']:.0f} SI·km³.</li>
  <li><b>The shape.</b> In every model the magnetic rock of the belt forms a bowl that is open upwards, with sheets reaching the surface at the ridges on both sides. This is the shape of a synform with iron formation on its limbs, the traditional reading of the Sandur belt, which a structural study disputes {cite('MM93')} (Section 6.3); and the bowl also resembles the way a compact inversion closes a body at depth, so the base of the bowl is no more certain than its depth.</li>
</ul>
</div>

<h3><span class="no">2.4</span>Lateral structure</h3>
<div class="prose">
<p>Figure {fig.ref('slices')} shows five models at about 1, 3 and 6 km below the mean ground, Figure {fig.ref('robust')} the vertically integrated susceptibility and the centroid depth of each column.</p>
<ul class="plain">
  <li><b>The belts are in the same place in every model:</b> the two limbs of the Sandur belt, its closure in the south-east and the parallel north-eastern belt.</li>
  <li><b>Cell by cell the models differ more than the gravity models did.</b> The sheets are one or two cells wide and move by a cell from one setting to the next: the integrated maps of β = 0.5, β = 1 and L1–L2 correlate at {V['shallow_corr_raw']} as they are and at {V['shallow_corr']} after smoothing over 5 km (table below). The deep settings differ from these also at 5 km.</li>
  <li><b>Bodies at the corners of the area</b> (south-west, south-east, north) appear in every model. They fit anomalies that the edge of the area cuts and should not be interpreted.</li>
  <li><b>North–south sheets</b> north of the Sandur belt (near easting 652, 660 and 683 km) need caution. At an inclination of 19° a body elongated along magnetic north gives a weak anomaly except at its ends, so its length is poorly constrained, and a compact inversion tends to stretch bodies in that direction.</li>
</ul>
</div>
{corr_table()}
<p class="note">Correlation between the vertically integrated susceptibility maps in the 70 × 70 km core, smoothed over 5 km. Green: ≥ 0.85.</p>
{fig('slices', "Layers of five models at about 1, 3 and 6 km below the mean ground (core area only); square-root colour scale, 0 to 1 SI.")}
{fig('robust', "Top: vertically integrated susceptibility (SI·km); the dashed boxes are the Sandur belt and the north-eastern belt. Bottom: centroid depth of the susceptibility below the ground in each column (only columns whose integral exceeds 20% of the maximum).")}

<h2><span class="no">3</span>Cross-check: the synthetic test</h2>
<div class="prose">
<p>On 28 September the same settings were tested on synthetic data with known bodies under the same inducing field: a shallow block (1–3 km), a deep block (4–7 km) and a dyke dipping 45°, on a flat earth, with the data 80 m above it on a 1 km grid (<code>examples/synthetic_magnetic_karnataka.py</code>). Depths are centroids of the recovered susceptibility in each body's footprint.</p>
</div>
{SYNTHETIC}
<div class="prose">
<ul class="plain">
  <li><b>Sensitivity weighting, β = 2 and β = 3 put the bodies 1.5–3 km too deep</b> and recovered little of them; β = 3 was the deepest. On the field data the same three settings are the ones that fill the bottom of the mesh.</li>
  <li><b>L1–L2 was the best of the four</b> (correlation 0.60 with the true model, the deep block at 6.7 km for 5.5). On the field data it agrees with β = 1 at 5 km resolution (correlation {corr('beta1', 'l1l2_irls'):.2f}).</li>
  <li><b>Smaller exponents were tried only on a coarse mesh in that test:</b> there β = 1 put the deep block at its true depth. The field-data runs support the same choice, and a full-resolution synthetic test with β = 0.5 and 1 would confirm it.</li>
</ul>
</div>

<h2><span class="no">4</span>Tests of the set-up</h2>
<div class="prose">
<p>Six runs change one thing at a time from the run with β = 1 (Figure {fig.ref('tests')}):</p>
</div>
{tests_table()}
{fig('tests', "The reference run and the six tests. Left: E–W sections through the Sandur belt (square-root colour scale, 0 to 1 SI). Right: vertically integrated susceptibility.")}
<div class="prose">
<ul class="plain">
  <li><b>α<sub>s</sub> = 0.1</b> gives a smoother model that stays below the bound (largest value {box('as0.1_beta1')['max']:.2f} SI), with the same depth and the same susceptibility × volume.</li>
  <li><b>An upper bound of 0.3 SI</b> gives nearly the same product ({box('beta1_ub0.3')['total']:.0f} against {box('beta1')['total']:.0f} SI·km³) in a larger body ({box('beta1_ub0.3')['volume_half_km3']:.0f} against {box('beta1')['volume_half_km3']:.0f} km³ above half the largest value), {box('beta1_ub0.3')['centroid_km'] - box('beta1')['centroid_km']:.1f} km deeper. The data fix the product, the bound fixes the volume. The two iron-formation samples of the area have 0.05 SI, far less than either bound: the magnetic rock is not represented by the hand samples, or fills only a part of each cell.</li>
  <li><b>An error of 2% + 5 nT</b> can be fitted (χ²/N {t['beta1_err2']['chi2']:.2f}, RMS {t['beta1_err2']['rms']:.0f} nT), with more detail in the belt and more susceptibility below the core ({pct(t['beta1_err2']['below'])} against {pct(ref['below'])}). The residual high south of the belt remains, at a lower level.</li>
  <li><b>Continuing to 500 m instead of 1 km</b> gives the same model at 5 km resolution (correlation {N['integrated_tests_smooth']['beta1_h500']:.2f}) and a rougher fit (RMS {t['beta1_h500']['rms']:.0f} nT).</li>
  <li><b>Without continuation</b> the inversion does not reach the target in 55 iterations (χ²/N {r80['chi2']:.2f}, largest residual {nt(r80['max_abs'])} nT) and spreads weak susceptibility through the whole mesh: the aliased data are not the field of any model on this mesh.</li>
  <li><b>A flat earth</b> (receivers 1 km above a plane) changes little: the centroid of the belt is at {box('beta1_flat')['centroid_km']:.1f} km against {box('beta1')['centroid_km']:.1f} km, the correlation at 5 km is {N['integrated_tests_smooth']['beta1_flat']:.2f}. At 1 km above the ground the relief of a few hundred metres matters little for the magnetic data; it matters for the joint inversion, where both models must share the mesh of the gravity data.</li>
</ul>
</div>

<h2><span class="no">5</span>Magnetics and gravity</h2>
<div class="prose">
<p>Figure {fig.ref('gravity')} compares the magnetic model with the gravity model of the same area (both sparse with β = 1, with terrain).</p>
<ul class="plain">
  <li><b>The dense belt and the magnetic belt are the same structure.</b> The Sandur belt is the main body of both models; at 5 km resolution the strongest magnetic rock lies within the south-eastern half of the dense belt, and the north-eastern magnetic belt runs beside a smaller dense belt.</li>
  <li><b>Inside the belt they are different rock.</b> The magnetic sheets follow the margins of the dense body and its south-eastern closure; the dense rock fills the belt. Column by column the integrated susceptibility and the integrated density correlate at {G['corr_integrated']:.2f}; {pct(G['magnetic_in_dense'])} of the most magnetic fifth of the columns are dense (more than 0.3 g/cc·km), and {pct(G['dense_in_magnetic'])} of the dense columns are in that fifth.</li>
  <li><b>They are at different depths.</b> The magnetic rock of the belt is centred at {box('beta1')['centroid_km']:.1f} km, the dense rock at {G['box_sandur_density_centroid']:.1f} km (both with β = 1), and both depths follow the regularization.</li>
</ul>
<p>This is what a greenstone belt with iron formation should look like: thin, strongly magnetic iron formation on the limbs, and a thick pile of dense, weakly magnetic metavolcanic rock in the core (the amphibolite and metabasalt samples: 2.9–3.0 g/cc, 0.0002–0.004 SI). For a joint inversion it means that the two models share boundaries rather than values, which favours a structural coupling; the joint inversion of these two datasets is the subject of a separate report.</p>
</div>
{fig('gravity', "Left: vertically integrated density contrast of the gravity model with terrain (β = 1). Centre: vertically integrated susceptibility (β = 1). Right: the susceptibility, smoothed over 5 km, as contours on the density.")}
{geology(fig)}
<h2><span class="no">7</span>Recommendations</h2>
<div class="prose">
<ul class="plain">
  <li><b>Preparation:</b> continue gridded magnetic data upwards to about one cell size above the ground before thinning them, and place the receivers there. Thinning alone is not enough.</li>
  <li><b>Settings:</b> sparse, α<sub>s</sub> = 1, p = [0,2,2,2], depth weighting with β between 0.5 and 1; L1–L2 as an independent check. Not β = 3, not β = 2, and not sensitivity weighting on a mesh with deep padding.</li>
  <li><b>Report ranges:</b> the magnetic rock of the Sandur belt is centred {box('beta0.5')['centroid_km']:.1f}–{box('l1l2_irls')['centroid_km']:.1f} km below the ground in the three recommended models and holds {span([box(k)['total'] for k in SHALLOW], '{:.0f}')} SI·km³. Do not read the cell values as rock susceptibilities.</li>
  <li><b>Remanence:</b> the systematic residuals call for a magnetization-vector inversion, or for an amplitude quantity that does not depend on the direction, in the belts.</li>
  <li><b>Depth needs independent constraints</b>, as for gravity: susceptibility logs or samples of the fresh iron formation, its mapped dips, or the gravity data through a joint inversion.</li>
</ul>
</div>

<h2>Files</h2>
<div class="prose">
<ul class="plain">
  <li>Inputs (DEM, continued magnetic data at the nodes): <code>examples/output/karnataka_inputs/</code> (<code>prepare_inputs.py</code>)</li>
  <li>This report and its figures: <code>examples/output/karnataka_magnetic/scripts/</code> (<code>make_figures.py</code>, <code>build_report.py</code>)</li>
  <li>The 13 full-resolution results: <code>data/</code>; launched with <code>deploy/ec2_multi_run.py karnataka-magnetic</code></li>
  <li>All runs in one interactive workflow (DAG viewer): <code>karnataka_magnetic.geoinv3d_viewer.html</code></li>
  <li>The gravity comparison with terrain: <code>examples/output/karnataka_gravity_terrain/</code></li>
  <li>The published localities, the references and the schematic map of Section 6: <code>examples/output/karnataka_inputs/shared/literature.py</code></li>
</ul>
</div>
{references_html(REF_KEYS)}
<footer>GeoInv3D · SimPEG 0.25.2 · Generated by build_report.py; every number comes from the runs above, except the table of Section 3 (LOGBOOK.md, 28 September) and the published values of Section 6, which come from the sources cited there.</footer>
"""


def main():
    fig = Figures("en", FIGS)
    OUT.write_text(page("en", "Karnataka Magnetic Comparison", body(fig), extra_css=LIT_CSS),
                   encoding="utf-8")
    print(OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")
    if "--pdf" in sys.argv:
        print(to_pdf(OUT))


if __name__ == "__main__":
    main()
