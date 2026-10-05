"""Build the Lp ablation report: English HTML, optionally the PDF.

    py examples/output/synthetic_ablation_lp/scripts/make_figures.py
    py examples/output/synthetic_ablation_lp/scripts/build_report.py [--pdf]

Every number in the text comes from figures/numbers.json, data/sweep.log (time and cost) or
the inputs (inputs/: the reference model of group B, the boreholes of group C).
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "karnataka_inputs" / "shared"))
from style import Figures, page, table, to_pdf  # noqa: E402

FIGS = ROOT / "figures"
N = json.loads((FIGS / "numbers.json").read_text(encoding="utf-8"))
T, R, BEST, TM = N["truth"], N["runs"], N["best"], N["truth_measures"]
E, GS, P, BH, RB, NT = N["effects"], N["group_stats"], N["pinned"], N["boreholes"], N["robust"], N["near_truth"]
SPEC_B = json.loads((ROOT / "inputs" / "spec_B.json").read_text())
HOLES = list(csv.DictReader((ROOT / "inputs" / "boreholes_C.csv").read_text(encoding="utf-8").splitlines()))
OUT = ROOT / "ablation_report.html"

GROUPS = {"A": "A: no prior", "Bw1": "B: reference model, weight 1", "Bw10": "B: reference model, weight 10",
          "Bw100": "B: reference model, weight 100", "C": "C: boreholes, 150 m reach"}
BG = ["Bw1", "Bw10", "Bw100"]
NORMS = {"n0000": "(0,0,0,0)", "n0111": "(0,1,1,1)", "n0221": "(0,2,2,1)", "n0222": "(0,2,2,2)", "n1111": "(1,1,1,1)"}
BETAS = {"b10": "1", "b15": "1.5", "b20": "2", "b30": "3"}
LENGTHS = {"L1": "1", "L3": "3"}
SETTINGS = [f"{n}_{b}_{ell}" for n in NORMS for b in BETAS for ell in LENGTHS]
GUI = "n0221_b15_L1"            # the page's default: p = (0,2,2,1), α 1, β 1.5
ROBUST = "n1111_b20_L3"


def label(setting):
    n, b, ell = setting.split("_")
    return f"p = {NORMS[n]}, β = {BETAS[b]}, L = {LENGTHS[ell]}"


def short(setting):
    n, b, ell = setting.split("_")
    return f"p {NORMS[n]} · β {BETAS[b]} · L {LENGTHS[ell]}"


def sweep_cost():
    """Wall time and cost of the AWS run, from its log's last line."""
    text = (ROOT / "data" / "sweep.log").read_text()
    m = re.search(r"([\d.]+) h × (\d+) instances ≈ \$([\d.]+)", text)
    return (float(m.group(1)) * 60, int(m.group(2)), float(m.group(3))) if m else (None, None, None)


def f2(v):
    return f"{v:.2f}"


def score(g, s):
    return R[f"{g}_{s}"]["score"]


def gain(s):
    return score("C", s) - score("A", s)


def med_diff(g, key):
    return statistics.median(R[f"{g}_{s}"][key] - R[f"A_{s}"][key] for s in SETTINGS)


def median(g, key):
    return statistics.median(R[f"{g}_{s}"][key] for s in SETTINGS)


# ── tables ─────────────────────────────────────────────────────────────

def table_groups():
    geo = N["geology"]
    rows = [["A: no prior", "reference model 0 everywhere; bounds 0–3 SI", "—"]]
    for g in BG:
        rows.append([GROUPS[g], f"the interpreted bodies of Table 2 as the reference model, at weight {g[2:]}; bounds 0–3 SI",
                     f"{geo[g]['n_constrained']:,} (+{geo[g]['n_touched'] - geo[g]['n_constrained']:,} in part)"])
    rows.append([GROUPS["C"], "five logged holes (Table 3): ±0.005 SI and weight 5 on the trace, fading linearly to nothing at 150 m",
                 f"{geo['C']['n_constrained']:,} (+{geo['C']['n_touched'] - geo['C']['n_constrained']:,} in part)"])
    return table(["group", "prior", "cells constrained (over half / in part)"], rows, compact=True, numeric_from=9)


def table_prior():
    c, d = T["cube"], T["dyke"]
    s0, s1 = SPEC_B["sources"]
    u = SPEC_B["units"]
    p, q = s0["polygon"], s1["polygon"]
    pcx = (min(x for x, _ in p) + max(x for x, _ in p)) / 2
    pcy = (min(y for _, y in p) + max(y for _, y in p)) / 2
    qx = (min(x for x, _ in q) + max(x for x, _ in q)) / 2
    qy0, qy1 = min(y for _, y in q), max(y for _, y in q)
    return table(["", "true model", "interpreted (group B)", "error"], [
        ["cube centre (E, N)", f"{c['centre'][0]:,.0f}, {c['centre'][1]:,.0f}", f"{pcx:,.0f}, {pcy:,.0f}",
         f"{pcx - c['centre'][0]:+.0f} m E, {pcy - c['centre'][1]:+.0f} m N"],
        ["cube depth", f"{c['top_depth']:.0f}–{c['top_depth'] + c['edge']:.0f} m", f"{s0['top_m']:.0f}–{s0['bottom_m']:.0f} m",
         f"{s0['top_m'] - c['top_depth']:.0f} m deeper"],
        ["cube susceptibility", f"{c['kappa']} SI", f"{u['cube (prior)']['value']} SI", f"{(u['cube (prior)']['value'] / c['kappa'] - 1) * 100:+.0f} %"],
        ["intrusion's top (easting)", f"{d['top_x']:,.0f}", f"{qx:,.0f}", f"{qx - d['top_x']:+.0f} m"],
        ["intrusion's strike (northing)", f"{d['y'][0]:,.0f}–{d['y'][1]:,.0f}", f"{qy0:,.0f}–{qy1:,.0f}", f"{qy0 - d['y'][0]:+.0f} m"],
        ["intrusion depth", f"{d['top_depth']:.0f}–{d['bottom_depth']:.0f} m", f"{s1['top_m']:.0f}–{s1['bottom_m']:.0f} m",
         f"{s1['top_m'] - d['top_depth']:.0f} m deeper"],
        ["intrusion dip", f"{d['dip']:.0f}° E", f"{s1['dip']:.0f}° E", f"{s1['dip'] - d['dip']:.0f}° steeper"],
        ["intrusion susceptibility", f"{d['kappa']} SI", f"{u['intrusion (prior)']['value']} SI",
         f"{(u['intrusion (prior)']['value'] / d['kappa'] - 1) * 100:+.0f} %"],
    ], compact=True, numeric_from=9)


def table_holes():
    names = {"H1": "through the cube's centre", "H2": "50 m east of the cube", "H3": "inclined, through the intrusion",
             "H4": "the intrusion's shallow part", "H5": "the intrusion's deeper part"}
    rows, seen = [], {}
    for h in HOLES:
        seen.setdefault(h["hole"], []).append(h)
    for name, ivs in seen.items():
        h0 = ivs[0]
        inc = float(h0["inclination"])
        kind = "vertical" if inc == 90 else f"azimuth {float(h0['bearing']):.0f}°, {inc:.0f}° down"
        s, hits = 0.0, []
        for iv in ivs:
            L = float(iv["length_m"])
            if iv["unit"] != "hole: background":
                hits.append(f"{s:.0f}–{s + L:.0f} m: {iv['unit'].split(': ')[1]}")
            s += L
        key = name.split()[0]
        rows.append([f"{key}, {names[key]}", f"{float(h0['x']):,.0f}, {float(h0['y']):,.0f}", kind, f"{s:.0f} m",
                     "; ".join(hits) or "background (0 SI) throughout"])
    return table(["hole", "collar (E, N)", "direction", "length", "magnetic intervals (along the hole)"], rows,
                 compact=True, numeric_from=9)


def table_groupstats():
    rows = []
    for g in GROUPS:
        st = GS[g]
        rows.append([GROUPS[g], f2(R[BEST[g]]["score"]), f2(st["median"]), f"{RB[g]['score'][1]:.2f}–{RB[g]['score'][3]:.2f}",
                     f2(RB[g]["score"][0]), f"{st['better_than_A']} / 40" if g != "A" else "—", f"{st['converged']} / 40"])
    return table(["group", "best", "median", "middle half", "worst", "beats A at the same setting", "converged"],
                 rows, compact=True)


def table_effects():
    rows = []
    for dim, vals, name in (("norms", NORMS, "p"), ("beta", BETAS, "β"), ("length", LENGTHS, "L")):
        for v, txt in vals.items():
            rows.append([f"{name} = {txt}"] + [f2(E[g][dim][v]) for g in GROUPS])
    return table(["setting (averaged over the others)"] + ["A", "B, w 1", "B, w 10", "B, w 100", "C"], rows, compact=True)


def table_best():
    rows = [["true model, measured the same way", "—", "1.00", "1.00", "1.00", f"{TM['cube_depth']:.0f}", f"{TM['dyke_base']:.0f}",
             f"{TM['dip']:.0f}", "0.50", "1.00", "—"]]
    for g in GROUPS:
        x = R[BEST[g]]
        rows.append([f"{GROUPS[g]}<br><span class=note>{short(x['setting'])}</span>", f"{x['chi2_per_datum']:.2f}",
                     f2(x["score"]), f2(x["overlap_cube"]), f2(x["overlap_dyke"]), f"{x['cube_depth']:.0f}",
                     f"{x['dyke_base']:.0f}", f"{x['dip']:.0f}", f2(x["k_cube"]), f2(x["k_dyke"]), f"{x['k_wrong']:.2f}"])
    return table(["group, best setting", "χ²/N", "score", "overlap: cube", "overlap: intrusion", "cube depth",
                  "intrusion depth", "dip", "κ cube", "κ intrusion", "κ wrong"], rows)


def table_robust():
    """The spread of each measure over the 40 settings, A against C."""
    rows = []
    spec = [("score", "score", "{:.2f}"), ("overlap_cube", "overlap: cube", "{:.2f}"),
            ("overlap_dyke", "overlap: intrusion", "{:.2f}"), ("cube_depth", "cube depth (m)", "{:.0f}"),
            ("dyke_base", "intrusion depth (m)", "{:.0f}"), ("dip", "dip (°)", "{:.0f}"),
            ("k_dyke", "κ in the intrusion (SI)", "{:.2f}")]
    for key, name, fmt in spec:
        truth = {"score": "1.00", "overlap_cube": "1.00", "overlap_dyke": "1.00"}.get(key, fmt.format(TM[key]))
        cells = [name, truth]
        for g in ("A", "C"):
            q = RB[g][key]
            cells += [fmt.format(q[2]), f"{fmt.format(q[1])}–{fmt.format(q[3])}", f"{fmt.format(q[0])}–{fmt.format(q[4])}"]
        rows.append(cells)
    rows.append(["settings scoring ≥ 0.7", "", f"{NT['A']['score_07']} / 40", "", "", f"{NT['C']['score_07']} / 40", "", ""])
    rows.append(["dip within 10° of the truth", "", f"{NT['A']['dip_10']} / 40", "", "", f"{NT['C']['dip_10']} / 40", "", ""])
    rows.append(["cube depth within 50 m", "", f"{NT['A']['cube_depth_50']} / 40", "", "", f"{NT['C']['cube_depth_50']} / 40", "", ""])
    return table(["measure", "truth", "A: median", "A: middle half", "A: range", "C: median", "C: middle half", "C: range"], rows)


# ── the text ───────────────────────────────────────────────────────────

def body(fig):
    c, d, mesh = T["cube"], T["dyke"], N["mesh"]
    minutes, n_inst, cost = sweep_cost()
    chi = [x["chi2_per_datum"] for x in R.values()]
    n_conv = sum(x["status"] == "converged" for x in R.values())
    its = [x["iterations"] for x in R.values()]
    mins = [x["minutes"] for x in R.values() if x["minutes"]]
    a, cb = R[BEST["A"]], R[BEST["C"]]
    sa, sc = BEST["A"].split("_", 1)[1], BEST["C"].split("_", 1)[1]
    a_scores = [score("A", s) for s in SETTINGS]
    gains = {s: gain(s) for s in SETTINGS}
    g0000 = [gains[s] for s in SETTINGS if s.startswith("n0000")]
    g_l1 = [gains[s] for s in SETTINGS if s.split("_")[0] in ("n0221", "n0222") and s.endswith("L1")]
    g1111 = [gains[s] for s in SETTINGS if s.startswith("n1111") and "_b30_" not in s]
    worst = min(gains, key=gains.get)
    weak = [s for s in SETTINGS if score("A", s) < 0.5]
    weak_gain = statistics.median(gains[s] for s in weak)
    strong = [s for s in SETTINGS if score("A", s) >= 0.7]
    strong_gain = statistics.median(gains[s] for s in strong)
    n_low_cube = {g: sum(R[f"{g}_{s}"]["overlap_cube"] < 0.4 for s in SETTINGS) for g in ("A", "C")}
    v = BH["virtual"]
    dec = BH["c_minus_a"][sc]
    run_line = (f"{len(R)} inversions on {n_inst} AWS c5.4xlarge instances (ap-south-1), four at a time on each: "
                f"{minutes:.0f} min, about ${cost:.0f}" if minutes else f"{len(R)} inversions on AWS")
    return f"""
<header>
  <div class="eyebrow">GeoInv3D · synthetic magnetic study · 5 October 2026</div>
  <h1>What a Reference Model and Boreholes Add to a Tuned Lp Magnetic Inversion</h1>
  <p class="lede">An airborne magnetic survey over two buried bodies, inverted with Lp (sparse) regularization under three kinds of prior knowledge: none, an interpreted geological model with realistic errors, and five boreholes. The true model is known, so every inversion can be scored. Each of 40 regularization settings was run with each prior (200 inversions), which separates what the prior does from what the settings do.</p>
  <dl class="meta">
    <div><dt>Model</dt><dd>5 × 5 km (UTM 43N). A cube, {c['edge']:.0f} m, {c['top_depth']:.0f}–{c['top_depth'] + c['edge']:.0f} m below the ground, {c['kappa']} SI; an intrusion {d['thickness']:.0f} m thick and {(d['y'][1] - d['y'][0]) / 1000:.1f} km long, {d['top_depth']:.0f}–{d['bottom_depth']:.0f} m deep, dipping {d['dip']:.0f}° east, {d['kappa']} SI</dd></div>
    <div><dt>Data</dt><dd>Total-field anomaly {T['station_height']:.0f} m above the ground on a {T['station_spacing']:.0f} m grid ({T['n_stations']:,} points, {T['tmi_range'][0]:,.0f} to {T['tmi_range'][1]:,.0f} nT), noise 2 % + 5 nT; inverted at 100 m ({mesh['n_data']:,} points). Inducing field of the Karnataka survey area: inclination {T['field'][1]}°, declination {T['field'][2]}°</dd></div>
    <div><dt>Inversion</dt><dd>OcTree mesh, {mesh['core_cell_m']:.0f} × {mesh['core_cell_m']:.0f} × {mesh['core_cell_z_m']:.0f} m core cells, {mesh['n_active']:,} active cells; susceptibility 0–3 SI; Lp regularization with depth weighting; up to 200 iterations (100 IRLS)</dd></div>
    <div><dt>Runs</dt><dd>{run_line}; {min(mins):.0f}–{max(mins):.0f} min and {min(its)}–{max(its)} iterations each</dd></div>
  </dl>
</header>

<section class="summary" aria-labelledby="sum">
  <h2 id="sum">Key findings</h2>
  <ul>
    <li><b>The regularization settings matter as much as any prior.</b> With no prior, the 40 settings score from {min(a_scores):.2f} to {max(a_scores):.2f} (median {GS['A']['median']:.2f}) on the same data, all fitting it equally well. The smoothness length scale matters most (L = 3 averages {E['A']['length']['L3']:.2f}, L = 1 {E['A']['length']['L1']:.2f}); the norms (1,1,1,1) are the most robust ({E['A']['norms']['n1111']:.2f} on average); depth weighting β = 3 is worse. GeoInv3D's current default Lp setting scores {score('A', GUI):.2f}.</li>
    <li><b>Well tuned, the magnetic data alone recover both bodies.</b> The best setting without a prior ({label(sa)}) scores {a['score']:.2f}: the cube's magnetization is centred at {a['cube_depth']:.0f} m (true {TM['cube_depth']:.0f} m), the intrusion dips {a['dip']:.0f}° (true {TM['dip']:.0f}°) and holds {a['k_dyke']:.2f} SI on average (true 1 SI).</li>
    <li><b>Boreholes are the prior that helps, and they help most where the inversion is weakest.</b>
      <ul>
        <li>With five holes reaching 150 m, the best setting scores {cb['score']:.2f}, and the dip comes to {cb['dip']:.0f}°.</li>
        <li>The holes raise the score at {GS['C']['better_than_A']} of the 40 settings: by a median of {weak_gain:+.2f} where the inversion alone scores below 0.5, and {strong_gain:+.2f} where it already scores 0.7 or more.</li>
        <li>The worst quarter of settings with holes ({RB['C']['score'][1]:.2f}) scores as well as the median without them ({RB['A']['score'][2]:.2f}); {NT['C']['score_07']} settings reach 0.7 with holes, {NT['A']['score_07']} without.</li>
        <li>The holes' effect reaches about 250 m, and a test hole {v['nearest_hole_m']:.0f} m from any of them sees the intrusion's susceptibility recovered ({v['C']:.2f} SI against {v['A']:.2f} without; true 1).</li>
      </ul></li>
    <li><b>An interpreted model about 100 m out does not help, and hurts more the harder it is enforced.</b> As a reference model at weight 1, 10 and 100 it scores at best {R[BEST['Bw1']]['score']:.2f}, {R[BEST['Bw10']]['score']:.2f} and {R[BEST['Bw100']]['score']:.2f} (no prior: {a['score']:.2f}), and beats no prior at the same setting {GS['Bw1']['better_than_A']}, {GS['Bw10']['better_than_A']} and {GS['Bw100']['better_than_A']} times out of 40. It leaves magnetization where it was drawn wrongly ({median('Bw1', 'k_wrong'):.2f}–{median('Bw100', 'k_wrong'):.2f} SI against {median('A', 'k_wrong'):.2f}) at the same data fit, so the misfit gives no warning.</li>
    <li><b>Recommendations:</b> use a smoother length scale (L = 3) or the norms (1,1,1,1) by default; bring boreholes in with a radius of influence; give interpreted bodies a weight of 1 at most and always compare with the inversion without them.</li>
  </ul>
</section>

<h2><span class="no">1</span>The question</h2>
<div class="prose">
<p>An inversion of magnetic data needs two kinds of decisions: how to regularize it (the norms that make the model compact or smooth, the depth weighting, the smoothness) and what prior knowledge to put in. Both are uncertain in practice, and the data cannot settle either: many different models fit them equally well. This study asks, on a model whose answer is known:</p>
<ul class="plain">
  <li>how much the regularization settings alone change the result;</li>
  <li>whether an interpreted geological model, roughly right but not exact, improves the result when it is used as the reference model, and how strongly it should be weighted;</li>
  <li>whether a few boreholes improve it, and whether they do so across settings or only for a lucky one.</li>
</ul>
</div>

<h2><span class="no">2</span>The model and the data</h2>
<div class="prose">
<p>Two magnetic bodies lie under random hilly ground ({T['ground_range'][0]:.0f}–{T['ground_range'][1]:.0f} m above sea level) in a 5 × 5 km area: a cube, {c['edge']:.0f} m on a side, {c['top_depth']:.0f}–{c['top_depth'] + c['edge']:.0f} m below the ground at {c['kappa']} SI, and a stubby intrusion, {d['thickness']:.0f} m thick, {(d['y'][1] - d['y'][0]) / 1000:.1f} km long and {d['top_depth']:.0f}–{d['bottom_depth']:.0f} m deep, dipping {d['dip']:.0f}° east at {d['kappa']} SI (Figure {fig.ref('setup')}). The intrusion has about {T['true_volume_m3']['dyke'] / T['true_volume_m3']['cube']:.0f} times the cube's volume. The inducing field is that of the Karnataka survey area: at an inclination of {T['field'][1]}° the anomaly of each body is mainly a low over it with a high to the south. The data were computed on 25 m cells, a different mesh from the inversion's, with noise of 2 % plus 5 nT.</p>
</div>
{fig('setup', "The synthetic model. (a) The ground, the true bodies (red: the cube, and the intrusion's footprint as it dips), the interpreted bodies of group B (dashed) and the holes of group C (triangles, with 150 m circles along them); dotted: the section shown in later figures. (b) The total-field anomaly with noise. (c) The true model on the section, with group B's interpretation (dashed) and the holes (thick where they cross a body).")}

<h2><span class="no">3</span>The inversions</h2>
<div class="prose">
<p>Every inversion uses Lp (sparse) regularization, solved by iteratively reweighted least squares (IRLS) until the data are fitted to their noise (χ²/N = 1). Three of its settings were varied, 40 combinations in all:</p>
<ul class="plain">
  <li><b>norms</b> p = (p<sub>s</sub>, p<sub>x</sub>, p<sub>y</sub>, p<sub>z</sub>): (0,0,0,0), (0,1,1,1), (0,2,2,1), (0,2,2,2), (1,1,1,1). p<sub>s</sub> acts on the model, p<sub>x,y,z</sub> on its gradients: 2 favours smooth, 1 blocky and 0 compact models.</li>
  <li><b>depth weighting</b> β: 1, 1.5, 2, 3. The cells are weighted by (z + z<sub>0</sub>)<sup>−β/2</sup>; a larger β makes deep magnetization cheaper.</li>
  <li><b>length scale</b> L = α<sub>x</sub> = α<sub>y</sub> = α<sub>z</sub> (SimPEG's length scales): 1, 3. A larger L weighs the gradients more against the model itself: smoother models.</li>
</ul>
<p>The rest is fixed: α<sub>s</sub> = 1, the first trade-off parameter from the largest eigenvalue, halved each iteration, automatic IRLS thresholds, at most 200 iterations of which 100 IRLS. Each setting was run in five groups (Table 1).</p>
</div>
<p class="note">Table 1. The five groups</p>
{table_groups()}
<div class="prose">
<p><b>Group B</b> stands for a geologist's interpretation that is roughly right: each body about 100 m out of place, 50 m too deep, the intrusion 10° too steep, and the susceptibilities 50 % off (Table 2). It enters as the reference model only: the inversion is pulled towards it, more strongly the larger its weight (1 = as strongly as every other cell is pulled towards zero), but its bounds are those of every cell, 0–3 SI, so the data may change it.</p>
</div>
<p class="note">Table 2. Group B's interpretation against the true model</p>
{table_prior()}
<div class="prose">
<p><b>Group C</b> has five holes logging the true susceptibility (Table 3). The cells a hole passes through are held at the log (±0.005 SI, weight 5); the cells around it share in the log, the share falling linearly from 1 at the hole to 0 at 150 m (the model builder's <code>radius_m</code>): a cell 75 m away has a reference, bounds and weight half the log's and half the unconstrained ones. The intervals logged as background (0 SI) reach out in the same way.</p>
</div>
<p class="note">Table 3. Group C's holes (logs from the true model)</p>
{table_holes()}

<h2><span class="no">4</span>How the results are scored</h2>
<div class="prose">
<p>The main measure is the <b>volume-matched overlap</b>. In the cube's half of the area, take the cells of highest susceptibility until they add up to the cube's volume, and ask what share of the cube they cover; do the same for the intrusion in its half. The <b>score</b> is the mean of the two: 1 is a perfect recovery. It judges position and shape only, not amplitude, so it is fair both to compact models (a little volume at high susceptibility) and to smooth ones (a lot of volume at low).</p>
<p>Further measures, all compared with the true model measured the same way: the depth of the centre of the cube's magnetization (true {TM['cube_depth']:.0f} m), the depth above which 90 % of the intrusion's magnetization lies (true {TM['dyke_base']:.0f} m), the intrusion's apparent dip (true {TM['dip']:.0f}°), the mean susceptibility in each body, and, for group B, the mean susceptibility where its interpretation is wrong (cells of the interpreted bodies with no true body).</p>
<p class="note">Choosing the best setting by its score needs the true model, which real surveys lack: the best results here are what tuning could achieve at most. The median over the 40 settings stands for a setting chosen without that knowledge.</p>
</div>

<h2><span class="no">5</span>How much the settings matter</h2>
<div class="prose">
<p>All {len(R)} inversions fit the data equally well (χ²/N {min(chi):.2f}–{max(chi):.2f}; {n_conv} reached the target within the iteration limit), yet their scores differ widely (Figure {fig.ref('tuning')}, Table 4). Three patterns hold (Figure {fig.ref('effects')}, Table 5):</p>
<ul class="plain">
  <li><b>The length scale matters most, and L = 3 is better in every group.</b> At L = 1 the gradients weigh too little, and gradient norms of 2 squeeze the magnetization into thin sheets: without a prior, p = (0,2,2,2) scores {score('A', 'n0222_b15_L1'):.2f} at L = 1 and {score('A', 'n0222_b15_L3'):.2f} at L = 3 (β = 1.5).</li>
  <li><b>The norms (1,1,1,1) average highest in every group</b> ({E['A']['norms']['n1111']:.2f} without a prior) and depend least on the other settings. (0,0,0,0) varies more, but gives the best result with the holes; (0,2,2,1) and (0,2,2,2) average lowest in most groups.</li>
  <li><b>Depth weighting between β = 1 and 2 changes little</b>; β = 3 puts the bodies too deep and lowers the score without a prior ({E['A']['beta']['b30']:.2f} on average against {E['A']['beta']['b15']:.2f} at β = 1.5), with the reference model at weight 1 and with the holes. When the reference model is weighted 10 or 100 it dominates, and β matters little.</li>
</ul>
<p>The default Lp setting of GeoInv3D's page, p = (0,2,2,1) with α 1 (L = 1) and β = 1.5, scores {score('A', GUI):.2f} without a prior: one of the weaker settings here.</p>
</div>
{fig('tuning', "The score of every inversion: rows are the norms, columns the depth weighting β and the length scale L; * not converged within the iteration limit; red: the best of each group.")}
<p class="note">Table 4. The scores of each group over the 40 settings</p>
{table_groupstats()}
{fig('effects', "The mean score (solid) and the best (dotted) at each value of one setting, over the other settings.")}
<p class="note">Table 5. The mean score at each value of one setting</p>
{table_effects()}

<h2><span class="no">6</span>The best result of each group</h2>
<p class="note">Table 6. The best setting of each group. Depths in m: the centre of the cube's magnetization, and the depth above which 90 % of the intrusion's lies; dip of the intrusion in degrees; κ: mean susceptibility (SI) in each body, and where group B's interpretation is wrong. The first row is the true model, measured the same way</p>
{table_best()}
<div class="prose">
<p>Without a prior, the best setting places the cube's magnetization at {a['cube_depth']:.0f} m (true {TM['cube_depth']:.0f} m) and 90 % of the intrusion's above {a['dyke_base']:.0f} m (true {TM['dyke_base']:.0f} m), but makes the intrusion about {a['dip'] - TM['dip']:.0f}° too steep ({a['dip']:.0f}°). With the holes the dip comes to {cb['dip']:.0f}° and the susceptibility in the intrusion to {cb['k_dyke']:.2f} SI, the closest of all. Each of group B's best results keeps part of the magnetization where the interpretation put it (Figure {fig.ref('best_sections')}): the cube east of and below its true place, the intrusion west of it and steeper. The heavier the weight, the more stays there, and the less the true intrusion holds ({R[BEST['Bw1']]['k_dyke']:.2f}, {R[BEST['Bw10']]['k_dyke']:.2f} and {R[BEST['Bw100']]['k_dyke']:.2f} SI).</p>
</div>
{fig('best_sections', "East–west sections (northing 2.5 km) of the best setting of each group. Black outlines: the true bodies; dashed: group B's interpretation; thick black: the holes' magnetic intervals. Colour scale 0–1 SI.")}
{fig('best_metrics', "The measures of each group's best result; dashed: the true model measured the same way. Depth axes point down.")}

<h2><span class="no">7</span>Boreholes: better across the settings</h2>
<div class="prose">
<p>A best result can be luck: one setting out of 40 that happens to suit the model. What matters in practice is whether the holes help whatever setting is chosen. Run with and without the holes at each of the 40 settings (Figure {fig.ref('c_scores')}):</p>
<ul class="plain">
  <li><b>The holes raise the score at {GS['C']['better_than_A']} of the 40 settings.</b> The median rises from {GS['A']['median']:.2f} to {GS['C']['median']:.2f}, the worst from {RB['A']['score'][0]:.2f} to {RB['C']['score'][0]:.2f}, and the worst quarter of settings with holes ({RB['C']['score'][1]:.2f}) does as well as the median without them.</li>
  <li><b>They help most where the inversion alone is weakest.</b> Where it scores below 0.5 ({len(weak)} settings), the holes add a median {weak_gain:+.2f}; where it already scores 0.7 or more ({len(strong)} settings), {strong_gain:+.2f}. The largest gains are with the most compact norms, (0,0,0,0) ({min(g0000):+.2f} to {max(g0000):+.2f}), and with gradient norms of 2 at L = 1 ({min(g_l1):+.2f} to {max(g_l1):+.2f}), the settings that otherwise squeeze the bodies into thin sheets (Figure {fig.ref('c_sections')}). With the robust norms (1,1,1,1) the change is small ({min(g1111):+.2f} to {max(g1111):+.2f} at β ≤ 2).</li>
  <li><b>The losses are small.</b> The largest is {gains[worst]:+.2f} ({label(worst)}); only {40 - GS['C']['better_than_A']} settings lose at all.</li>
</ul>
</div>
{fig('c_scores', "(a) The scores of all 40 settings in each group (boxes: the middle half; whiskers: the range). (b) Each setting without (horizontal) and with the holes (vertical): above the dashed line the holes help. (c) The change in score from the holes at each setting.")}
<div class="prose">
<p>Measure by measure (Figure {fig.ref('c_measures')}, Table 7), the holes narrow the spread most for the cube, which hole H1 passes through: its overlap falls below 0.4 at {n_low_cube['C']} settings with the holes against {n_low_cube['A']} without, and the middle half of the settings ranges {RB['C']['overlap_cube'][1]:.2f}–{RB['C']['overlap_cube'][3]:.2f} against {RB['A']['overlap_cube'][1]:.2f}–{RB['A']['overlap_cube'][3]:.2f} without. The intrusion gains less (median overlap {RB['A']['overlap_dyke'][2]:.2f} → {RB['C']['overlap_dyke'][2]:.2f}). Its dip stays too steep in most settings (median {RB['A']['dip'][2]:.0f}° → {RB['C']['dip'][2]:.0f}°, true {TM['dip']:.0f}°): three holes fix its susceptibility where they cross it, but not the slope between them.</p>
</div>
<p class="note">Table 7. The spread of each measure over the 40 settings, without (A) and with the holes (C)</p>
{table_robust()}
{fig('c_measures', "The measures of all 40 settings without (A) and with the holes (C); dashed: the true model measured the same way. Depth axes point down.")}
{fig('c_sections', "The same seven settings without (left) and with the holes (right): the holes' best, a blocky setting, the best without a prior, β = 3, GeoInv3D's default, a setting that fails without a prior, and the setting where the holes lose most. Black outlines: the true bodies; thick black: the holes' magnetic intervals.")}
<div class="prose">
<p><b>How far the holes reach.</b> Comparing each setting with and without the holes (Figure {fig.ref('boreholes')}a): at the holes' best setting the model changes by about {max(dec[:3]):.2f} SI within 150 m of a hole, {dec[3]:.2f} SI at 150–250 m, {dec[4]:.3f} SI at 250–500 m and nothing beyond: the 150 m radius carries, through the regularization, to about 250 m. A test hole {v['nearest_hole_m']:.0f} m from the nearest hole, not given to the inversion, crosses the intrusion at {v['hit_m'][0]:.0f}–{v['hit_m'][1]:.0f} m. There the mean susceptibility is {v['C']:.2f} SI with the holes and {v['A']:.2f} SI without (true 1), and the intrusion's top and base are close to the truth only with the holes (Figure {fig.ref('boreholes')}d). The holes and their 150 m reach take in {100 * BH['cells_within']['150'] / BH['core_cells']:.1f} % of the cells of the area.</p>
<p><b>A caveat.</b> Where the holes lose ({short(worst)} and its neighbours), a likely cause is that the background intervals (0 SI) also reach 150 m: H3 and H5 run through background just above the intrusion, which may hold down its top. This was not checked setting by setting; a smaller radius for the background intervals would avoid it.</p>
</div>
{fig('boreholes', "(a) The mean change |C − A| with distance from the nearest hole (thin: every setting; thick: the holes' best setting); shaded: beyond the 150 m radius. (b)–(c) The susceptibility along two holes: the log (black), without (grey) and with the holes (orange), at the holes' best setting. (d) A test hole not given to the inversion.")}

<h2><span class="no">8</span>The interpreted model as a reference</h2>
<div class="prose">
<p>The interpretation is judged the same way, at its best and setting by setting (Figure {fig.ref('paired')}):</p>
<ul class="plain">
  <li><b>Weight 1</b>: best {R[BEST['Bw1']]['score']:.2f}, level with no prior ({a['score']:.2f}). It beats no prior at {GS['Bw1']['better_than_A']} of 40 settings, by a median of only {med_diff('Bw1', 'overlap_cube'):+.2f} (cube) and {med_diff('Bw1', 'overlap_dyke'):+.2f} (intrusion), mostly where no prior does badly. At the best setting without a prior it lowers the score from {a['score']:.2f} to {R['Bw1_' + sa]['score']:.2f} (Figure {fig.ref('same_setting_sections')}). The data do change it: the interpreted cube, referenced at 0.75 SI, ends at {P['Bw1']['0.75']['mean']:.2f} SI on average, the interpreted intrusion (0.5 SI) at {P['Bw1']['0.5']['mean']:.2f} SI.</li>
  <li><b>Weight 10</b>: best {R[BEST['Bw10']]['score']:.2f}, median {GS['Bw10']['median']:.2f}; better than no prior at {GS['Bw10']['better_than_A']} of 40 settings, the intrusion's overlap down by a median of {-med_diff('Bw10', 'overlap_dyke'):.2f}. The interpreted bodies stay closer to their reference ({P['Bw10']['0.75']['mean']:.2f} and {P['Bw10']['0.5']['mean']:.2f} SI).</li>
  <li><b>Weight 100</b>: best {R[BEST['Bw100']]['score']:.2f}, median {GS['Bw100']['median']:.2f}; better at {GS['Bw100']['better_than_A']} of 40 settings. The interpreted bodies are held exactly at their reference ({P['Bw100']['0.75']['mean']:.3f} and {P['Bw100']['0.5']['mean']:.3f} SI), as by a hard constraint, and the data are fitted elsewhere: at two settings with spurious bodies over 1 SI outside the cube (at the southern and northern edges of the area, 400–1,200 m deep), which leave the cube's overlap at 0.</li>
</ul>
<p>Every one of these inversions fits the data as well as those without a prior (χ²/N about 1). <b>An interpretation off by about 100 m and 10°, with susceptibilities 50 % off, did not improve the result.</b> Weakly weighted, the data undo most of it and it changes little; heavily weighted, it keeps magnetization where it was drawn, and nothing in the data fit shows it. A reference model is worth a heavy weight only if its geometry is better known than this; an accurate interpretation was not tested.</p>
</div>
{fig('paired', "Each setting with (vertical) and without a prior (horizontal): the cube (left) and the intrusion (right). Above the dashed line the prior helps.")}
{fig('same_setting_sections', "One setting (the best without a prior) in all five groups: only the prior differs.")}

<h2><span class="no">9</span>Conclusions and recommendations</h2>
<div class="prose">
<ol class="steps">
  <li><b>The data fit cannot choose the settings or test a prior.</b> All {len(R)} inversions have χ²/N {min(chi):.2f}–{max(chi):.2f}; their scores run from {min(x['score'] for x in R.values()):.2f} to {max(x['score'] for x in R.values()):.2f}.</li>
  <li><b>Tune the Lp settings; it matters as much as the prior.</b> For compact, stubby bodies like these, L = 3, the norms (1,1,1,1) and β 1.5–2 are the most robust: {label(ROBUST)} scores {score('A', ROBUST):.2f} without a prior and {score('C', ROBUST):.2f} with the holes. GeoInv3D's default (p = (0,2,2,1), L = 1) scores {score('A', GUI):.2f} here and should change, after a check on the field data.</li>
  <li><b>Bring boreholes in, with a radius of influence.</b> They were the only prior that helped at most settings, most of all at the settings where the inversion alone fails. Give the background intervals a smaller radius than the magnetic ones, and let the page's borehole import set <code>radius_m</code> (now only in a JSON spec).</li>
  <li><b>Weight interpreted bodies lightly.</b> Where their position, dip or susceptibility is uncertain, a weight of 1 at most, and always compare with the inversion without them: a structure that appears only with the prior is not a result.</li>
  <li><b>On field data, choose settings with the holes.</b> The best settings found here need the true model and do not carry over directly. The patterns may: a smoother L, the norms (1,1,1,1), β at most 2. Leaving one hole out at a time and checking how well the inversion predicts it is a test that needs no true model.</li>
</ol>
</div>

<h2><span class="no">10</span>Limitations</h2>
<div class="prose">
<ul class="plain">
  <li>The best settings were chosen against the true model, which real surveys do not have.</li>
  <li>One model and one noise realization; induced magnetization only, no remanence.</li>
  <li>One interpretation error and one hole radius (150 m, the same for magnetic and background intervals) were tested.</li>
  <li>α<sub>s</sub>, the cooling of the trade-off parameter and the IRLS thresholds were not varied; the data were thinned to 100 m.</li>
</ul>
</div>

<h2>Files</h2>
<div class="prose">
<ul class="plain">
  <li>Inputs and the parameters of the {len(R)} inversions: <code>examples/output/synthetic_ablation_lp/inputs/</code> (<code>scripts/make_synthetic.py</code>; <code>runs.json</code>)</li>
  <li>Runs: <code>deploy/ec2_sweep.py</code> (several EC2 instances) and <code>deploy/sweep_runner.py</code> (one machine); log <code>data/sweep.log</code></li>
  <li>Results: <code>data/runs/</code> (all {len(R)}, not in the repository); the best of each group in <code>data/best/</code>; every measure in <code>figures/numbers.json</code></li>
  <li>Measures <code>scripts/metrics.py</code>, figures <code>scripts/make_figures.py</code>, this report <code>scripts/build_report.py</code></li>
  <li>The boreholes' radius: <code>geoinv3d/methods/geology.py</code> (<code>radius_m</code>), described in <code>docs/geology_constraints.md</code></li>
</ul>
</div>
<footer>GeoInv3D · SimPEG 0.25.2 · Generated by build_report.py; every number comes from figures/numbers.json, data/sweep.log and inputs/.</footer>
"""


def main():
    fig = Figures("en", FIGS)
    for name in ("setup", "tuning", "effects", "best_sections", "best_metrics", "c_scores", "c_measures", "c_sections",
                 "boreholes", "paired", "same_setting_sections"):
        fig.ref(name)                      # numbered in the order they appear, before the text refers ahead
    OUT.write_text(page("en", "Lp Ablation Study", body(fig)), encoding="utf-8")
    print(OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")
    if "--pdf" in sys.argv:
        print(to_pdf(OUT))


if __name__ == "__main__":
    main()
