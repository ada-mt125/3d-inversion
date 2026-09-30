"""The coupling comparison report (report/index.html) from comparison.json and the figures.

    py examples/output/coupling_comparison/make_report_html.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYS = ["none", "cross_gradient", "joint_total_variation", "linear_correspondence", "pgi", "group_lasso"]
NAME = {"none": "No coupling", "cross_gradient": "Cross-gradient", "joint_total_variation": "Joint total variation",
        "linear_correspondence": "Linear correspondence", "pgi": "PGI", "group_lasso": "Group lasso"}
WORKFLOW = {"blocks": "https://claude.ai/artifact/XzpkrMuoTQtn2VVzoY4vGb",
            "dipping": "https://claude.ai/artifact/Rv224oiaVHMcvRQfkCsTyQ"}
N = 357
TRUE = {"A": (0.3, 0.05), "B": (0.3, 0.01), "C": (0.3, 0.0)}


def table(shape):
    t = json.loads((HERE / shape / "comparison.json").read_text(encoding="utf-8"))
    best = {m: min(KEYS, key=lambda k: t[k][m]) for m in ("rho_error_rms", "chi_error_rms", "rho_outside_mean_abs")}
    rows = []
    for k in KEYS:
        e, b = t[k], t[k]["bodies"]
        cells = "".join(
            f'<td class="num rho">{b[n]["rho_mean"]:.3f}</td><td class="num chi">{b[n]["chi_mean"]:.4f}</td>'
            for n in "ABC")
        g, m = e["chi2"]["gravity"] / N, e["chi2"]["magnetics"] / N
        mark = lambda key: ' class="num best"' if best[key] == k else ' class="num"'   # noqa: E731
        rows.append(f'<tr><th scope="row">{NAME[k]}</th>{cells}'
                    f'<td{mark("rho_error_rms")}>{e["rho_error_rms"]:.4f}</td>'
                    f'<td{mark("chi_error_rms")}>{e["chi_error_rms"]:.4f}</td>'
                    f'<td{mark("rho_outside_mean_abs")}>{e["rho_outside_mean_abs"]:.4f}</td>'
                    f'<td class="num">{g:.2f} / {m:.2f}</td><td class="num">{e["seconds"]}</td></tr>')
    truth = "".join(f'<td class="num rho">{TRUE[n][0]:.3f}</td><td class="num chi">{TRUE[n][1]:.4f}</td>'
                    for n in "ABC")
    pgi = t["pgi"]
    return f"""
<div class="tablewrap"><table>
<thead>
<tr><th rowspan="2" scope="col">Coupling</th><th colspan="2" scope="colgroup">Body A</th><th colspan="2" scope="colgroup">Body B</th>
<th colspan="2" scope="colgroup">Body C</th><th rowspan="2" scope="col">Δρ rms error</th><th rowspan="2" scope="col">χ rms error</th>
<th rowspan="2" scope="col">|Δρ| outside bodies</th><th rowspan="2" scope="col">χ²/N grav / mag</th><th rowspan="2" scope="col">s</th></tr>
<tr><th class="rho">Δρ</th><th class="chi">χ</th><th class="rho">Δρ</th><th class="chi">χ</th><th class="rho">Δρ</th><th class="chi">χ</th></tr>
</thead>
<tbody>
<tr class="truth"><th scope="row">True model</th>{truth}<td class="num">0</td><td class="num">0</td><td class="num">0</td><td class="num">1</td><td></td></tr>
{''.join(rows)}
</tbody></table></div>
<p class="note">Body means of the inverted models (Δρ in g/cc, χ in SI) over the cells inside each body; errors are the rms
difference from the true model over the core of the mesh; the lowest of each is in bold. χ²/N near 1 fits the data to
their noise. PGI put {pgi['pgi_cells_right']['A']} of A's cells in unit A, {pgi['pgi_cells_right']['B']} of B's in B,
{pgi['pgi_cells_right']['C']} of C's in C, and called {pgi['pgi_false_body_cells']} cells outside the bodies a body.</p>"""


def figure(src, alt, caption):
    return (f'<figure><div class="plate"><img src="{src}" alt="{alt}" loading="lazy"></div>'
            f'<figcaption>{caption}</figcaption></figure>')


def shape_section(shape, heading, lede, extra):
    return f"""
<section id="{shape}">
<h2>{heading}</h2>
<p class="lede">{lede}</p>
<p class="links"><a href="{WORKFLOW[shape]}" target="_blank" rel="noopener">Open the interactive workflow</a>
<span>3D models against the true model, every run's data fit and settings</span></p>
{figure(f"{shape}_1_model.png", f"The {heading.lower()} synthetic model",
        "The synthetic model and its true sections at y = 600 m. All three bodies have the same density contrast (+0.3 g/cc); only their susceptibility differs.")}
{figure(f"{shape}_2_sections.png", f"Inverted sections of the six couplings, {heading.lower()}",
        "Each coupling's susceptibility (left) and density contrast (right) on the section through the bodies, on the true model's colour scales. Outlines mark the true bodies.")}
<h3>Recovered values and fit</h3>
{table(shape)}
{figure(f"{shape}_3_fit_profiles.png", f"Data along the centre line, {heading.lower()}",
        "Observed data (grey, ±σ) and each coupling's predicted data (blue) along y = 600 m.")}
{figure(f"{shape}_4_residuals.png", f"Normalized residual maps, {heading.lower()}",
        "Residuals divided by the data error. A fit that has explained the bodies leaves noise; red or blue patches over a body mean part of its anomaly is not explained.")}
{extra}
</section>"""


BLOCKS_NOTE = """<div class="findings">
<p><strong>Blocks.</strong> Without coupling, L1–L2 recovers each body at about a third of its density contrast
(0.08–0.11 of 0.3) and B's weak susceptibility barely at all (0.0011 of 0.01). The cross-gradient raises the densities to
0.10–0.15; the joint total variation leaves them about where they were. The group lasso recovers 80–90 % of every density
contrast, both susceptibilities (0.049 and 0.009) and none in C, with the least density outside the bodies. The linear
correspondence, told Δρ = 6χ everywhere, lowers C's density to 0.029 and gives it susceptibility it does not have; the
magnetic residual map shows the misfit over C. PGI, given the exact rock units, raises the densities to 0.13–0.22 but fits
the data to χ²/N 0.37 / 0.45 and puts a false dense zone under A.</p></div>"""
DIPPING_NOTE = """<div class="findings">
<p><strong>Dipping intrusions.</strong> Every method loses amplitude and some of the dip: the bodies reach 375 m and the data
resolve their tops best. The group lasso still gives the highest densities (0.14–0.16 of 0.3) and follows A's dip in
susceptibility; the cross-gradient raises the L1–L2 densities from 0.05–0.07 to 0.08–0.10. The linear correspondence
again drags C's density down. PGI classifies fewer cells correctly than for the blocks (A 18, B 19, C 33 of 84).</p></div>"""


def control_table():
    rows = []
    for shape, title in (("blocks", "Blocks"), ("dipping", "Dipping")):
        t = json.loads((HERE / shape / "comparison.json").read_text(encoding="utf-8"))
        for key, name in (("none", "1. L1–L2, IRLS, no coupling"), ("group_lasso_uncoupled", "2. L1 + L2, ADMM, no coupling"),
                          ("group_lasso", "3. Group lasso, ADMM")):
            e, b = t[key], t[key]["bodies"]
            cells = "".join(f'<td class="num rho">{b[n]["rho_mean"]:.3f}</td>' for n in "ABC") + \
                "".join(f'<td class="num chi">{b[n]["chi_mean"]:.4f}</td>' for n in "ABC")
            rows.append(f'<tr><th scope="row">{title}</th><td>{name}</td>{cells}'
                        f'<td class="num">{e["chi2"]["gravity"] / N:.2f} / {e["chi2"]["magnetics"] / N:.2f}</td></tr>')
    return f"""<div class="tablewrap"><table>
<thead><tr><th scope="col">Shape</th><th scope="col">Run</th><th class="rho" scope="col">Δρ A</th><th class="rho" scope="col">Δρ B</th>
<th class="rho" scope="col">Δρ C</th><th class="chi" scope="col">χ A</th><th class="chi" scope="col">χ B</th><th class="chi" scope="col">χ C</th>
<th scope="col">χ²/N grav / mag</th></tr>
<tr class="truth"><th scope="row">True</th><td></td><td class="num rho">0.300</td><td class="num rho">0.300</td><td class="num rho">0.300</td>
<td class="num chi">0.0500</td><td class="num chi">0.0100</td><td class="num chi">0.0000</td><td class="num">1</td></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>"""


def irls_table():
    rows = []
    for shape, title in (("blocks", "Blocks"), ("dipping", "Dipping")):
        old = json.loads((HERE / shape / "comparison_irls20.json").read_text(encoding="utf-8"))
        new = json.loads((HERE / shape / "comparison.json").read_text(encoding="utf-8"))
        for key in ("none", "cross_gradient", "joint_total_variation", "linear_correspondence"):
            log = (HERE / shape / f"log_{key}.txt").read_text(encoding="utf-8")
            end = "converged" if "Minimum decrease in regularization" in log else "40-cycle limit"
            cells = []
            for t in (old, new):
                b = t[key]["bodies"]
                cells.append(" / ".join(f"{b[n]['rho_mean']:.3f}" for n in "ABC"))
            rows.append(f'<tr><th scope="row">{title}</th><td>{NAME[key]}</td><td class="num">{cells[0]}</td>'
                        f'<td class="num">{cells[1]}</td><td>{end}</td></tr>')
    return f"""<div class="tablewrap"><table>
<thead><tr><th scope="col">Shape</th><th scope="col">Coupling</th><th scope="col">Δρ A / B / C, 20 IRLS cycles</th>
<th scope="col">Δρ A / B / C, 40 cycles</th><th scope="col">40 cycles ended by</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>"""


WHY = r"""
<section id="grouping">
<h2>The group lasso, taken apart</h2>
<p class="lede">The group lasso is an L1 + L2 regularization in which the L1 term acts on each cell's pair of values
(density, susceptibility) instead of on each value alone. The pairing is the coupling. Three runs show what comes from
where:</p>
<ol class="steps runs">
  <li><strong>L1–L2 solved by IRLS, no coupling.</strong> The "No coupling" run above, and the base of the cross-gradient,
  joint total variation and linear correspondence runs.</li>
  <li><strong>L1 + L2 solved by ADMM, no coupling.</strong> An ordinary L1 on every value, with the same weighting,
  λ<sub>2</sub> = 0.3, L-curve choice of λ<sub>1</sub> and solver as run 3. It is not a group lasso: nothing links the models.</li>
  <li><strong>Group lasso, solved by ADMM.</strong> Run 2 with the L1 applied to each cell's pair.</li>
</ol>
<div class="formulas">
  <p><span class="tag">Run 2: L1 + L2</span>
  <code>½‖b − Zζ‖² + λ<sub>1</sub> Σ<sub>k</sub> (|Δρ<sub>k</sub>| + |χ<sub>k</sub>|) + ½λ<sub>2</sub>‖ζ‖²</code></p>
  <p><span class="tag">Run 3: group lasso</span>
  <code>½‖b − Zζ‖² + λ<sub>1</sub> Σ<sub>k</sub> √(Δρ<sub>k</sub>² + χ<sub>k</sub>²) + ½λ<sub>2</sub>‖ζ‖²</code></p>
</div>
<p class="note">Runs 1 and 2 have the same kind of regularization and differ in how it is solved; runs 2 and 3 are solved
the same way and differ only in the pairing. Since the gravity data depend only on Δρ and the magnetic data only on χ,
run 2 is two independent inversions; only the square root in run 3 ties the models together.</p>

<h3>Runs 2 and 3: what the pairing does, in one step of the solver</h3>
<div class="findings why">
<p>Every iteration of the ADMM solver does two things:</p>
<ol class="steps">
  <li>From the data, it computes a trial value of every cell's density and susceptibility.</li>
  <li>It shrinks these values towards zero; a value below a threshold becomes exactly zero. This step is what makes the
  models compact.</li>
</ol>
<p>Runs 2 and 3 differ only in step 2:</p>
<ul class="steps">
  <li><strong>Run 2, L1 + L2:</strong> the density and the susceptibility of a cell are compared with the threshold
  separately. Either one is set to zero if it is small, whatever the other value of the same cell is.</li>
  <li><strong>Run 3, group lasso:</strong> the two values of a cell are compared with the threshold together, by their
  joint size √(Δρ² + χ²). The cell is set to zero only if both are small; otherwise both are kept and scaled down by the
  same factor.</li>
</ul>
</div>
{shrink}
<div class="findings why">
<p><strong>A cell of body B</strong> has a clear density and a weak susceptibility. In run 2 its weak susceptibility falls
below the threshold and becomes zero, so the cell ends up looking like a cell of C. B's magnetic anomaly still has to be
explained, and the few cells whose susceptibility does pass the threshold take it all: they end up at eight times the
true value. In run 3 the cell is kept because of its density, and its susceptibility is kept with it: B's susceptibility
spreads over its dense cells at about the right level.</p>
<p><strong>A cell of body C</strong> has density but no magnetic signal, so its trial susceptibility is about zero in both
runs and stays zero. The pairing does not create magnetization; it lets a weak one survive in cells that are active anyway.</p>
</div>

<h3>The three runs side by side</h3>
{control_table}
{why_blocks}
{why_dipping}
<div class="findings why">
<p><strong>Runs 1 and 2: the solver.</strong> With the same kind of regularization and no coupling, the ADMM solution
recovers 0.23–0.25 g/cc in the blocks against 0.08–0.11 for IRLS, and C, which has no magnetic signal to borrow, reaches
0.25 as well, so this gain does not come from coupling. ADMM finds the exact sparse solution (cells set exactly to zero),
the sensitivity weighting 1/‖k<sub>j</sub>‖ offsets the decay of the kernels with depth, and λ<sub>1</sub> comes from the
L-curve; IRLS approximates the L1 norm by iterative reweighting with a smoothing threshold and weights the cells
differently.</p>
<p><strong>Runs 2 and 3: the pairing, which is the coupling.</strong> The effect described above, measured: in run 2, B's
susceptibility survives in only a few cells, pushed to 0.083 SI (the truth is 0.01), and 60 % of B's dense cells have none.
In run 3 it fills 95 % of them with a peak of 0.027 (dipping bodies: 18 % → 84 % of the dense cells, peak 0.038 → 0.018).
The magnetic fit improves (χ²/N 1.22 against 1.36) and the densities gain up to 0.02. The pairing suits bodies whose
magnetic parts are also dense, as here; it does not force a dense cell to be magnetic (C keeps χ ≈ 0) or the two
properties to share a ratio or sign.</p>
<p><strong>So each coupling is best judged against its own uncoupled run:</strong> the group lasso against run 2; the
cross-gradient, the joint total variation and the linear correspondence against run 1.</p>
</div>

<h3>A note on the L1–L2 runs: reweighting cycles</h3>
<p class="note">With 20 IRLS cycles every L1–L2 run stopped at the limit before converging. With 40, the uncoupled run and
the joint total variation converge and their densities rise by 60–85 %; the cross-gradient gains most from the extra
cycles, and it and the linear correspondence still run to the limit, so more cycles may raise them further. All results
above use 40 cycles.</p>
{irls_table}
</section>
"""


WHY_HTML = WHY.format(
    control_table=control_table(), irls_table=irls_table(),
    shrink=figure("shrinkage.png", "How a cell's density and susceptibility are shrunk towards zero",
                  "The one step in which runs 2 and 3 differ, drawn for a threshold of 1 in the solver's scaled units "
                  "(an illustration, not data). Grey: the trial values that are set to zero."),
    why_blocks=figure("blocks_5_why.png", "Three runs, blocks",
                      "Blocks. Row 1: L1–L2 solved by IRLS, no coupling. Row 2: L1 + L2 solved by ADMM, no coupling. "
                      "Row 3: the group lasso (L1 on each cell's pair), solved by ADMM."),
    why_dipping=figure("dipping_5_why.png", "Three runs, dipping intrusions",
                       "Dipping intrusions: the same three runs."))

HTML = f"""<title>Three-Body Coupling Test</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>
:root {{
  --bg: #f6f7f9; --paper: #ffffff; --ink: #151a21; --ink-2: #4f5866; --ink-3: #7b8491;
  --rule: #d8dde4; --rho: #1c5cab; --chi: #b24a1c; --accent: #256abf; --best-bg: #e6eefa;
  --truth-bg: #eef1f5;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", "Helvetica Neue", Arial, sans-serif;
  --body: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --bg: #111418; --paper: #1a1e24; --ink: #e8ebf0; --ink-2: #aab2be; --ink-3: #7d8693;
    --rule: #2d333c; --rho: #7fb0ef; --chi: #f0996b; --accent: #86b6ef; --best-bg: #1d2c40; --truth-bg: #20252c;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --bg: #111418; --paper: #1a1e24; --ink: #e8ebf0; --ink-2: #aab2be; --ink-3: #7d8693;
  --rule: #2d333c; --rho: #7fb0ef; --chi: #f0996b; --accent: #86b6ef; --best-bg: #1d2c40; --truth-bg: #20252c;
}}
body {{ background: var(--bg); color: var(--ink); font: 400 15px/1.6 var(--body); }}
.wrap {{ max-width: 1180px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 72px; display: grid; gap: 56px; }}
header {{ display: grid; gap: 14px; max-width: 72ch; }}
.eyebrow {{ font: 500 12px/1.4 var(--mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-3); }}
h1 {{ font: 600 clamp(30px, 5vw, 44px)/1.08 var(--display); letter-spacing: -0.01em; margin: 0; text-wrap: balance; }}
h2 {{ font: 600 28px/1.15 var(--display); margin: 0; text-wrap: balance; }}
h3 {{ font: 600 19px/1.3 var(--display); margin: 8px 0 0; }}
p {{ margin: 0; }}
.lede {{ color: var(--ink-2); max-width: 72ch; }}
header .lede {{ font-size: 17px; }}
section {{ display: grid; gap: 22px; }}
.setup {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px 32px;
  border-block: 1px solid var(--rule); padding-block: 20px; }}
.setup div {{ display: grid; gap: 2px; }}
.setup dt {{ font: 500 12px/1.4 var(--mono); letter-spacing: 0.04em; text-transform: uppercase; color: var(--ink-3); }}
.setup dd {{ margin: 0; color: var(--ink); }}
.couplings {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px 28px; }}
.couplings div {{ display: grid; gap: 3px; padding-top: 12px; border-top: 1px solid var(--rule); }}
.couplings strong {{ font: 600 16px/1.3 var(--display); }}
.couplings .fam {{ font: 400 12px/1.4 var(--mono); color: var(--ink-3); }}
.couplings p {{ color: var(--ink-2); font-size: 14px; }}
figure {{ margin: 0; display: grid; gap: 8px; }}
.plate {{ background: #ffffff; border: 1px solid var(--rule); border-radius: 4px; padding: 10px; overflow-x: auto; }}
.plate img {{ display: block; width: 100%; height: auto; min-width: 640px; }}
figcaption {{ color: var(--ink-2); font-size: 13.5px; max-width: 90ch; }}
.links {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 14px; }}
.links a {{ font-weight: 600; color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 3px; }}
.links a:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.links span {{ color: var(--ink-3); font-size: 13.5px; }}
.tablewrap {{ overflow-x: auto; border: 1px solid var(--rule); border-radius: 4px; background: var(--paper); }}
table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
th, td {{ padding: 7px 10px; border-bottom: 1px solid var(--rule); text-align: left; white-space: nowrap; }}
thead th {{ font: 500 12px/1.3 var(--mono); color: var(--ink-3); letter-spacing: 0.02em; vertical-align: bottom; }}
tbody th {{ font-weight: 600; }}
td.num {{ font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; }}
.rho {{ color: var(--rho); }}
.chi {{ color: var(--chi); }}
td.best {{ background: var(--best-bg); font-weight: 600; }}
tr.truth {{ background: var(--truth-bg); }}
tbody tr:last-child th, tbody tr:last-child td {{ border-bottom: 0; }}
.note {{ color: var(--ink-3); font-size: 13px; max-width: 95ch; }}
.findings {{ background: var(--paper); border: 1px solid var(--rule); border-radius: 4px; padding: 16px 20px; max-width: 95ch; }}
.findings p {{ color: var(--ink-2); }}
.findings strong {{ color: var(--ink); }}
.summary {{ display: grid; gap: 12px; max-width: 80ch; }}
.findings.why {{ display: grid; gap: 10px; }}
.steps {{ margin: 0; padding-left: 22px; display: grid; gap: 6px; color: var(--ink-2); }}
.steps strong {{ color: var(--ink); }}
.runs {{ max-width: 90ch; }}
.formulas {{ display: grid; gap: 8px; background: var(--paper); border: 1px solid var(--rule); border-radius: 4px;
  padding: 14px 18px; overflow-x: auto; }}
.formulas p {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 14px; }}
.formulas code {{ font-size: 14px; color: var(--ink); white-space: nowrap; }}
.formulas .tag {{ font: 500 12px/1.4 var(--mono); letter-spacing: 0.04em; text-transform: uppercase; color: var(--ink-3);
  min-width: 190px; }}
.summary li {{ color: var(--ink-2); }}
.summary li strong {{ color: var(--ink); }}
.summary ul {{ margin: 0; padding-left: 20px; display: grid; gap: 8px; }}
footer {{ color: var(--ink-3); font-size: 13px; border-top: 1px solid var(--rule); padding-top: 16px; max-width: 95ch; }}
code {{ font: 400 12.5px var(--mono); }}
@media (max-width: 520px) {{ .wrap {{ padding-inline: 16px; gap: 44px; }} body {{ font-size: 14.5px; }} }}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">GeoInv3D · joint gravity–magnetic inversion · synthetic test</p>
  <h1>Three-Body Coupling Test</h1>
  <p class="lede">Three bodies with the same density contrast (+0.3 g/cc) and different susceptibility: A strongly magnetic
  (0.05 SI), B weakly (0.01 SI), C not at all. Can a joint inversion tell them apart, and which coupling of the density and
  susceptibility models helps? Six couplings are compared on the same data, first with block-shaped bodies, then with
  dipping intrusions.</p>
</header>

<section>
<h2>Set-up</h2>
<dl class="setup">
  <div><dt>Data</dt><dd>gz and TMI at 357 stations (75 m grid), vertical inducing field, 2 % noise of each peak</dd></div>
  <div><dt>Mesh</dt><dd>50 m tensor cells, core to 600 m depth, 600 m padding; data modelled on a finer 25 m mesh</dd></div>
  <div><dt>Regularization</dt><dd>L1–L2 elastic net per model (IRLS up to 40 cycles, L1 share 0.8, sensitivity weighting); PGI and the group lasso bring their own</dd></div>
  <div><dt>Coupling weight</dt><dd>1 for every weighted coupling: as strong as the regularization (unit-free)</dd></div>
  <div><dt>Blocks</dt><dd>200 × 300 × 200 m, 100–300 m deep, centred at x = 300, 750, 1200 m</dd></div>
  <div><dt>Dipping intrusions</dt><dd>120 m wide, dipping 55° east, from 75 to 375 m depth</dd></div>
</dl>
<div class="couplings">
  <div><strong>No coupling</strong><span class="fam">reference</span><p>Both models inverted together with one β, nothing linking them.</p></div>
  <div><strong>Cross-gradient</strong><span class="fam">structural · Gallardo &amp; Meju 2003</span><p>The models' gradients are parallel: boundaries coincide, values stay free.</p></div>
  <div><strong>Joint total variation</strong><span class="fam">structural · Haber &amp; Holtzman Gazit 2013</span><p>The models change in the same places and are flat elsewhere.</p></div>
  <div><strong>Linear correspondence</strong><span class="fam">petrophysical · SimPEG</span><p>One relation in every cell, here Δρ = 6χ (body A's ratio): a deliberately partial rule.</p></div>
  <div><strong>PGI</strong><span class="fam">petrophysical · Astic &amp; Oldenburg 2019</span><p>Each cell belongs to one rock unit; given the three true units, the best case for it.</p></div>
  <div><strong>Group lasso</strong><span class="fam">joint sparsity · Utsugi 2025</span><p>Anomalies share their cells, with any sign or ratio; λ1 from the L-curve, λ2 = 0.3.</p></div>
</div>
</section>

{shape_section("blocks", "Blocks", "Three boxes at the same depth. The anomalies overlap little, so this is the easier case.", BLOCKS_NOTE)}
{shape_section("dipping", "Dipping intrusions", "The same three bodies as tabular intrusions dipping 55° to the east, reaching deeper.", DIPPING_NOTE)}

{WHY_HTML}

<section class="summary">
<h2>What the couplings did</h2>
<ul>
  <li><strong>Group lasso</strong> recovered the most: densities close to the truth, the weak susceptibility of B in the
  right cells, none in C, and the least structure outside the bodies. Its L1 + L2 solved by ADMM gives the compact, strong
  bodies (an uncoupled L1 + L2 solved the same way does nearly as well on density); its pairing, the coupling, puts B's weak
  susceptibility in B's dense cells and improves the fit. It is the slowest here (3–4 min).</li>
  <li><strong>Cross-gradient</strong> improved clearly once the reweighting had enough cycles: densities 20–90 % higher than the
  uncoupled L1–L2 run, the largest gain among the couplings added to L1–L2.</li>
  <li><strong>Joint total variation</strong> changed little at weight 1: its densities stayed slightly below the uncoupled run's.</li>
  <li><strong>Linear correspondence</strong> is only as good as its relation: a single Δρ–χ line cannot describe three bodies
  with three different ratios, and C (dense, not magnetic) is the casualty.</li>
  <li><strong>PGI</strong> sharpened the bodies when told the exact units, but overfitted the data and misplaced dense cells below A.
  With real units known only roughly, expect less.</li>
</ul>
</section>

<footer>
<p>One synthetic, one noise realization, one weight per coupling; timings on a laptop. Scripts: <code>examples/output/coupling_comparison/</code>
(<code>run_coupling_comparison.py</code>, <code>make_report_figures.py</code>, <code>make_why_figure.py</code>,
<code>make_report_html.py</code>). Couplings: <code>docs/joint_couplings.md</code>.</p>
</footer>
</div>
"""

(HERE / "report" / "index.html").write_text(HTML, encoding="utf-8")
print(HERE / "report" / "index.html")
