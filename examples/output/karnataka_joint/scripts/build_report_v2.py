"""The Karnataka joint gravity-magnetic report, second series of runs: English HTML (and PDF).

    py examples/output/karnataka_joint/scripts/make_figures.py --set v2
    py examples/output/karnataka_joint/scripts/build_report_v2.py [--pdf]

Every number in the text comes from figures_v2/numbers.json; the interpretive text is in
report_text_v2.py.  Runs: data/ec2_runs_fixed (full resolution), data/lowres_runs_fixed (2 km),
karnataka_magnetic/data/ec2_runs/beta1 and beta1_mvi (the magnetic data alone), and with the
rock-sample constraints data/lowres_bounds (2 km) and data/ec2_runs_bounds (full resolution).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "karnataka_inputs" / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import Figures, page, table, to_pdf  # noqa: E402
import literature as L  # noqa: E402

FIGS = ROOT / "figures_v2"
N = json.loads((FIGS / "numbers.json").read_text(encoding="utf-8"))
F, LOW, S, RUNS, M, MAG = N["full"], N["low"], N["single"], N["runs"], N["mesh"], N.get("magnetic", {})
GL = N.get("gl_info", {})
C = N.get("constraints", {})
LIT = N.get("literature", {})
DS = N.get("depth_sensitivity", [])
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


def behaviour_table():
    base = c("none")
    rows = []

    def row(k, asks, did, why):
        if has(k):
            rows.append([(f"<b>{LABEL[k]}</b>", "wrap"), (asks, "wrap"), (did, "wrap"), (why, "wrap")])
    k = "cross_gradient"
    if has(k):
        row(k, "The gradients of the two models are parallel wherever both change: ∇ρ × ∇χ = 0.",
            f"Cross-gradient measure {base['cross_gradient']:.2f} → {c(k)['cross_gradient']:.2f}, but only "
            f"{pct(c(k)['edges_shared'])} of the susceptibility edges on a density edge ({pct(base['edges_shared'])} "
            f"uncoupled). Density hardly changes ({g(k)['corr_with_none']:.2f}); susceptibility smoother "
            f"({m(k)['corr_with_none']:.2f}), centroid in the belt {m('none')['box_sandur']['centroid_km']:.1f} → "
            f"{m(k)['box_sandur']['centroid_km']:.1f} km.",
            "A cross product is also zero where one of the two gradients is zero: keeping the edges of the two "
            "models apart is cheaper than lining them up.")
    k = "joint_total_variation"
    if has(k):
        row(k, "The two models change in the same places: one total variation of both, √(|∇ρ|² + |∇χ|²).",
            f"Susceptibility edges on a density edge {pct(base['edges_shared'])} → {pct(c(k)['edges_shared'])}; "
            f"magnetic cells anomalous in density {pct(base['support']['magnetic_in_dense'])} → "
            f"{pct(c(k)['support']['magnetic_in_dense'])}. Density as uncoupled ({g(k)['corr_with_none']:.2f}); "
            f"fit {g(k)['chi2']:.2f} / {m(k)['chi2']:.2f}.",
            "√(a² + b²) &lt; a + b: two edges in the same cell cost less than two edges apart. Neither the values nor "
            "the direction of the gradients are tied.")
    k = "linear_correspondence"
    if has(k):
        row(k, "One linear relation between the values in every cell: density = 0.5 × susceptibility.",
            f"{pct(c(k)['support']['magnetic_in_dense'])} of the magnetic cells anomalous in density; the dense body "
            f"under the main high {rng(g('none')['main'])} → {rng(g(k)['main'])} km; {pct(g(k)['below'])} of the "
            f"density model below the core ({pct(g('none')['below'])} uncoupled). Fit {g(k)['chi2']:.2f} / "
            f"{m(k)['chi2']:.2f}.",
            "A cell can be dense only if it is magnetic: density goes where the magnetic data put the magnetic rock "
            "(near the surface) or where they cannot see it (below the core) (4.2).")
    k = GLK
    if has(k):
        row(k, "Few cells are anomalous, and a cell is anomalous in both models or in neither; no ratio between "
               "the values.",
            f"{pct(c(k)['support']['magnetic_in_dense'])} of the magnetic cells anomalous in density "
            f"({pct(c(CTRL)['support']['magnetic_in_dense']) + ' in its control, ' if has(CTRL) else ''}"
            f"{pct(base['support']['magnetic_in_dense'])} uncoupled); {pct(c(k)['edges_shared'])} of the edges "
            f"shared; the dense body {rng(g('none')['main'])} → {rng(g(k)['main'])} km; susceptibility spread "
            f"through the dense body. Fit {g(k)['chi2']:.2f} / {m(k)['chi2']:.2f}.",
            "The group norm ‖(ρ, χ)‖ &lt; |ρ| + |χ|: density is cheaper in a cell that already holds "
            "susceptibility; with no fixed ratio it can still stay elsewhere, so the shift is partial (4.2).")
    return table(["Coupling", "What it asks of the models", "What it did here", "Why"], rows, numeric_from=9)


def guide_table():
    rows = [
        ["One rock carries both anomalies (a body that is both dense and magnetic: a mafic–ultramafic "
         "intrusion, massive iron formation or magnetite ore alone)",
         "Group lasso", "PGI, if the rock's density and susceptibility are measured",
         "Whether the paired support is geologically one body; the depth of the dense body moves towards "
         "the magnetic one"],
        ["Different rocks share their boundaries (magnetic rock along the contacts or margins of dense rock, "
         "as the iron formation along the greenstone of this belt, or a magnetic unit along a fault that also "
         "bounds a dense one)",
         "Joint total variation", "Cross-gradient",
         "The share of common edges: the cross-gradient can meet its condition by keeping the edges apart "
         "(2% here). Avoid the linear correspondence and the group lasso: they move one model to the other"],
        ["A few rock types with known properties at the scale of a cell (logs, measured samples)",
         "PGI, with the measured units", "Joint total variation",
         "Whether each unit's cells make geological sense; units chosen by us are assumptions, not constraints"],
        ["One rock type with a known, fixed relation between density and susceptibility (fitted to samples "
         "of that rock)", "Linear correspondence", "Group lasso",
         "The depth and the share below the core of the density model: a wrong relation moves it, and the data "
         "fit does not warn"],
        ["The relation is not known",
         "No coupling first: invert separately and compare the two models (common edges, shared support)",
         "Then the coupling whose assumption the comparison supports",
         "Report what the couplings agree on; where they differ, the data do not decide"],
        ["The magnetic anomalies do not fit an induced magnetization (a systematic residual, highs and lows "
         "in the wrong ratio)",
         "A magnetization-vector inversion of the magnetic data", "—",
         "Couplings of the induced susceptibility are an approximation where the rock is remanent"],
    ]
    return table(["Geological setting", "First choice", "Also", "What to check"],
                 [[(x, "wrap") for x in r] for r in rows], numeric_from=9)


def depth_table():
    head = ["Depth below the ground (km)", "Gravity", "Magnetics", "Magnetics / gravity"]
    rows = [[f"{d['from_km']}–{d['to_km']}" if d["to_km"] < 40 else f"below {d['from_km']} (below the core)",
             f"{d['gravity']:.2f}", f"{d['magnetics']:.3f}", f"{d['magnetics'] / d['gravity']:.2f}"] for d in DS]
    return table(head, rows, compact=True)


ROCK_ORDER = ["Metabasalt / amphibolite", "Dolerite / gabbro", "Banded iron formation", "Granite", "Gneiss / TTG",
              "Schist, phyllite, quartzite", "Felsic volcanic"]
C_KEYS = ["rho50", "rho40", "rho35", "rho30", "rho35_gb05", "rho35_b05"]
C_LABEL = {"none": "No coupling", "joint_total_variation": "Joint total variation"}


def signed(x, nd=2):
    return f"{0.0:.{nd}f}" if round(x, nd) == 0 else f"{x:+.{nd}f}".replace("-", "−")


def c_setting(key):
    s = C["settings"][key]
    lo, hi = s["bounds"]
    return f"{signed(lo)} / {signed(hi)}", f"{s['betas'][0]:g} / {s['betas'][1]:g}"


def rocks_table():
    R, bg = C["rocks"], C["background"]
    head = ["Rock family", "Inside the block: median (range), n", "Region: median (range), n",
            f"Contrast to {bg} g/cc (block / region median)", "Susceptibility inside the block: median (5–95%), 10⁻³ SI"]
    rows = []
    for f in ROCK_ORDER:
        if f not in R:
            continue
        b, r = R[f].get("block"), R[f].get("region")

        def dens(x):
            if not x:
                return "—"
            if x["n"] == 1:
                return f"{x['rho_median']:.2f}, n = 1"
            return f"{x['rho_median']:.2f} ({x['rho_min']:.2f}–{x['rho_max']:.2f}), n = {x['n']}"
        con = " / ".join(signed(x["rho_median"] - bg) for x in (b, r) if x)
        def milli(x):
            return f"{max(x, 0.0) * 1e3 + 0.0:.2g}"
        sus = "—" if not b else (milli(b["k_median"]) if b["n"] == 1 else
                                 f"{milli(b['k_median'])} ({milli(b['k_p05'])}–{milli(b['k_p95'])})")
        rows.append([f, dens(b), dens(r), con, sus])
    return table(head, rows, compact=True)


def constraints_low_table():
    head = ["Coupling", "Density bounds (g/cc)", "β gravity / magnetics", "Gravity χ²/N", "Magnetic χ²/N",
            "Dense cells at the upper bound", "Main high, half-max km", "centroid km", "Column mass (g/cc·km)",
            "Dense rock of the belt within 1 km of the ground", "Magnetic rock within 1 km",
            "Susceptibility edges on a density edge", "Magnetic cells that are dense or light"]
    rows = []
    for cpl in ("none", "joint_total_variation"):
        for key in C_KEYS:
            x = C["low"].get(key, {}).get(cpl)
            if not x:
                continue
            b, beta = c_setting(key)
            mn = x["main"]
            rows.append([C_LABEL[cpl], (b, "n"), (beta, "n"),
                         warn(f"{x['gravity_chi2']:.2f}", not 0.7 <= x["gravity_chi2"] <= 1.3),
                         warn(f"{x['magnetics_chi2']:.2f}", not 0.7 <= x["magnetics_chi2"] <= 1.3),
                         pct(x["dense_at_upper"]), rng(mn), f"{mn['centroid_km']:.1f}", f"{mn['mass']:.2f}",
                         pct(x["box_dense_top1km"]), pct(x["box_magnetic_top1km"]),
                         pct(x["coupling"]["edges_shared"]), pct(x["coupling"]["support"]["magnetic_in_dense"])])
    return table(head, rows, compact=True)


def constraints_full_table():
    head = ["Coupling", "Density bounds (g/cc)", "β gravity / magnetics", "Gravity χ²/N", "RMS (mGal)",
            "Magnetic χ²/N", "RMS (nT)", "Main high, half-max km", "centroid km",
            "Dense rock of the belt within 1 km of the ground", "Susceptibility below the core",
            "Belt volume above half the largest susceptibility (km³)", "Susceptibility edges on a density edge",
            "Magnetic cells that are dense or light", "Time (min)"]
    rows = []
    for cpl in ("none", "joint_total_variation"):
        for key in ("rho50", C["full_key"]):
            x = C["full"].get(key, {}).get(cpl)
            if not x:
                continue
            b, beta = c_setting(key)
            mn, run = x["main"], x.get("run") or RUNS.get(cpl, {})
            rows.append([C_LABEL[cpl], (b, "n"), (beta, "n"), f"{x['gravity_chi2']:.2f}", f"{x['rms']['gravity']:.2f}",
                         f"{x['magnetics_chi2']:.2f}", f"{x['rms']['magnetics']:.0f}", rng(mn),
                         f"{mn['centroid_km']:.1f}", pct(x["box_dense_top1km"]),
                         warn(pct(x["magnetic_shares"]["below"]), x["magnetic_shares"]["below"] > 0.3),
                         f"{x['magnetic_box']['volume_half_km3']:.0f}", pct(x["coupling"]["edges_shared"]),
                         pct(x["coupling"]["support"]["magnetic_in_dense"]),
                         f"{run['minutes']:.0f}" if run else "—"])
    return table(head, rows, compact=True)


LIT_REFS = ["MM93", "MK12", "GSI", "IJERT", "MEAI", "MGR23", "IBMK", "ROM", "SK18", "SB14", "SIN20", "GEM", "IBMN",
            "NMET"]


def lit_table(fig):
    """Each result of the inversions against the published geology (Section 7)."""
    if not LIT:
        return ""
    k = C["full_key"]
    sh = LIT["shape"]
    u, c_ = sh["density_unconstrained"], sh["density_constrained"]
    far = max(r["distance_km"] for r in LIT["distances"]["susceptibility_single"] if r["kind"] == "iron")
    base = max(x["main"]["dense_bottom_km"] for x in C["full"][k].values())
    gb = LIT["gravity_base_km"]
    a, b = MAG["induced"], MAG["mvi"]
    z = b.get("magnetization") or {}
    m0, m1 = LIT["magnetic_centroid_km"]
    tag = {"Confirmed": "ok", "Supported": "ok", "Consistent, not independent": "mid", "Plausible, untested": "mid",
           "Disputed": "no", "Below resolution": "na"}
    rows = [
        ("Position and outline of the greenstone belt",
         f"Dense belt, NW–SE (strike {u['strike_deg']:.0f}°), about {u['length_km']:.0f} × {u['width_km']:.0f} km "
         f"({c_['length_km']:.0f} × {c_['width_km']:.0f} km with the constraints); the same in every model and coupling.",
         f"Sandur schist belt, about 60 km long and up to 18 km wide, NW–SE, enclosed by granite {L.cite('MM93', 'GSI')}.",
         "Confirmed"),
        ("Iron formation on the margins of the belt",
         "Thin, strongly magnetic sheets on both sides and at the south-eastern closure.",
         f"Iron-formation ridges of the Copper Mountain and Sandur ranges; all four iron mines lie on or within {far:.1f} km "
         f"of a strongly magnetic column (magnetic report, Section 6.2) {L.cite('MEAI', 'GSI')}.", "Confirmed"),
        ("Two rocks: dense greenstone, dense and magnetic iron formation",
         f"Low cell-by-cell correlation of density and susceptibility ({c('none')['corr_cells']:.2f}); structural coupling "
         "preferred (Section 4.3).",
         f"Metabasalt core with iron formation in the metasedimentary belts {L.cite('MM93', 'MK12')}; measured densities of "
         f"2.86 to 3.56 g/cc for the mafic rocks {L.cite('MGR23')}.", "Supported"),
        ("Iron formation at the top of the succession, on the margins in plan",
         "Dense metavolcanic rock in the centre; steep magnetic sheets reaching the surface at the ridges of both "
         "flanks, with a magnetic body in the centre as well (Section 7.2).",
         f"Western sequence from metabasalt at the base to iron formation at the top {L.cite('MM93')}; a Bababudan-type "
         f"assemblage {L.cite('GSI')}; steep, near-isoclinal folds bring the iron formation up on both flanks.",
         "Supported"),
        ("The belt crops out",
         "With sample bounds and β = 0.5 the dense body reaches the ground from easting 660 to 677 km (Section 5.3).",
         f"The belt forms hills of 900 to 1,050 m with mines at the surface {L.cite('MEAI')}.", "Confirmed"),
        ("Depth of the base",
         f"About {base:.1f} km under the main high with the constraints; {min(gb.values()):.1f} to {max(gb.values()):.0f} km "
         "without, depending on β (gravity report).",
         f"A basin about 6 km deep from a joint gravity–magnetic interpretation {L.cite('MGR23')}.",
         "Consistent, not independent"),
        ("Remanent magnetization in the south of the belt",
         f"MVI lowers the RMS from {a['rms']:.0f} to {b['rms']:.0f} nT; strong cells at I "
         f"{z.get('resultant_inclination', 0):.0f}°, D {z.get('resultant_declination', 0):.0f}° under and south of "
         f"Kumaraswamy (Figure {fig.ref('lit_mvi')}).".replace("-", "−"),
         f"Hematite and magnetite ore of the richest part of the belt {L.cite('IBMK', 'ROM')}; no palaeomagnetic data "
         "found.", "Plausible, untested"),
        ("A synform closing at depth",
         "The magnetic models form a bowl open upwards.",
         f"Two metasedimentary belts flank a central metavolcanic terrane and do not join; steep bedding, folds plunging "
         f"about 45° N {L.cite('MM93')}. A simple synform would put the youngest iron formation in the core.", "Disputed"),
        ("Ore bodies and gold",
         f"Cells 1 km wide and {LIT['cell_thickness_m']:.0f} m thick; the magnetic rock is centred {m0:.1f} to {m1:.1f} km "
         "deep.",
         f"Ore within about 70 to 170 m of the surface {L.cite('IBMK', 'ROM', 'SK18')}; gold in sulphidic chert and quartz "
         f"veins at 0.05 to 1.5 g/t {L.cite('SIN20', 'SB14')}.", "Below resolution"),
    ]
    return table(["Result of the inversions", "What the inversions show", "What the published work shows", "Verdict"],
                 [[(f"<b>{r}</b>", "wrap"), (shows, "wrap"), (pub, "wrap"),
                   (f'<span class="tag {tag[v]}">{v}</span>', "")] for r, shows, pub, v in rows], numeric_from=9)


def body(fig, notes):
    n = len(KEYS)
    cost = sum(r["cost_usd"] for r in RUNS.values())
    minutes = [r["minutes"] for r in RUNS.values()]
    mag_cost = ((MAG.get("mvi") or {}).get("run") or {}).get("cost_usd", 0.0)
    gl = GL.get(GLK) or {}
    c_runs = [x["run"] for x in C.get("full", {}).get(C.get("full_key"), {}).values() if x.get("run")]
    c_cost = sum(r["cost_usd"] for r in c_runs)
    return f"""
<header>
  <div class="eyebrow">GeoInv3D · field-data test · 30 September 2026 · second series · literature cross-check 1 October 2026</div>
  <h1>Karnataka Joint Gravity–Magnetic Inversion: Couplings, Remanence and Rock-Sample Constraints</h1>
  <p class="lede">The gravity and the magnetic data of the same area inverted together on one mesh with terrain, once for each coupling of the joint inversion, and the magnetic data alone with a magnetization vector per cell. The report shows what each coupling does to the two models, what it costs in data fit, which assumption suits this belt, what the magnetic data say about the direction of magnetization, and how the measured densities of the rocks, used as bounds, change the models.</p>
  <dl class="meta">
    <div><dt>Area</dt><dd>70 × 70 km, easting 641–711 km, northing 1634–1704 km (UTM 43N, EPSG:32643)</dd></div>
    <div><dt>Data</dt><dd>Complete Bouguer anomaly, {N['n_data']['gravity']:,} points, error 0.5 mGal; magnetic anomaly continued to 1 km above the ground, {N['n_data']['magnetics']:,} points, error 5% + 10 nT; second-order trends removed</dd></div>
    <div><dt>Models</dt><dd>Density contrast (−0.2 to +0.5 g/cc) and susceptibility (0 to 1 SI) on {M['n_active']:,} cells below the ground (1 km × 1 km × 250 m); each sparse with α<sub>s</sub> = 1 and depth weighting β = 1</dd></div>
    <div><dt>Runs</dt><dd>{n} joint inversions and one magnetization-vector inversion on AWS EC2 (ap-south-1, c5.9xlarge and c5.18xlarge), {min(minutes):.0f}–{max(minutes):.0f} minutes each, ${cost + mag_cost:.2f} in total; the same couplings on a 2 km mesh locally</dd></div>
    <div><dt>Constraints</dt><dd>The density bounds from the measured rock samples (−0.15 to +0.35 g/cc) with β = 0.5 for the gravity model: {len(c_runs)} runs at full resolution (${c_cost:.2f}) and {sum(len(v) for v in C.get('low', {}).values())} on the 2 km mesh (Section 5)</dd></div>
  </dl>
</header>

<section class="summary" aria-labelledby="sum">
  <h2 id="sum">Summary</h2>
  <ul>
@@LIT_SUMMARY@@
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

<h2><span class="no">4</span>Choosing a coupling</h2>
<div class="prose">
<p>There is no known model of this area to compare the results with, and every coupling fits the data equally (Section 2.1), so the data do not rank the couplings. What this study can say falls into three kinds, of decreasing certainty: what each coupling does to the two models, which is a property of the method and would be the same whatever the true rock (4.1, 4.2); what the magnetic data themselves show, the remanence (Section 3); and which coupling suits this belt, a judgement from the geology and the rock samples, not from the inversions (4.3). Section 4.4 turns them into a first choice for other geological settings.</p>
</div>

<h3><span class="no">4.1</span>What each coupling does</h3>
{behaviour_table()}
<p class="note">Measured in the core of the 1 km runs against the uncoupled run (Sections 2.2 and 2.3). The behaviour follows from what each coupling asks of the models; it describes the method, not whether its result is right.</p>

<h3><span class="no">4.2</span>Why the couplings that tie the values lift the dense body</h3>
<div class="prose">
@@WHY@@
</div>
{depth_table() if DS else ""}
<p class="note">Median column norm of the error-weighted sensitivities per km³ of cell, in the core columns of the 2 km mesh, relative to the cells 0–2 km below the ground.</p>

<h3><span class="no">4.3</span>Which coupling suits this belt</h3>
<div class="prose">
@@DISCUSSION@@
</div>

<h3><span class="no">4.4</span>Which coupling for which geology</h3>
<div class="prose">
<p>A coupling is an assumption about how the two properties are related. The first choice follows from what is known of the rocks before the inversion; the last column says what to check in the result.</p>
</div>
{guide_table()}
<p class="note">From the behaviour of Section 4.1, the belt of this report and the synthetic comparison of 30 September (three dense bodies, of high, low and no susceptibility), where the group lasso recovered the bodies best because each magnetic body was a dense body. PGI was not run in this series.</p>

<h2><span class="no">5</span>Constraints from the rock samples</h2>
<div class="prose">
@@C_INTRO@@
</div>

<h3><span class="no">5.1</span>The measured densities</h3>
{rocks_table()}
<p class="note">Rock samples of the area (toposheets 57A and 57B), grouped by rock family, from the density report of 30 September 2026 (<code>data/rock_properties/measured_by_rock_type.csv</code>). Block: the 70 × 70 km area of this report; region: the two toposheets. Densities in g/cc.</p>
{fig('constraints_rocks', "The measured densities as contrasts to the 2.66 g/cc background: range and median of each rock family inside the block (dark) and in the region (light), with the bounds of Sections 2–4 (grey, dashed) and those from the samples (red).")}
<div class="prose">
@@C_ROCKS@@
</div>

<h3><span class="no">5.2</span>Bounds and depth weighting on the 2 km mesh</h3>
{constraints_low_table()}
<p class="note">Depths below the ground. Main high: the density profile under the main Bouguer high; column mass: its positive density × thickness. Within 1 km: the share of the positive density × volume (or of the susceptibility × volume) in the Sandur belt that lies within 1 km of the ground. Red: χ²/N outside 0.7–1.3.</p>
{fig('constraints_column', "The density contrast under the main Bouguer high: on the 2 km mesh for each upper bound and depth weighting, uncoupled (left) and with the joint total variation (middle); at full resolution without and with the constraints (right).")}
<div class="prose">
@@C_LOW@@
</div>

<h3><span class="no">5.3</span>Full resolution</h3>
{constraints_full_table()}
<p class="note">As the table of Section 5.2; susceptibility below the core: its share of the susceptibility × volume (red above 30%). The rows with −0.20 / +0.50 are the runs of Sections 2–4.</p>
{fig('constraints_sections', "E–W sections through the Sandur belt (northing 1667.5 km) without and with the constraints: density contrast (left, ±0.5 g/cc) and susceptibility (right, square-root scale to 1 SI).")}
<div class="prose">
@@C_FULL@@
</div>

<h3><span class="no">5.4</span>What the constraints settle and what they do not</h3>
<div class="prose">
@@C_LIMITS@@
</div>

<h2><span class="no">6</span>The 2 km study</h2>
<div class="prose">
<p>Every coupling was also run locally on a 2 km × 2 km × 500 m mesh (1,296 + 1,296 data):</p>
</div>
{low_table()}
<p class="note">Red: χ²/N outside 0.7–1.3.</p>
<div class="prose">
@@LOWTEXT@@
</div>

<h2><span class="no">7</span>How well did the inversions work? An assessment against the published geology</h2>
<div class="added">Added 1 October 2026 · literature cross-check</div>
<div class="prose">
@@LIT_INTRO@@
</div>
{lit_table(fig)}
<p class="note"><span class="tag ok">Confirmed</span> matches published mapping or mining records. <span class="tag ok">Supported</span> consistent with them and with the rock samples. <span class="tag mid">Consistent, not independent</span> agrees, but the published value also rests on a model. <span class="tag mid">Plausible, untested</span> no published data bear on it. <span class="tag no">Disputed</span> the published structural work contradicts it. <span class="tag na">Below resolution</span> the question is finer than the mesh.</p>
{fig('lit_mvi', '@@LIT_MVI_CAPTION@@') if LIT else ''}

<h3><span class="no">7.1</span>The scales the inversions can and cannot see</h3>
{fig('lit_depths', '@@LIT_DEPTHS_CAPTION@@') if LIT else ''}
<div class="prose">
@@LIT_SCALES@@
</div>

<h3><span class="no">7.2</span>The schematic geology and the stratigraphy of the belt</h3>
{fig('lit_geology', '@@LIT_GEOLOGY_CAPTION@@') if LIT else ''}
<div class="prose">
@@LIT_GEOLOGY@@
</div>
@@LIT_VERDICT@@

<h2><span class="no">8</span>Conclusions and recommendations</h2>
<div class="prose">
<ul class="plain">
@@LIT_CONCLUSION@@
@@RECOMMEND@@
</ul>
</div>

<h2>Files</h2>
<div class="prose">
<ul class="plain">
  <li>Parameters of every run: <code>examples/output/karnataka_joint/scripts/joint_params.py</code> (joint) and <code>deploy/ec2_multi_run.py</code> (the magnetic runs <code>beta1</code> and <code>beta1_mvi</code>); launched with <code>deploy/ec2_multi_run.py</code> and <code>scripts/run_lowres.py --out data/lowres_runs_fixed</code></li>
  <li>This report and its figures: <code>scripts/make_figures.py --set v2</code> (with <code>scripts/constraint_figures.py</code>), <code>scripts/build_report_v2.py</code>, <code>scripts/report_text_v2.py</code>; results in <code>data/ec2_runs_fixed/</code>, <code>data/lowres_runs_fixed/</code> and <code>karnataka_magnetic/data/ec2_runs/</code></li>
  <li>The constraints (Section 5): the variants of <code>joint_params.VARIANTS</code>, run with <code>scripts/run_lowres.py --bounds KEY</code> (2 km, <code>data/lowres_bounds/</code>) and <code>deploy/ec2_multi_run.py karnataka-joint --only none_rho35_gb05,joint_total_variation_rho35_gb05</code> (<code>data/ec2_runs_bounds/</code>); compared by <code>scripts/compare_bounds.py [--full]</code>; the rock samples in <code>data/rock_properties/</code></li>
  <li>The assessment against the published geology (Section 7): <code>scripts/literature_figures.py</code>; the references, localities and the schematic map in <code>karnataka_inputs/shared/literature.py</code></li>
  <li>The single-method reports: <code>examples/output/karnataka_gravity_terrain/</code>, <code>examples/output/karnataka_magnetic/</code></li>
</ul>
</div>
{L.references_html(LIT_REFS) if LIT else ''}
<footer>GeoInv3D · SimPEG 0.25.2 · Generated by build_report_v2.py; every number comes from the runs above, except the published values of Section 7, which come from the sources cited there.</footer>
"""


def main():
    from report_text_v2 import (c_full, c_intro, c_limits, c_low, c_rocks, coupling_notes, discussion, fit_text,
                                lit_conclusion, lit_depths_caption, lit_geology, lit_geology_caption, lit_intro,
                                lit_mvi_caption, lit_scales, lit_summary, lit_verdict, low_text, mvi_text, recommend,
                                summary, why)
    fig = Figures("en", FIGS)
    ctx = dict(F=F, LOW=LOW, S=S, MAG=MAG, GL=GL, DS=DS, KEYS=KEYS, g=g, m=m, c=c, has=has, pct=pct,
               rng=rng, GLK=GLK, CTRL=CTRL, C=C, LIT=LIT, figref=fig.ref)
    html = body(fig, coupling_notes(**ctx))
    for tag, fn in (("@@SUMMARY@@", summary), ("@@FITTEXT@@", fit_text), ("@@MVITEXT@@", mvi_text),
                    ("@@DISCUSSION@@", discussion), ("@@WHY@@", why), ("@@LOWTEXT@@", low_text),
                    ("@@RECOMMEND@@", recommend), ("@@C_INTRO@@", c_intro), ("@@C_ROCKS@@", c_rocks),
                    ("@@C_LOW@@", c_low), ("@@C_FULL@@", c_full), ("@@C_LIMITS@@", c_limits),
                    ("@@LIT_SUMMARY@@", lit_summary), ("@@LIT_INTRO@@", lit_intro), ("@@LIT_MVI_CAPTION@@", lit_mvi_caption),
                    ("@@LIT_DEPTHS_CAPTION@@", lit_depths_caption), ("@@LIT_SCALES@@", lit_scales),
                    ("@@LIT_GEOLOGY_CAPTION@@", lit_geology_caption), ("@@LIT_GEOLOGY@@", lit_geology),
                    ("@@LIT_VERDICT@@", lit_verdict), ("@@LIT_CONCLUSION@@", lit_conclusion)):
        html = html.replace(tag, fn(**ctx))
    OUT.write_text(page("en", "Karnataka Joint Inversion", html, extra_css=L.LIT_CSS), encoding="utf-8")
    print(OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")
    if "--pdf" in sys.argv:
        print(to_pdf(OUT))


if __name__ == "__main__":
    main()
