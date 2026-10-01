"""The mineral prospectivity assessment of the Sandur area, for an industry reader: English HTML
(and PDF).

    py examples/output/karnataka_minerals/scripts/make_figures.py
    py examples/output/karnataka_minerals/scripts/build_report.py [--pdf]

A synthesis of the three Karnataka inversion reports (gravity, magnetics, the joint inversion)
and the published record; no new inversion.  Every number from the models comes from
figures/numbers.json (the screens of make_figures.py and the validation of validation.py) and
karnataka_joint/figures_v2/numbers.json (the joint report); the published values come from the
sources cited.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent
sys.path.insert(0, str(OUTPUT / "karnataka_inputs" / "shared"))
import literature as L  # noqa: E402
from style import Figures, page, table, to_pdf  # noqa: E402

FIGS = ROOT / "figures"
M = json.loads((FIGS / "numbers.json").read_text(encoding="utf-8"))
J = json.loads((OUTPUT / "karnataka_joint" / "figures_v2" / "numbers.json").read_text(encoding="utf-8"))
C, MAG = J["constraints"], J["magnetic"]
V = M["validation"]
OUT = ROOT / "karnataka_mineral_report_en.html"
REFS = ["MM93", "MK12", "GSI", "IJERT", "MEAI", "MGR23", "IBMK", "IBMN", "GEM", "ROM", "SK18", "SB14", "SIN20",
        "BHAT25", "NMET"]
cite, citet = L.cite, L.citet
RESERVES_MT = {"Kumaraswamy": 195.17, "Donimalai": 127.6, "Ramandurg": 47.7, "NEB Range": 2.6}
EXTRA_CSS = """
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 18px 0 26px; }
.cards div { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 12px 14px; }
.cards b { display: block; font: 600 22px/1.2 var(--serif); color: var(--accent); margin-bottom: 4px; }
.cards span { font-size: 13px; color: var(--muted); line-height: 1.45; }
.disclaimer { font-size: 12.5px; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 10px; margin-top: 8px; }
"""


def pct(x):
    return f"{100 * x:.0f}%"


def minus(x, fmt="{:.0f}"):
    return fmt.format(x).replace("-", "−")


def dist(table_key, name):
    return next(r for r in V["distances"][table_key] if r["name"] == name)


def mines_range(table_key):
    rows = [r for r in V["distances"][table_key] if r["kind"] == "iron"]
    return (max(r["distance_km"] for r in rows), min(r["area_as_close"] for r in rows),
            max(r["area_as_close"] for r in rows))


def captioned(n, title, html):
    return f'<p class="note"><b>Table {n}</b> · {title}</p>\n{html}'


def tag(level):
    cls = {"A": "ok", "High": "ok", "B": "mid", "Moderate": "mid", "C": "na", "Low": "na", "Not resolved": "no",
           "Unknown": "na"}[level]
    return f'<span class="tag {cls}">{level}</span>'


# ---------------------------------------------------------------- tables

def outlook_table():
    segs = M["segments"]
    rows = [
        ("Iron ore", f"About 1,876 Mt at about 63% Fe in six ranges {cite('GSI')}; four mines located here hold about "
         f"{round(sum(RESERVES_MT.values()), -1):.0f} Mt of reserves {cite('IBMK', 'GEM', 'IBMN')}",
         f"Maps the host iron formation: {M['bif_km2']} km² of horizons, {M['unmined_km2']} km² of them more than "
         f"{M['thresholds']['mine_km']:g} km from the located mines; a remanent zone of {M['remanent_km2']} km² south of "
         "Kumaraswamy", "High", "Lease and map check of F1–F" + str(len(segs)) + " and R1; finer magnetic inversion"),
        ("Manganese", f"Deogiri Formation, discontinuous over about 40 km of strike {cite('IJERT')}",
         "Only indirectly: the iron-formation horizons mark the belts in which the manganese horizon lies",
         "Moderate", "Mapping and geochemistry along the iron-formation belts"),
        ("Gold", f"Two low-grade occurrences: up to 0.49 g/t (Joga), 0.05–1.48 ppm (Taranagar) {cite('SB14', 'SIN20')}",
         f"Not resolved: dense, weakly magnetic ground beside the iron formation covers {pct(M['gold_in_belt_share'])} of "
         "the belt", "Unknown", "Induced polarization and EM, soil geochemistry, structural mapping"),
        ("Copper", f"Reconnaissance only: sporadic copper, up to 4,767 ppm in one sample, Mincheri block {cite('NMET')}",
         "No dense or magnetic body under the block; a thin shallow dense layer only", "Low",
         "Electrical and EM methods over the block and the north-eastern belt"),
    ]
    return table(["Commodity", "Known endowment", "What the geophysics adds", "Prospectivity", "Next step"],
                 [[(f"<b>{a}</b>", ""), (b, "wrap"), (c, "wrap"), (tag(d), ""), (e, "wrap")]
                  for a, b, c, d, e in rows], numeric_from=9)


def targets_table():
    segs = M["segments"]
    s1, s2 = segs[0], segs[1]
    w, e, so, no = M["remanent_extent"]
    rows = [
        ("R1", "Iron", f"{w:.0f}–{e:.0f}, {so:.0f}–{no:.0f}; at and south of Kumaraswamy", f"{M['remanent_km2']} km²",
         f"Magnetization not along the present field (RMS {MAG['induced']['rms']:.0f} → {MAG['mvi']['rms']:.0f} nT with a "
         f"vector model), {M['remanent_south_km']:.0f} km beyond the Kumaraswamy B block, under the richest part of the belt",
         "Moderate", "A", "Ground magnetics, oriented samples, lease check"),
        ("F1", "Iron", f"{s1['easting']:.0f}, {s1['northing']:.0f}; {s1['town']} {s1['town_km']:.0f} km",
         f"{s1['cells']} km²", f"The largest strongly magnetic body of the belt, dense ({pct(s1['dense_share'])} of its "
         "columns), in its centre under the main gravity high; folded iron formation or magnetite-bearing mafic rock",
         "Moderate", "A", "Lease check; ground magnetics and mapping to tell the two apart"),
        ("F2", "Iron", f"{s2['easting']:.0f}, {s2['northing']:.0f}; western range NW of NEB Range", f"{s2['cells']} km²",
         f"About {s2['length_km']:.0f} km of strongly magnetic horizon on a mapped iron-formation range "
         f"{cite('MEAI')}", "High", "B", "Lease check; may be held by leases not located here"),
    ]
    for i, s in enumerate(segs[2:], 3):
        side = "western" if s["easting"] < 666 else "eastern"
        where = ("north-western end, towards Hosapete" if s["easting"] < 656 else
                 f"beside the {side} range" if s["in_belt"] < 0.5 else f"{side} range")
        prio = "B" if s["cells"] >= 10 and s["in_belt"] >= 0.5 else "C"
        rows.append((f"F{i}", "Iron", f"{s['easting']:.0f}, {s['northing']:.0f}; {where}", f"{s['cells']} km²",
                     f"{s['length_km']:.0f} km of strongly magnetic horizon (mean {s['mean_susceptibility']:.1f} SI·km)",
                     "Moderate" if s["in_belt"] >= 0.5 else "Low", prio, "Lease check"))
    rows.append(("G1", "Gold", "Taranagar–Joga corridor, eastern margin of the belt", "—",
                 "Known occurrences in sulphidic chert and iron formation; not resolved by the potential-field models",
                 "Low", "C", "IP/EM, soil geochemistry, structural mapping"))
    rows.append(("C1", "Copper", "Mincheri block and the north-eastern belt", "—",
                 "Reconnaissance copper; no supporting dense or magnetic body", "Low", "C", "IP/EM over the block"))
    return table(["Area", "Commodity", "Location (E, N km)", "Size", "Evidence", "Confidence", "Priority",
                  "Recommended action"],
                 [[(f"<b>{a}</b>", ""), b, (c, "wrap"), (d, "n"), (e_, "wrap"), (tag(f), ""), (tag(g), ""), (h, "wrap")]
                  for a, b, c, d, e_, f, g, h in rows], numeric_from=9)


def deposits_table():
    def at(name):
        p = next((m for m in L.MINES if m[0] == name), None) or next(g for g in L.GOLD if g[0] == name)
        return f"{p[1]:.1f}, {p[2]:.1f}"
    w, e, s, n = L.MINCHERI
    rows = [
        ("Iron", "Kumaraswamy (NMDC)", "Supergene hematite ore in iron formation; B block 1.2 km × 650–700 m, C block "
         "3.5 km × 450 m", "Balance reserves 195.17 Mt at 62.01% Fe (1 April 2023)", "Mean drilled depth 70 m (B), 62 m (C)",
         at("Kumaraswamy"), cite("IBMK")),
        ("Iron", "Donimalai (NMDC)", "Amphibole–hematite–magnetite–chert iron formation, enriched near the surface",
         "Reserves 127.6 Mt; capacity 7 Mt/a, 5.7 Mt mined in 2024", "Ore grade in the upper 170 m", at("Donimalai"),
         cite("GEM", "ROM")),
        ("Iron", "Ramandurg (SKME)", "Iron formation", "Reserves 47.7 Mt, resource 86.5 Mt; 2.6 Mt mined in 2024", "—",
         at("Ramandurg"), cite("GEM")),
        ("Iron", "NEB Range (ML 2150)", "Iron formation", "Balance reserve 2.6 Mt; lump and fines above 55% Fe", "—",
         at("NEB Range"), cite("IBMN")),
        ("Iron", "The belt", "Six ranges: Donimalai, Kumaraswamy, Ramandurg, Kanavehalli, Devagiri, Thimmappanagudi",
         "About 1,876 Mt at about 63% Fe", "—", "—", cite("GSI", "MEAI")),
        ("Manganese", "Deogiri Formation", "Manganiferous phyllite with chert below the iron formation",
         "Discontinuous over about 40 km of strike", "—", "—", cite("GSI", "IJERT")),
        ("Gold", "Joga", "Quartz veins in the iron formation", "Up to 0.49 g/t", "—", at("Joga"), cite("SB14")),
        ("Gold", "Taranagar", "Banded sulphidic chert", "0.05 to 1.48 ppm", "—", at("Taranagar"), cite("SIN20")),
        ("Copper", "Mincheri block (G4 reconnaissance)", "Narrow ridges of banded hematite quartzite, ferruginous chert "
         "and low-grade magnetite quartzite", "Sporadic copper, up to 4,767 ppm in one sample", "—",
         f"about {w:.0f}–{e:.0f}, {s:.0f}–{n:.0f} (part)", cite("NMET")),
    ]
    return table(["Commodity", "Locality", "Host and form", "Size and grade", "Depth", "E, N (km)", "Source"],
                 [[a, (f"<b>{b}</b>", "wrap"), (c, "wrap"), (d, "wrap"), (e_, "wrap"), (f, "n"), (g, "wrap")]
                  for a, b, c, d, e_, f, g in rows], numeric_from=9)


def succession_table():
    rows = [
        ("Iron formation (Donimalai Formation, top)", "Banded iron formation: amphibole, hematite, magnetite, chert; "
         "Algoma type, conformable", "Iron ore by supergene enrichment; host of gold in quartz veins",
         cite("GSI", "ROM", "SB14")),
        ("Manganiferous phyllite with chert (Deogiri Formation)", "Below the iron formation", "Manganese ore",
         cite("GSI", "IJERT", "MM93")),
        ("Quartzite, carbonate, greywacke, argillite", "Metasedimentary rocks of the flanks", "—", cite("MM93")),
        ("Metabasalt (base; the central metavolcanic terrane)", "Dense, weakly magnetic; sulphidic chert interbedded",
         "Gold in sulphidic chert", cite("MM93", "SIN20")),
        ("Granite and gneiss (around the belt)", "The Closepet granite complex; younger granites on the eastern margin",
         "—", cite("MM93", "MGR23")),
    ]
    return table(["Unit (top to bottom)", "Rocks", "Ore", "Source"],
                 [[(f"<b>{a}</b>", "wrap"), (b, "wrap"), (c, "wrap"), (d, "wrap")] for a, b, c, d in rows],
                 numeric_from=9)


def validation_table():
    rows = []
    for name, *_ in L.MINES + [(g[0],) for g in L.GOLD]:
        s, d = dist("susceptibility_single", name), dist("density_single", name)
        kind = "" if any(m[0] == name for m in L.MINES) else " (gold)"
        rows.append([f"<b>{name}</b>{kind}", f"{s['distance_km']:.1f}", pct(s["area_as_close"]),
                     f"{d['distance_km']:.1f}", pct(d["area_as_close"])])
    return table(["Locality", "To a strongly magnetic column (km)", "Area as close", "To a dense column (km)",
                  "Area as close"], rows, compact=True)


def risks_table():
    rows = [
        ("Lease status unknown", "F1–F" + str(len(M["segments"])) + " may lie in existing leases (the belt has six "
         "iron ranges, four located here)", "Check against the IBM lease records before any spend"),
        ("Resolution", "1 km cells on data continued 1 km upwards: the ore (within 170 m of the surface) and single "
         "iron-formation layers (100–200 m) are not resolved", "Finer inversion of the 37.5 m survey grid"),
        ("Remanence untested", "R1 rests on a magnetization direction no palaeomagnetic data confirm",
         "Oriented samples at Kumaraswamy and over R1"),
        ("Depth", "The depth of the belt follows the inversion's assumptions", "Use the depths as ranges, not values"),
        ("Gold and copper", "Sulphide hosts are not seen by gravity and magnetics", "Electrical and EM methods"),
        ("Data", "Four mines and two gold occurrences located, to about 1 km; reserves as reported by the sources",
         "Full drilling and lease data from the operators or IBM"),
    ]
    return table(["Risk", "Effect", "Mitigation"], [[(f"<b>{a}</b>", "wrap"), (b, "wrap"), (c, "wrap")]
                                                    for a, b, c in rows], numeric_from=9)


def programme_table():
    rows = [
        ("1 · Desk (weeks)", "Check F1–F" + str(len(M["segments"])) + " and R1 against the 1:50,000 geology (GSI Bhukosh) "
         "and the lease records; invert the 37.5 m magnetic grid without upward continuation over R1, F1 and F2 "
         "(cloud compute, a few hours)", "Which areas are open and still anomalous at 100 m cells"),
        ("2 · Ground (months)", "Ground magnetics and geological mapping over R1, F1 and F2; oriented samples for "
         "remanence; rock-property samples", "Whether R1 and F1 are ore-bearing iron formation"),
        ("3 · Follow-up", "Induced polarization / resistivity and EM over the Taranagar–Joga corridor and the Mincheri "
         "block with soil geochemistry; drilling of the best iron area", "Drill targets; gold and copper go/no-go"),
    ]
    return table(["Phase", "Work", "Decision it informs"], [[(f"<b>{a}</b>", ""), (b, "wrap"), (c, "wrap")]
                                                           for a, b, c in rows], numeric_from=9)


# ---------------------------------------------------------------- body

def body(fig):
    k = C["full_key"]
    main = C["full"][k]["none"]["main"]
    sh = V["shape"]["density_unconstrained"]
    sm = V["schematic_match"]
    far_s, lo_s, hi_s = mines_range("susceptibility_single")
    far_d, lo_d, hi_d = mines_range("density_single")
    segs = M["segments"]
    mi = M["mincheri"]
    a, b = MAG["induced"], MAG["mvi"]
    z = b.get("magnetization") or {}
    inc, dec = z.get("resultant_inclination", 0), z.get("resultant_declination", 0)
    loc = M["localities"]
    return f"""
<header>
  <div class="eyebrow">GeoInv3D · mineral prospectivity · desk study · 1 October 2026</div>
  <h1>Sandur Schist Belt, Karnataka: Mineral Prospectivity Assessment</h1>
  <p class="lede">An exploration summary of the Sandur greenstone belt from regional gravity and aeromagnetic inversion and the published geology and mining record: the known endowment, where the geophysics points beyond the existing mines, how far it can be trusted, and a phased programme to test it.</p>
  <dl class="meta">
    <div><dt>Area</dt><dd>70 × 70 km around the Sandur schist belt, Ballari district, Karnataka (UTM 43N, easting 641–711 km, northing 1634–1704 km)</dd></div>
    <div><dt>Commodities</dt><dd>Iron ore (principal), manganese, gold, copper</dd></div>
    <div><dt>Basis</dt><dd>Three inversion studies (gravity, magnetics, joint; 1 km cells), measured rock samples, {len(REFS)} published sources; no new data, no field visit</dd></div>
    <div><dt>Status</dt><dd>Desk study for screening. Nothing here is a Mineral Resource, Ore Reserve or Exploration Target in the sense of the JORC Code or NI 43-101; reserves are as reported by their sources.</dd></div>
  </dl>
</header>

<section class="summary" aria-labelledby="sum">
  <h2 id="sum">Key messages</h2>
  <ul>
    <li><b>A proven iron district.</b> The belt holds about 1,876 Mt of iron ore at about 63% Fe {cite('GSI')}; the four mines located here (NMDC's Kumaraswamy and Donimalai, SKME's Ramandurg, NEB Range) report about {round(sum(RESERVES_MT.values()), -1):.0f} Mt of reserves, Kumaraswamy 195 Mt at 62% Fe {cite('IBMK', 'GEM', 'IBMN')}.</li>
    <li><b>The geophysics finds the ore's host rock.</b> Every iron mine lies on or within {far_s:.1f} km of a strongly magnetic column, a condition only {pct(lo_s)}–{pct(hi_s)} of the area meets: the models map the iron formation that hosts the ore.</li>
    <li><b>Two priority areas beyond the located mines.</b> R1, a remanently magnetized zone of {M['remanent_km2']} km² reaching {M['remanent_south_km']:.0f} km south of Kumaraswamy, the richest part of the belt; and F1, the largest unexplained magnetic body of the belt ({segs[0]['cells']} km²), in its dense centre. Further iron-formation segments (F2–F{len(segs)}) follow the mapped ranges.</li>
    <li><b>Manganese, gold and copper need other methods.</b> Their hosts are not dense or magnetic enough to show at this scale; induced polarization, electromagnetics and geochemistry are the tools.</li>
    <li><b>Next:</b> a lease and map check, then a finer inversion of the existing 37.5 m aeromagnetic grid over R1, F1 and F2, before any ground work.</li>
  </ul>
</section>

<div class="cards">
  <div><b>≈1,876 Mt</b><span>iron ore at ≈63% Fe in the belt {cite('GSI')}</span></div>
  <div><b>≈{round(sum(RESERVES_MT.values()), -1):.0f} Mt</b><span>reserves of the four located mines</span></div>
  <div><b>{M['bif_km2']} km²</b><span>iron-formation horizons mapped by the magnetics; {M['unmined_km2']} km² away from the located mines</span></div>
  <div><b>{M['remanent_km2']} km²</b><span>remanent zone R1 south of Kumaraswamy</span></div>
  <div><b>≈{main['dense_bottom_km']:.1f} km</b><span>depth of the belt under its centre (constrained model)</span></div>
</div>

<h2><span class="no">1</span>Commodity outlook</h2>
{captioned(1, "The commodities of the area and what the geophysics adds", outlook_table())}
<p class="note">Prospectivity: High — the geophysics maps the host and the record proves the commodity; Moderate — the host is known but not mapped by the geophysics; Low — no supporting geophysical signal; Unknown — the host is below what these data resolve.</p>

<h2><span class="no">2</span>Priority areas</h2>
<div class="prose">
<p>The areas below come from two screens of the models: iron-formation horizons (strongly magnetic columns on or beside the ridges of the belt) more than {M['thresholds']['mine_km']:g} km from the four located mines, and the zone where the magnetic data ask for a remanent magnetization. Priority A: a specific geophysical signal that the existing mines do not explain, beside the richest part of the belt; B: a long horizon on a mapped iron-formation range; C: small, off the belt, or a commodity the data do not resolve.</p>
</div>
{fig('prospectivity', "The priority areas: iron-formation horizons within " + f"{M['thresholds']['mine_km']:g}" + " km of a located mine (grey) and farther from them (red, F1 to F" + str(len(segs)) + "), and the remanent zone R1 south of Kumaraswamy (purple). Green: the schist belts of the schematic map (Section 3). " + L.LEGEND)}
{captioned(2, "Priority areas, ranked", targets_table())}
<div class="prose">
<ul class="plain">
  <li><b>R1 — the remanent zone.</b> South of Kumaraswamy the magnetic data cannot be fitted by rock magnetized along the present field; a magnetization pointing at I {inc:.0f}°, D {minus(dec)}° fits them. Hematite–magnetite ore with a remanent magnetization would explain it, and so would a magnetite-rich unit under cover. It lies beside the largest reserves of the belt and beyond the mapped ridges, which makes it the most specific iron signal of the models; it is untested.</li>
  <li><b>F1 — the central magnetic body.</b> The belt's centre is mapped as metavolcanic {cite('MM93')}, yet the models place a dense and strongly magnetic body there, under the main gravity high between Ramandurg and Kumaraswamy. It is either iron formation folded into the core or magnetite-bearing mafic to ultramafic rock; ground magnetics and mapping can tell which.</li>
  <li><b>F2 and the other segments</b> follow the two ranges of the belt. They are where iron formation is expected; their value depends on whether they are already held by the leases not located here.</li>
</ul>
</div>

<h2><span class="no">3</span>Geological setting</h2>
<div class="prose">
<p>The Sandur schist belt is an Archaean greenstone belt of the Dharwar craton, within the Closepet granite complex {cite('MGR23', 'MK12')}: a NW–SE belt about 60 km long and up to 18 km wide {cite('MM93', 'GSI')}, standing as a ring of hills 900–1,050 m high, the Copper Mountain range in the east and the Sandur range in the west {cite('MEAI')}.</p>
<p><b>The ore horizon.</b> The banded iron formation (Algoma type) is a chemical sediment at the top of the succession and the host of the ore. In the Sandur belt the succession comprises the Yeshwantnagar, Deogiri and Donimalai formations from the bottom up; the Deogiri Formation carries the manganese, the Donimalai Formation the iron formation {cite('GSI', 'IJERT')}. The western sequence runs from metabasalt at the base to manganiferous phyllite with chert and the iron formation at the top {cite('MM93')}; the assemblage is of the Bababudan type {cite('GSI')} (Table 3).</p>
<p><b>The structure.</b> Bedding is steep and the early folds near-isoclinal, and two metasedimentary belts flank a central metavolcanic terrane {cite('MM93')}. The iron formation therefore stands in near-vertical sheets on both flanks, which is where the ranges and the mines are.</p>
</div>
{captioned(3, "The succession and its ores", succession_table())}
{fig('geology', "Schematic geology with the published localities, drawn from the 450 m DEM: ridges where the ground stands more than 80 m above the median of a 12 km window, read as the iron-formation ridges because the iron formation holds up the hills of the belt " + cite('MEAI') + f"; the schist belts are the envelope of the ridges (the main belt {M['schematic']['length_km']:.0f} × {M['schematic']['width_km']:.0f} km, strike {M['schematic']['strike_deg']:.0f}°); the central metavolcanic terrane after " + citet('MM93') + ". A sketch for orientation, not a geological map: the mapped belt is about 60 km long, continuing north-west towards Hosapete. " + L.LEGEND)}

<h2><span class="no">4</span>Known deposits</h2>
{captioned(4, "Deposits and occurrences of the area", deposits_table())}
<p class="note">{L.POSITIONS_NOTE} Reserves and production as reported by the sources, not re-estimated.</p>
<div class="prose">
<p>The iron ore is supergene enrichment of the iron formation near the surface: two E–W bands with a mean drilled depth of 62–70 m at Kumaraswamy {cite('IBMK')}, ore grade within the upper 170 m at Donimalai {cite('ROM')}, and hematite–magnetite ore altered from the iron formation at Obulapuram {cite('BHAT25')}. The gold occurrences are in the chemical sediments, quartz veins in the iron formation and sulphidic chert {cite('SB14', 'SIN20')}; the copper of the Mincheri block is sporadic, in narrow iron-rich quartzite and chert ridges {cite('NMET')}.</p>
</div>

<h2><span class="no">5</span>How far the geophysics can be trusted</h2>
<div class="prose">
<p><b>It finds the known deposits.</b> The test of the models is where the mines fall. Every iron mine lies on or within {far_s:.1f} km of a strongly magnetic column (≥ {V['thresholds']['magnetic']:g} SI·km integrated over depth), which only {pct(lo_s)}–{pct(hi_s)} of the area does, and on or within {far_d:.1f} km of a dense column (≥ {V['thresholds']['dense']:g} g/cc·km; {pct(lo_d)}–{pct(hi_d)} of the area). A random point would rarely do as well (Table 5).</p>
<p><b>It reproduces the belt.</b> The dense belt of the models runs NW–SE, about {sh['length_km']:.0f} × {sh['width_km']:.0f} km; {pct(sm['density_constrained']['in_outline'])} of its strongly dense columns lie inside the schematic belt outline, which covers {pct(sm['density_constrained']['area_in_outline'])} of the area, and the magnetic rock reaches the surface in steep sheets at the ridges of both flanks (Figures {fig.ref('models')} and {fig.ref('section')}), as the stratigraphy and folding predict. With the measured rock densities as bounds the belt reaches about {main['dense_bottom_km']:.1f} km under its centre, close to a published estimate of about 6 km {cite('MGR23')}.</p>
<p><b>It does not see the ore itself.</b> The models have 1 km cells on data continued 1 km upwards. The ore lies within 170 m of the surface, and single iron-formation layers are 100–200 m thick; a model sheet is at least one cell (1 km) wide, and the magnetic data constrain only its susceptibility × width. At Donimalai and Ramandurg the mine's own column is weakly magnetic ({loc['Donimalai']['susceptibility']:.2f} and {loc['Ramandurg']['susceptibility']:.2f} SI·km) beside a strongly magnetic one, consistent with hematite-rich enriched ore beside magnetite-bearing fresh iron formation {cite('ROM', 'BHAT25')}, but not proven at this scale.</p>
</div>
{captioned(5, "The known localities against the models", validation_table())}
<p class="note">Distance from each locality to the nearest 1 km column above the threshold, and the share of the area (5 km edge band left out) at least as close to such a column. Susceptibility: the magnetic study, β = 1; density: the gravity study with terrain, β = 1.</p>
{fig('models', "The three models, integrated over depth on the 1 km columns: (a) density contrast (joint inversion with the rock-sample bounds), (b) susceptibility along the present field, (c) amplitude of the magnetization vector. Black: the schematic belt outline (solid) and ridges (dotted). " + L.LEGEND)}
{fig('section', "E–W section through the belt at northing 1664.5 km (through the main gravity high and Donimalai): density contrast and susceptibility of the joint inversion with the rock-sample bounds. A dense metavolcanic core with steep magnetic sheets on both flanks; the black line is the ground.")}

<h2><span class="no">6</span>Manganese, gold and copper</h2>
<div class="prose">
<ul class="plain">
  <li><b>Manganese</b> lies in the Deogiri Formation below the iron formation, over about 40 km of strike {cite('GSI', 'IJERT')}. Manganese oxide in phyllite is neither dense nor magnetic enough at 1 km cells; the iron-formation horizons of Section 2 mark the belts in which it lies.</li>
  <li><b>Gold</b> is in sulphidic chert and iron formation {cite('SB14', 'SIN20')}, magnetite-poor rock that is dense but weakly magnetic. Such ground beside the iron formation covers {pct(M['gold_in_belt_share'])} of the belt and includes Taranagar but not Joga: the screen does not single out targets. Induced polarization, EM and soil geochemistry along the eastern margin between Taranagar and Joga are the next step.</li>
  <li><b>Copper.</b> Under the Mincheri block the models hold only a thin, shallow dense layer (up to {mi['max_density']:.2f} g/cc·km, centred {mi['dense_centroid_km']:.1f} km deep) and weak magnetization (up to {mi['max_susceptibility']:.2f} SI·km), consistent with the narrow low-grade ridges mapped there {cite('NMET')} and with no large body. Copper sulphides need electrical and EM methods.</li>
</ul>
</div>

<h2><span class="no">7</span>Recommended programme</h2>
{captioned(6, "A phased programme with a decision after each phase", programme_table())}
<div class="prose">
<p>The first phase uses data already in hand. The aeromagnetic survey was flown 80 m above the ground on lines 300 m apart, oriented NE–SW across the strike of the belt, and gridded at 37.5 m: inverted without upward continuation on a finer mesh, it can resolve the iron-formation layers that the regional 1 km models merge, and test R1 and F1 at a cost of hours of cloud computing.</p>
</div>

<h2><span class="no">8</span>Risks and limitations</h2>
{captioned(7, "Risks and their mitigation", risks_table())}

<h2>Appendix: method notes</h2>
<div class="prose">
<ul class="plain">
  <li><b>Models.</b> Density from the joint gravity–magnetic inversion with the measured rock densities as bounds (−0.15 to +0.35 g/cc around 2.66 g/cc) and a depth weighting of β = 0.5; susceptibility and the magnetization-vector amplitude from the magnetic study (β = 1); 1 km × 1 km × 250 m cells. Details: the gravity, magnetic and joint inversion reports.</li>
  <li><b>Iron-formation horizons.</b> 1 km columns with integrated susceptibility ≥ {M['thresholds']['magnetic']:g} SI·km within {M['thresholds']['ridge_km']:g} km of a schematic ridge or inside a schist belt; F-segments: those more than {M['thresholds']['mine_km']:g} km from the four located mines, at least 3 connected columns.</li>
  <li><b>R1.</b> Columns with integrated MVI amplitude ≥ {M['thresholds']['magnetic']:g} SI·km within 2 km of a station the induced model underfits by more than 150 nT, within 10 km of Kumaraswamy.</li>
  <li><b>Gold screen.</b> Within {M['thresholds']['gold_dense_km']:g} km of a dense column (≥ {M['thresholds']['dense']:g} g/cc·km), within {M['thresholds']['gold_bif_km']:g} km of an iron-formation horizon, integrated susceptibility &lt; {M['thresholds']['weak']:g} SI·km.</li>
  <li><b>Schematic map.</b> Drawn from the 450 m DEM of the inputs; no published map is copied (<code>karnataka_inputs/shared/literature.py</code>).</li>
  <li><b>Scripts.</b> <code>examples/output/karnataka_minerals/scripts/make_figures.py</code> (screens), <code>validation.py</code> (the localities against the models), <code>build_report.py</code>.</li>
</ul>
</div>
{L.references_html(REFS)}
<p class="disclaimer">This assessment screens publicly available data and published information; it has not been verified in the field. It is not a Mineral Resource or Ore Reserve estimate, nor an Exploration Target under the JORC Code or NI 43-101, and should not be used as one. Reserve and production figures are those reported by the cited sources.</p>
<footer>GeoInv3D · Generated by build_report.py; the model values come from the runs of the three inversion reports, the published values from the sources cited.</footer>
"""


def main():
    fig = Figures("en", FIGS)
    html = body(fig)
    OUT.write_text(page("en", "Sandur Mineral Prospectivity", html, extra_css=L.LIT_CSS + EXTRA_CSS), encoding="utf-8")
    print(OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")
    if "--pdf" in sys.argv:
        print(to_pdf(OUT))


if __name__ == "__main__":
    main()
