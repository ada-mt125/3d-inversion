"""The mineral assessment of the Sandur area: English HTML (and PDF).

    py examples/output/karnataka_minerals/scripts/make_figures.py
    py examples/output/karnataka_minerals/scripts/build_report.py [--pdf]

A synthesis of the three Karnataka reports (gravity, magnetics, the joint inversion) and the
published record; no new inversion.  Every number from the models comes from figures/numbers.json
(this report's screens) and karnataka_joint/figures_v2/numbers.json (the joint report); the
published values come from the sources cited.
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
C, LIT, MAG = J["constraints"], J["literature"], J["magnetic"]
OUT = ROOT / "karnataka_mineral_report_en.html"
REFS = ["MM93", "MK12", "GSI", "IJERT", "MEAI", "MGR23", "IBMK", "IBMN", "GEM", "ROM", "SK18", "SB14", "SIN20",
        "BHAT25", "NMET"]
cite, citet = L.cite, L.citet


def pct(x):
    return f"{100 * x:.0f}%"


def dist(table_key, name):
    return next(r for r in LIT["distances"][table_key] if r["name"] == name)


def mines_range(table_key):
    rows = [r for r in LIT["distances"][table_key] if r["kind"] == "iron"]
    return (max(r["distance_km"] for r in rows), min(r["area_as_close"] for r in rows),
            max(r["area_as_close"] for r in rows))


# ---------------------------------------------------------------- tables

def captioned(n, title, html):
    return f'<p class="note"><b>Table {n}</b> · {title}</p>\n{html}'


def succession_table():
    rows = [
        ("Iron formation (Donimalai Formation, top of the succession)",
         "Banded iron formation: amphibole, hematite, magnetite and chert; Algoma type, conformable",
         "Iron ore by supergene enrichment (Kumaraswamy, Donimalai, Ramandurg, NEB Range); host of gold in quartz "
         "veins (Joga)", cite("GSI", "ROM", "SB14")),
        ("Manganiferous phyllite with chert (Deogiri Formation)", "Below the iron formation in the western sequence",
         "Manganese ore, discontinuous over about 40 km of strike", cite("GSI", "IJERT", "MM93")),
        ("Quartzite, carbonate, greywacke, argillite", "Metasedimentary rocks of the flanks", "—", cite("MM93")),
        ("Metabasalt (base of the western sequence; the central metavolcanic terrane)",
         "Dense, weakly magnetic metavolcanic rock; sulphidic chert interbedded", "Gold in sulphidic chert (Taranagar)",
         cite("MM93", "SIN20")),
        ("Granite and gneiss (around the belt)", "The Closepet granite complex; younger granites (2.5–2.6 Ga) override the "
         "eastern margin", "—", cite("MM93", "MGR23")),
    ]
    return table(["Unit (top to bottom)", "Rocks", "Ore", "Source"], [[(f"<b>{a}</b>", "wrap"), (b, "wrap"), (c, "wrap"),
                                                                       (d, "wrap")] for a, b, c, d in rows],
                 numeric_from=9)


def deposits_table():
    def at(name):
        p = next((m for m in L.MINES if m[0] == name), None) or next(g for g in L.GOLD if g[0] == name)
        return f"{p[1]:.1f}, {p[2]:.1f}"
    w, e, s, n = L.MINCHERI
    rows = [
        ("Iron", "Kumaraswamy (NMDC)", "Supergene hematite ore in iron formation; two E–W bands: B block 1.2 km × "
         "650–700 m, C block 3.5 km × 450 m", "Balance reserves 195.17 Mt at 62.01% Fe (1 April 2023)",
         "Mean drilled depth 70 m (B), 62 m (C)", at("Kumaraswamy"), cite("IBMK")),
        ("Iron", "Donimalai (NMDC)", "Iron formation of amphibole, hematite, magnetite and chert, enriched near the "
         "surface", "Reserves 127.6 Mt; capacity 7 Mt/a, 5.7 Mt mined in 2024", "Ore grade in the upper 170 m",
         at("Donimalai"), cite("GEM", "ROM")),
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


def mines_table():
    rows = []
    for name, *_ in L.MINES:
        s, d, v = dist("susceptibility_single", name), dist("density_single", name), dist("mvi_single", name)
        rows.append([f"<b>{name}</b>", f"{s['distance_km']:.1f}", pct(s["area_as_close"]), f"{d['distance_km']:.1f}",
                     pct(d["area_as_close"]), f"{v['distance_km']:.1f}", pct(v["area_as_close"])])
    for name, *_ in L.GOLD:
        s, d, v = dist("susceptibility_single", name), dist("density_single", name), dist("mvi_single", name)
        rows.append([f"{name} (gold)", f"{s['distance_km']:.1f}", pct(s["area_as_close"]), f"{d['distance_km']:.1f}",
                     pct(d["area_as_close"]), f"{v['distance_km']:.1f}", pct(v["area_as_close"])])
    return table(["Locality", "To a strongly magnetic column (km)", "Area as close", "To a dense column (km)",
                  "Area as close", "To a strong MVI column (km)", "Area as close"], rows, compact=True)


def segments_table():
    notes = {0: "Dense and magnetic body in the centre of the belt, under the main gravity high",
             1: "Western (Sandur) range north-west of NEB Range"}
    rows = []
    for i, s in enumerate(M["segments"], 1):
        side = "western" if s["easting"] < 666 else "eastern"
        if s["easting"] < 656:
            place = "North-western end of the belt, towards Hosapete"
        elif s["in_belt"] < 0.5:
            place = f"Beside the {side} range, mostly outside the schematic belt"
        else:
            place = f"{side.capitalize()} range"
        place = notes.get(i - 1, place)
        rows.append([f"<b>F{i}</b>", f"{s['easting']:.1f}, {s['northing']:.1f}", f"{s['cells']}",
                     f"{s['length_km']:.0f}", f"{s['mean_susceptibility']:.2f}", pct(s["dense_share"]),
                     f"{s['nearest_mine_km']:.1f}", f"{s['town']} {s['town_km']:.0f} km", (place, "wrap")])
    return table(["Segment", "Centre E, N (km)", "Area (km²)", "Length (km)", "Mean integrated susceptibility (SI·km)",
                  "Columns that are dense", "To the nearest located mine (km)", "Nearest town", "Where"], rows,
                 compact=True)


# ---------------------------------------------------------------- body

def body(fig):
    k = C["full_key"]
    main = C["full"][k]["none"]["main"]
    sh = LIT["shape"]["density_unconstrained"]
    far_s, lo_s, hi_s = mines_range("susceptibility_single")
    far_d, lo_d, hi_d = mines_range("density_single")
    sm = LIT["schematic_match"]
    a, b = MAG["induced"], MAG["mvi"]
    z = dict(b.get("magnetization") or {})
    segs = M["segments"]
    loc = M["localities"]
    mi = M["mincheri"]
    m0, m1 = LIT["magnetic_centroid_km"]
    rocks = C["rocks"]
    bg = C["background"]
    mb, bif, gr = rocks["Metabasalt / amphibolite"], rocks["Banded iron formation"], rocks["Granite"]
    return f"""
<header>
  <div class="eyebrow">GeoInv3D · mineral assessment · synthesis of the gravity, magnetic and joint reports · 1 October 2026</div>
  <h1>Sandur Schist Belt, Karnataka: A Mineral Assessment from Gravity, Magnetics and the Published Record</h1>
  <p class="lede">What the three inversion studies of the Sandur area and the published geology and mining record say about its iron, manganese, gold and copper: where the host rocks are, how the known deposits sit in the models, which ground the models single out for follow-up, and what they cannot see.</p>
  <dl class="meta">
    <div><dt>Area</dt><dd>70 × 70 km around the Sandur schist belt, easting 641–711 km, northing 1634–1704 km (UTM 43N), Ballari district, Karnataka</dd></div>
    <div><dt>Basis</dt><dd>The gravity, magnetic and joint inversion reports of 28 September to 1 October 2026 (1 km × 1 km × 250 m cells), the measured rock samples, and {len(REFS)} published sources; no new data, no field check</dd></div>
    <div><dt>Models used</dt><dd>Density from the joint run with the rock-sample bounds (−0.15 to +0.35 g/cc, β = 0.5); susceptibility along the present field and the magnetization-vector amplitude of the magnetic study (β = 1)</dd></div>
  </dl>
</header>

<section class="summary" aria-labelledby="sum">
  <h2 id="sum">Summary</h2>
  <ul>
    <li><b>An iron district with a manganese horizon and minor gold.</b> The belt holds about 1,876 Mt of iron ore at about 63% Fe in six ranges {cite('GSI')}; the four mines located here together report about {round(195.17 + 127.6 + 47.7 + 2.6, -1):.0f} Mt of reserves, Kumaraswamy alone 195 Mt at 62% Fe {cite('IBMK', 'GEM', 'IBMN')}. Manganese lies in the Deogiri Formation over about 40 km of strike {cite('IJERT')}; the gold occurrences are low grade (up to 0.49 g/t and 1.48 ppm) {cite('SB14', 'SIN20')}, and copper is known only from a reconnaissance block {cite('NMET')}.</li>
    <li><b>The ore lies in the iron formation, and the models find the iron formation.</b> The iron formation is a layer at the top of the belt's succession, folded steeply onto both flanks of a dense metavolcanic core {cite('MM93', 'GSI')}. The models show it most clearly in the dense rock and in section, and every iron mine lies on or within {far_s:.1f} km of a strongly magnetic column, a condition met by only {pct(lo_s)} to {pct(hi_s)} of the area.</li>
    <li><b>Iron: {M['unmined_km2']} km² of iron-formation horizons lie more than {M['thresholds']['mine_km']:g} km from the four located mines</b>, in {len(segs)} segments (F1 to F{len(segs)}). The largest, F1 ({segs[0]['cells']} km²), is a dense and strongly magnetic body in the centre of the belt; F2 follows the western range for about {segs[1]['length_km']:.0f} km. The belt has more leases than the four located here, so these segments are to be checked against the lease and geological maps before they are called new.</li>
    <li><b>A remanent zone south of Kumaraswamy.</b> The magnetic data there need a magnetization not along the present field (RMS {a['rms']:.0f} → {b['rms']:.0f} nT with a magnetization vector); its strong cells cover {M['remanent_km2']} km² and reach {M['remanent_south_km']:.0f} km south of the Kumaraswamy B block, under the richest part of the belt. It is the most specific iron target of the models, and untested.</li>
    <li><b>Manganese, gold and copper are below what these data resolve.</b> Manganese oxide is neither dense nor magnetic enough at 1 km; the models only locate the iron formation that overlies it in the succession. The dense, magnetite-poor ground beside the iron formation where sulphide-facies gold could lie covers {pct(M['gold_in_belt_share'])} of the belt, which singles nothing out; under the Mincheri copper block the models hold only a thin, shallow dense layer.</li>
    <li><b>Next:</b> check F1 to F{len(segs)} and the remanent zone against the 1:50,000 geology and the lease map; invert the 37.5 m magnetic grid without upward continuation over the belt; ground magnetics and oriented samples south of Kumaraswamy; induced polarization and electromagnetics, not potential fields, for gold and copper.</li>
  </ul>
</section>

<h2><span class="no">1</span>Geology of the area</h2>
<div class="prose">
<p>The Sandur schist belt is one of the Archaean greenstone belts of the Dharwar craton. It lies within the Closepet granite complex and preserves rocks common to both the Eastern and Western Dharwar cratons {cite('MGR23', 'MK12')}. It is mapped as a NW–SE belt about 60 km long and up to 18 km wide, enclosed by granite and gneiss {cite('MM93', 'GSI')}, and it stands out as a ring of elongated hills 900 to 1,050 m above sea level around a lower centre, the Copper Mountain range in the east and the Sandur range in the west {cite('MEAI')}.</p>
<p><b>The succession.</b> A greenstone belt is the whole Archaean supracrustal pile: mafic and ultramafic volcanic rocks, minor felsic volcanics and sediments. Its banded iron formation, mostly of Algoma type (single layers a few metres to a few hundred metres thick, a few to tens of kilometres along strike), is a chemical sediment interbedded with volcanics, chert and clastic rocks, usually at the end of a volcanic cycle; it lies conformably in the upper part of the succession and is the host of the ore. In the Sandur belt the succession is divided into the Yeshwantnagar, Deogiri and Donimalai formations from the bottom up; the Deogiri Formation carries the manganese ore and the iron ore goes with the iron formation of the Donimalai Formation {cite('GSI', 'IJERT')}. The western sequence runs from metabasalt at the base through quartzite, carbonate, greywacke and argillite to manganiferous phyllite with chert, with the iron formation at the top {cite('MM93')}; the Geological Survey of India places the assemblage in the Bababudan type {cite('GSI')} (Table 1).</p>
<p><b>The structure.</b> The belt is strongly folded: bedding is steep (strike 115° to 150° in the east), the early folds are near-isoclinal and plunge about 45° to the north, and two metasedimentary belts flank a central metavolcanic terrane without joining at either end {cite('MM93')}. The iron formation, stratigraphically above the metabasalt, therefore stands in near-vertical sheets on both flanks rather than lying on top of the greenstone.</p>
</div>
{captioned(1, "The succession of the Sandur belt and its ores", succession_table())}
{fig('geology', "Schematic geology of the area with the published localities. Drawn for this report from the 450 m DEM of the inputs, not a geological map and not copied from any publication: the ridges are where the ground stands more than 80 m above the median of a 12 km window, read as the iron-formation ridges because the iron formation holds up the hills of the belt " + cite('MEAI') + f"; the schist belts are the envelope of the ridges (the main belt {M['schematic']['length_km']:.0f} × {M['schematic']['width_km']:.0f} km, strike {M['schematic']['strike_deg']:.0f}°); the central metavolcanic terrane after " + citet('MM93') + ". The mapped belt is longer, about 60 km, continuing north-west towards Hosapete. " + L.LEGEND)}

<h2><span class="no">2</span>The known deposits and occurrences</h2>
{captioned(2, "The known deposits and occurrences of the area", deposits_table())}
<p class="note">{L.POSITIONS_NOTE} Reserves and production as reported by the sources, not re-estimated.</p>
<div class="prose">
<p>The iron ore is the product of supergene enrichment of the iron formation near the surface: at Kumaraswamy two E–W bands with a mean drilled depth of 62 to 70 m {cite('IBMK')}, at Donimalai ore grade within the upper 170 m of an amphibole–hematite–magnetite–chert iron formation {cite('ROM')}, and at Obulapuram, in the same belt, a hematite and magnetite ore altered from it {cite('BHAT25')}. The manganese lies below the iron formation in the succession, in the Deogiri Formation. The gold is of two kinds, both in the chemical sediments: quartz veins in the iron formation at Joga and sulphidic chert at Taranagar {cite('SB14', 'SIN20')}. The copper of the Mincheri block, on the eastern edge of the area, is sporadic, in narrow ridges of iron-rich quartzite and chert {cite('NMET')}.</p>
</div>

<h2><span class="no">3</span>What the inversions show</h2>
<div class="prose">
<p>The three reports inverted the Bouguer gravity and the aeromagnetic data of the area on 1 km cells; the joint report bounded the density by the measured rock samples ({mb['block']['rho_median']:.2f} g/cc for the metabasalt, {bif['block']['rho_median']:.2f} for the iron formation, {gr['block']['rho_median']:.2f} for the granite, against a background of {bg} g/cc) and let the belt reach the ground. Their main results are these: the position and outline of the belt and of its magnetic sheets are shared by every setting and coupling, while the depths follow the bounds and the depth weighting.</p>
<ul class="plain">
  <li><b>A dense metavolcanic core.</b> The dense belt runs NW–SE (strike {sh['strike_deg']:.0f}°), about {sh['length_km']:.0f} × {sh['width_km']:.0f} km; {pct(sm['density_constrained']['in_outline'])} of its strongly dense columns lie inside the schematic belt outline, which covers {pct(sm['density_constrained']['area_in_outline'])} of the area. With the sample bounds its half-maximum under the main gravity high lies at {main['top_km']:.1f}–{main['bottom_km']:.1f} km and its base at about {main['dense_bottom_km']:.1f} km, close to the one published estimate, a basin about 6 km deep {cite('MGR23')}.</li>
  <li><b>Steep iron-formation sheets on both flanks.</b> The strongly magnetic rock forms thin sheets that reach the surface on the ridges of both ranges and at the south-eastern closure (Figures {fig.ref('models')}b and {fig.ref('section')}); {pct(sm['susceptibility_single']['near_ridge'])} of the strongly magnetic columns lie within 1 km of a ridge, {sm['susceptibility_single']['near_ridge'] / sm['susceptibility_single']['area_near_ridge']:.0f} times the share of the area (5 km edge band left out). Their centroids lie {m0:.1f} to {m1:.1f} km deep, which describes the iron-formation belt as a whole, not the ore.</li>
  <li><b>A dense core with magnetic flanks.</b> A simple synform would put its youngest rock, the iron formation, in its core. The models put the dense metavolcanic rock in the centre and the steep magnetic sheets on the flanks, which is the reading of {citet('MM93')}: two iron-bearing metasedimentary belts flanking a central metavolcanic terrane. The pattern is clearest in the dense rock and in section; in plan a strongly magnetic body also lies in the centre (F1, Section 4.1).</li>
  <li><b>Remanent magnetization in the south.</b> A magnetization vector per cell lowers the magnetic RMS from {a['rms']:.0f} to {b['rms']:.0f} nT; its strongly magnetized cells point at I {z.get('resultant_inclination', 0):.0f}°, D {str(round(z.get('resultant_declination', 0))).replace('-', '−')}°, not along the present field, under and south of Kumaraswamy.</li>
</ul>
</div>
{fig('models', "The three models of the area, each integrated over depth on the 1 km columns: (a) density contrast of the joint run with the rock-sample bounds, (b) susceptibility along the present field, (c) amplitude of the magnetization vector. Black: the schematic belt outline (solid) and ridges (dotted) of Figure " + str(fig.ref('geology')) + ". " + L.LEGEND)}
{fig('section', f"E–W section through the belt at northing {1664.5:.1f} km (through the main gravity high and Donimalai): the density contrast and susceptibility of the joint run with the rock-sample bounds, with the interpretation of Section 1. The black line is the ground of the mesh.")}
<div class="prose">
<p><b>The mines sit on the predicted host rock.</b> The test of the models is where the known deposits fall (Table 3). Every iron mine lies on or within {far_s:.1f} km of a strongly magnetic column (integrated susceptibility ≥ {LIT['thresholds']['magnetic']:g} SI·km), which only {pct(lo_s)} to {pct(hi_s)} of the area does, and on or within {far_d:.1f} km of a dense column (≥ {LIT['thresholds']['dense']:g} g/cc·km; {pct(lo_d)} to {pct(hi_d)} of the area). At Donimalai and Ramandurg the strongly magnetic column lies 1 km beside the mine, and the mine's own column is weakly magnetic ({loc['Donimalai']['susceptibility']:.2f} and {loc['Ramandurg']['susceptibility']:.2f} SI·km): the enriched ore is hematite-rich {cite('ROM', 'BHAT25')}, and hematite is far less magnetic than the magnetite of the fresh iron formation beside it. At 1 km cells this is suggestive, not proven.</p>
</div>
{captioned(3, "The localities against the models", mines_table())}
<p class="note">Distance from each locality to the nearest 1 km column above the threshold, and the share of the area (5 km edge band left out) that lies at least as close to such a column; a small share means that the locality is not where a random point would fall. Susceptibility and MVI: the magnetic study, β = 1; density: the gravity study with terrain, β = 1.</p>

<h2><span class="no">4</span>Commodity by commodity</h2>

<h3><span class="no">4.1</span>Iron</h3>
<div class="prose">
<p><b>The screen.</b> The iron ore of the belt is enriched iron formation, so the first question is where the iron formation is. Columns that are strongly magnetic (≥ {M['thresholds']['magnetic']:g} SI·km) and lie within {M['thresholds']['ridge_km']:g} km of a ridge or inside a schist belt are taken as iron-formation horizons: {M['bif_km2']} km², {pct(M['bif_share'])} of the area. All four located mines lie within 1.5 km of one. Of these horizons, {M['unmined_km2']} km² lie more than {M['thresholds']['mine_km']:g} km from the four mines, in {len(segs)} segments of at least 3 km² (Figure {fig.ref('prospectivity')}, Table 4).</p>
<ul class="plain">
  <li><b>F1</b> ({segs[0]['cells']} km², centre {segs[0]['easting']:.1f}, {segs[0]['northing']:.1f} km) is not on a ridge: a dense ({pct(segs[0]['dense_share'])} of its columns) and strongly magnetic body in the centre of the belt, under the main gravity high between Ramandurg and Kumaraswamy. In the structural reading of the belt the centre is metavolcanic {cite('MM93')}, so F1 is either iron formation folded into the core or magnetite-bearing mafic to ultramafic rock; the gravity and magnetic data alone cannot tell which. It is the largest and least explained magnetic body of the belt.</li>
  <li><b>F2</b> ({segs[1]['cells']} km², about {segs[1]['length_km']:.0f} km long) follows the western (Sandur) range north-west of NEB Range, a mapped iron-formation range {cite('MEAI')}. Some of it may be held by leases not located here: the belt has six iron ranges, among them Kanavehalli, Devagiri and Thimmappanagudi {cite('GSI')}.</li>
  <li><b>F3 to F{len(segs)}</b> are shorter segments of the two ranges and of the north-western end of the belt towards Hosapete.</li>
</ul>
<p><b>The remanent zone.</b> South of Kumaraswamy the induced model underfits the data by more than 150 nT and the magnetization-vector model places strongly magnetized cells ({M['remanent_km2']} km², easting {M['remanent_extent'][0]:.0f}–{M['remanent_extent'][1]:.0f} km, northing {M['remanent_extent'][2]:.0f}–{M['remanent_extent'][3]:.0f} km). They reach {M['remanent_south_km']:.0f} km south of the Kumaraswamy B block, beyond the mapped ridges. This is the richest part of the belt; hematite and magnetite ore with a remanent magnetization would explain the anomaly, and so would a magnetite-rich unit under cover. No palaeomagnetic data from the belt were found, so the zone is a target for ground magnetics and oriented samples, not a resource.</p>
<p><b>What the models cannot do.</b> The ore bodies lie within 70 to 170 m of the surface, within the top cell of the mesh, and the data were continued 1 km upwards before the inversion. The screen ranks belt segments; it does not delineate ore.</p>
</div>
{fig('prospectivity', "The iron screens on the 1 km columns: iron-formation horizons (strongly magnetic within 1.5 km of a ridge or inside a schist belt) within 3 km of a located mine (grey) and farther from them (red, segments F1 to F" + str(len(segs)) + " of Table 4), and the remanent zone south of Kumaraswamy (purple: strong MVI amplitude where the induced model underfits the data by more than 150 nT). Green: the schist belts of the schematic. " + L.LEGEND)}
{captioned(4, "Iron-formation horizons more than 3 km from the four located mines", segments_table())}
<p class="note">Segments of at least 3 connected 1 km columns, sorted by area. "Located mines": the four of Table 2; the belt has more leases, so a segment far from them is not necessarily unworked.</p>

<h3><span class="no">4.2</span>Manganese</h3>
<div class="prose">
<p>The manganese ore lies in the Deogiri Formation, manganiferous phyllite with chert below the iron formation, discontinuous over about 40 km of strike {cite('GSI', 'IJERT', 'MM93')}. Manganese oxide in phyllite is neither dense enough nor magnetic enough to show at 1 km cells, so the models give no direct guide. Because the manganese horizon lies just below the iron formation in the succession and the beds are steep, the iron-formation horizons of Section 4.1 mark the belts in which it should be sought; within them it needs mapping and geochemistry.</p>
</div>

<h3><span class="no">4.3</span>Gold</h3>
<div class="prose">
<p>The known gold is in the chemical sediments: quartz veins in the iron formation at Joga and banded sulphidic chert at Taranagar {cite('SB14', 'SIN20')}. Sulphide-facies iron formation carries little magnetite, so a gold-bearing unit would be dense but weakly magnetic. Both occurrences lie in or within 1 km of the dense belt; Taranagar lies {dist('susceptibility_single', 'Taranagar')['distance_km']:.0f} km from a strongly magnetic column and Joga {dist('susceptibility_single', 'Joga')['distance_km']:.0f} km. Ground that is dense (within {M['thresholds']['gold_dense_km']:g} km of a dense column), within {M['thresholds']['gold_bif_km']:g} km of an iron-formation horizon and itself weakly magnetic (&lt; {M['thresholds']['weak']:g} SI·km) covers {M['gold_km2']} km², {pct(M['gold_in_belt_share'])} of the belt: it includes Taranagar but not Joga, whose host iron formation is weakly magnetic at this scale. A screen that covers {pct(M['gold_in_belt_share'])} of the belt does not single out targets. Gold in sulphidic units is a target for induced polarization and electromagnetics, soil geochemistry and structural mapping of the contacts, not for 1 km potential-field models.</p>
</div>

<h3><span class="no">4.4</span>Copper</h3>
<div class="prose">
<p>The copper of the Mincheri block lies in narrow, low-grade ridges of banded hematite quartzite, ferruginous chert and magnetite quartzite {cite('NMET')}. Under the part of the block inside the area ({mi['km2']} km²) the joint model holds only a thin, shallow dense layer (integrated density up to {mi['max_density']:.2f} g/cc·km, centroid {mi['dense_centroid_km']:.1f} km below the ground; no column reaches {LIT['thresholds']['dense']:g} g/cc·km) and weak magnetization (up to {mi['max_susceptibility']:.2f} SI·km), consistent with the narrow ridges and with no large dense or magnetic body. The north-eastern belt that runs into the block is weakly dense (mean {M['ne_belt_mean_density']:.2f} g/cc·km) and has {M['ne_belt_magnetic_km2']} km² of strongly magnetic columns; Maurya et al. report a previously unmapped arm of the schist belt from gravity and magnetic data {cite('MGR23')}. Copper sulphides are conductive and chargeable; their search needs electrical and electromagnetic methods.</p>
</div>

<h2><span class="no">5</span>Confidence and limits</h2>
<div class="prose">
<ul class="plain">
  <li><b>Scale.</b> The models have 1 km × 1 km × 250 m cells on data continued 1 km upwards; ore bodies within 170 m of the surface and gold-bearing units metres to tens of metres thick are far below that.</li>
  <li><b>Depth.</b> The depth of the dense body follows the density bounds and the depth weighting; the constrained base agrees with a published estimate that itself rests on an assumed contrast {cite('MGR23')}. Do not interpret the base of the magnetic bowl {cite('MM93')}.</li>
  <li><b>Remanence.</b> Supported by the magnetic data, untested by any published palaeomagnetic measurement.</li>
  <li><b>The schematic map</b> is drawn from the DEM; the mapped belt continues beyond its ridges. It is a guide for comparison, not geology.</li>
  <li><b>The record.</b> Four mines and two gold occurrences are located, to about 1 km; the belt has more leases. Reserves and grades are as reported by the sources.</li>
</ul>
</div>

<h2><span class="no">6</span>Recommendations</h2>
<div class="prose">
<ol class="steps">
  <li><b>Check F1 to F{len(segs)} and the remanent zone against the 1:50,000 geology and the lease map</b> (GSI Bhukosh; the IBM lease records) to separate known ranges from unworked ground.</li>
  <li><b>Invert the 37.5 m magnetic grid without upward continuation</b> over the belt, on a finer mesh: iron-formation bands with tops at 70 to 130 m {cite('SK18')} are resolved there, not at 1 km.</li>
  <li><b>Ground magnetics and oriented samples south of Kumaraswamy</b> to test the remanent zone and its direction (I {z.get('resultant_inclination', 0):.0f}°, D {str(round(z.get('resultant_declination', 0))).replace('-', '−')}°), and over F1 to tell folded iron formation from magnetite-bearing mafic rock.</li>
  <li><b>Induced polarization, resistivity and electromagnetics</b> along the eastern range between Taranagar and Joga and over the Mincheri block, with soil geochemistry, for sulphide-hosted gold and copper.</li>
  <li><b>Calibrate with the mines' drilling</b>: the drilled ore of Kumaraswamy and Donimalai gives the density, susceptibility and depth of ore that the next, finer inversions should reproduce.</li>
</ol>
</div>

<h2>The three reports</h2>
<div class="prose">
<ul class="plain">
  <li>Gravity: <code>examples/output/karnataka_gravity/karnataka_gravity_report_en.html</code> (Section 6: the published geology)</li>
  <li>Magnetics: <code>examples/output/karnataka_magnetic/karnataka_magnetic_report_en.html</code> (Section 6)</li>
  <li>Joint inversion: <code>examples/output/karnataka_joint/karnataka_joint_report_v2_en.html</code> (Sections 5 and 7)</li>
  <li>This report: <code>examples/output/karnataka_minerals/scripts/make_figures.py</code> (the screens) and <code>build_report.py</code>; the references, localities and schematic map in <code>karnataka_inputs/shared/literature.py</code></li>
</ul>
</div>
{L.references_html(REFS)}
<footer>GeoInv3D · Generated by build_report.py; the model values come from the runs of the three reports, the published values from the sources cited.</footer>
"""


def main():
    fig = Figures("en", FIGS)
    html = body(fig)
    OUT.write_text(page("en", "Sandur Mineral Assessment", html, extra_css=L.LIT_CSS), encoding="utf-8")
    print(OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")
    if "--pdf" in sys.argv:
        print(to_pdf(OUT))


if __name__ == "__main__":
    main()
