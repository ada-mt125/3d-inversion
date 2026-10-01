"""The published geology of the Sandur area, shared by the Karnataka reports: references, the
mapped mines and occurrences, a schematic geological map drawn from our DEM, and helpers to
put them on the figures.

The schematic map copies no published figure.  The ridges are where the ground stands more
than RIDGE_RELIEF_M above its surroundings in the 450 m DEM of the inputs; the iron formation
holds up the hills of the belt (MEAI 2024), so they stand for the metasedimentary belts with
iron formation.  The outline of the schist belt is the envelope of the ridges (a closing of
about 4 km and the holes filled), the central terrane of metavolcanic rock lies between them
(Mukhopadhyay & Matin 1993), and granite and gneiss are everywhere else.  It is a sketch for
comparison, not a geological map.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

INPUTS = Path(__file__).resolve().parents[1]
DEM = INPUTS / "dem_utm43n_450m.tif"
AOI_KM = (641.0, 711.0, 1634.0, 1704.0)
RIDGE_RELIEF_M = 80.0         # ground above the median of a 12 km window
RIDGE_MIN_M = 600.0           # and above this elevation

# ---------------------------------------------------------------- references

REFS = {
    "MM93": 'Mukhopadhyay, D. and Matin, A. (1993). The structural anatomy of the Sandur schist belt, a greenstone belt in the Dharwar craton of South India. <i>Journal of Structural Geology</i>, 15(3–5), 309–322. <a href="https://www.sciencedirect.com/science/article/abs/pii/019181419390128W">www.sciencedirect.com</a>',
    "MK12": 'Manikyamba, C. and Kerrich, R. (2012). Eastern Dharwar Craton, India: continental lithosphere growth by accretion of diverse plume and arc terranes. <i>Geoscience Frontiers</i>, 3(3), 225–240. <a href="https://www.sciencedirect.com/science/article/pii/S1674987111001125">www.sciencedirect.com</a>',
    "GSI": 'Geological Survey of India (n.d.). <i>Geology and mineral maps of Karnataka</i>, explanatory report (Sandur schist belt, iron and manganese resources). <a href="https://prod-qt-images.s3.amazonaws.com/indiawaterportal/import/sites/default/files/iwp2/Report_Karnataka_State_Geology_and_Mineral_Maps_Geological_Survey_of_India.pdf">prod-qt-images.s3.amazonaws.com</a>',
    "IJERT": 'Anon. (n.d.). Geochemistry and ore distribution pattern of the manganese ores of Sandur area, Karnataka. <i>International Journal of Engineering Research &amp; Technology</i>. <a href="https://www.ijert.org/geochemistry-and-ore-distribution-pattern-of-the-manganese-ores-of-sandur-area-karnataka">www.ijert.org</a>',
    "MEAI": 'Mining Engineers\' Association of India (2024). <i>Iron ore deposits of Dharwar Craton: Sandur schist belt</i> (presentation). <a href="https://meai.org/wp-content/uploads/2024/02/Fe-Ore-Sandur-presentation.pdf">meai.org</a>',
    "MGR23": 'Maurya, D., Ganguli, S.S. and Rani, P. (2023). Gravity magnetic appraisal of the Sandur Superterrane, Dharwar Craton, India for shallow crustal architecture and mineralization. Extended abstract, EAGE, EarthDoc. doi:10.3997/2214-4609.202375020. <a href="https://www.earthdoc.org/content/papers/10.3997/2214-4609.202375020">www.earthdoc.org</a>',
    "IBMK": 'Indian Bureau of Mines (2024). Inspection report, Kumaraswamy iron ore mine (NMDC Ltd), Sandur taluk. <a href="https://ibm.gov.in/writereaddata/files/171108657465fd1bee1b5femcdr_Kumarswamy_NMDC.pdf">ibm.gov.in</a>',
    "IBMN": 'Indian Bureau of Mines (2018). Inspection report, NEB Range iron ore mine (ML 2150), Sandur taluk. <a href="https://ibm.gov.in/writereaddata/files/09052018120832mcdr_rep_NEB%20Range%202150%20of%20H%20G%20Rangan%20Goud.pdf">ibm.gov.in</a>',
    "GEM": 'Global Energy Monitor, GEM.wiki: <a href="https://www.gem.wiki/NMDC_Donimalai_Mine">NMDC Donimalai Mine</a> and <a href="https://www.gem.wiki/SKME_Ramandurga_Mine">SKME Ramandurga Mine</a> (mine coordinates, reserves). Accessed 1 October 2026.',
    "ROM": 'Turnstone Geological Services (n.d.). <i>Banded iron formation from Donimalai, southern India</i> (specimen note). <a href="https://www.turnstone.ca/rom98bif.htm">www.turnstone.ca</a>',
    "SK18": 'Satish Kumar, K., Srinivas, K.N.S.S.S., Pradeep Kumar, V., Prabhakara Prasad, P. and Seshunarayana, T. (2018). Magnetic mapping of banded iron formation of Sandur schist belt, Dharwar Craton, India. <i>Journal of the Geological Society of India</i>. doi:10.1007/s12594-018-0831-z. <a href="https://link.springer.com/article/10.1007/s12594-018-0831-z">link.springer.com</a>',
    "SB14": 'Suresh, S.R. and Basavanna, M. (2014). Geology and geochemistry of banded iron formations from Joga (Sandur schist belt) and associated gold mineralization. <a href="https://www.academia.edu/8314191/Geology_and_Geochemistry_of_Banded_Iron_Formations_from_Joga_Sandur_Schist_Belt_and_associated_gold_mineralization">www.academia.edu</a>',
    "SIN20": 'Sindhuja, C.S., Manikyamba, C., Pahari, A. and Satyanarayanan, M. (2020). Geochemistry of banded sulphidic cherts of Sandur greenstone belt, Dharwar Craton, India: constraints on hydrothermal processes and gold mineralization. <i>Ore Geology Reviews</i>. <a href="https://www.sciencedirect.com/science/article/abs/pii/S0169136820300809">www.sciencedirect.com</a>',
    "BHAT25": 'Bhat, N.A. et al. (2025). Geochemistry of banded iron ore from Obulapuram area of Sandur schist belt, southern India. <i>Journal of the Geological Society of India</i>. <a href="https://www.geosocindia.org/index.php/jgsi/article/view/174263">www.geosocindia.org</a>',
    "NMET": 'National Mineral Exploration Trust (n.d.). <i>Reconnaissance survey (G4) for copper in the Mincheri block, Ballari district, Karnataka</i> (MECL). <a href="https://nmet.gov.in/upload/uploadfiles/files/Mincheri_G4.pdf">nmet.gov.in</a>',
}
SHORT = {"MM93": "Mukhopadhyay &amp; Matin 1993", "MK12": "Manikyamba &amp; Kerrich 2012", "GSI": "GSI, Karnataka",
         "IJERT": "IJERT, Sandur Mn", "MEAI": "MEAI 2024", "MGR23": "Maurya et al. 2023",
         "IBMK": "IBM 2024, Kumaraswamy", "IBMN": "IBM 2018, NEB Range", "GEM": "GEM.wiki",
         "ROM": "Turnstone, Donimalai BIF", "SK18": "Satish Kumar et al. 2018", "SB14": "Suresh &amp; Basavanna 2014",
         "SIN20": "Sindhuja et al. 2020", "BHAT25": "Bhat et al. 2025", "NMET": "NMET, Mincheri G4"}
POSITIONS_NOTE = ("Mine and occurrence positions were converted to UTM 43N (EPSG:32643). Those taken from GEM.wiki "
                  "and from latitude–longitude ranges in the papers are approximate, to about 1 km, and so is the "
                  "outline of the Mincheri block. The Kumaraswamy and NEB Range coordinates are as printed in the "
                  "IBM inspection reports, read as UTM 43N.")

LIT_CSS = """
/* the published geology */
a.cite { color: var(--accent); text-decoration: none; border-bottom: 1px dotted var(--accent); }
a.cite:hover { border-bottom-style: solid; }
.refs ol { padding-left: 1.5em; display: grid; gap: 8px; font-size: 13.5px; line-height: 1.55; }
.refs a { overflow-wrap: anywhere; color: var(--accent); }
.added { font: 500 11px/1.4 var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted);
  border: 1px dashed var(--rule); border-radius: 3px; display: inline-block; padding: 2px 8px; margin: 0 0 10px; }
.tag { font: 600 11px/1.3 var(--mono); padding: 2px 7px; border-radius: 3px; display: inline-block; white-space: nowrap; }
.tag.ok { color: var(--good); background: var(--good-bg); }
.tag.mid { color: var(--mid); background: var(--mid-bg); }
.tag.no { color: var(--warn); background: var(--warn-bg); }
.tag.na { color: var(--muted); background: var(--bg); border: 1px solid var(--rule); }
:root { --mid: #8a5a00; --mid-bg: #fbf1dc; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --mid: #f0c674; --mid-bg: #3a2f17; } }
:root[data-theme="dark"] { --mid: #f0c674; --mid-bg: #3a2f17; }
.verdict { background: var(--surface); border: 1px solid var(--rule); border-left: 4px solid var(--good); border-radius: 0 6px 6px 0;
  padding: 16px 20px; margin: 18px 0 24px; max-width: 920px; }
.verdict h3 { margin-top: 0; }
.scores { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px 18px; margin: 12px 0 14px; }
.scores div { border-top: 2px solid var(--rule); padding-top: 6px; font-size: 13.5px; line-height: 1.5; }
.scores b { display: block; margin-bottom: 3px; }
"""


def cite(*keys, text=None):
    """An inline citation in parentheses: (<a class="cite" ...>Mukhopadhyay &amp; Matin 1993</a>; ...)."""
    if text is not None:
        return f'(<a class="cite" href="#ref-{keys[0]}">{text}</a>)'
    return "(" + "; ".join(f'<a class="cite" href="#ref-{k}">{SHORT[k]}</a>' for k in keys) + ")"


def citet(key):
    """A citation in the sentence: <a class="cite" ...>Mukhopadhyay &amp; Matin (1993)</a>."""
    short = SHORT[key]
    head, _, year = short.rpartition(" ")
    text = f"{head} ({year})" if year.isdigit() else short
    return f'<a class="cite" href="#ref-{key}">{text}</a>'


def references_html(keys, number=None):
    """The reference list of a report, in the order of ``keys``."""
    head = f'<h2><span class="no">{number}</span>References</h2>' if number else "<h2>References</h2>"
    items = "\n".join(f'<li id="ref-{k}">{REFS[k]}</li>' for k in keys)
    return f'{head}\n<div class="prose refs"><ol>\n{items}\n</ol>\n<p class="note">{POSITIONS_NOTE}</p></div>'


# ---------------------------------------------------------------- localities (UTM 43N, km)

MINES = [  # name, easting, northing, source, operator
    ("Kumaraswamy", 671.0, 1660.1, "IBMK", "NMDC"),
    ("Donimalai", 675.5, 1664.3, "GEM", "NMDC"),
    ("Ramandurg", 662.0, 1667.0, "GEM", "SKME"),
    ("NEB Range", 665.0, 1675.4, "IBMN", ""),
]
# the two development blocks of the Kumaraswamy lease (IBM 2024): (west, east, south, north)
KUMARASWAMY_BLOCKS = {"B": (672.56, 673.59, 1659.71, 1660.44), "C": (669.25, 670.87, 1660.00, 1660.39)}
GOLD = [  # name, easting, northing, source, what
    ("Joga", 668.5, 1680.0, "SB14", "gold up to 0.49 g/t in quartz veins in the iron formation"),
    ("Taranagar", 672.5, 1672.2, "SIN20", "gold 0.05 to 1.48 ppm in sulphidic chert"),
]
MINCHERI = (700.8, 711.0, 1661.2, 1668.2)   # approximate; the block reaches beyond the area
TOWNS = [("Hosapete", 649.4, 1688.5), ("Sandur", 666.4, 1668.3), ("Ballari", 706.5, 1674.6)]
LOCALITIES = ([(n, e, k, "iron") for n, e, k, *_ in MINES] + [(n, e, k, "gold") for n, e, k, *_ in GOLD])

LEGEND = ("White squares: iron-ore mines (the two small rectangles beside Kumaraswamy are its B and C "
          "development blocks); yellow stars: gold occurrences; dashed box: the Mincheri copper "
          "reconnaissance block (clipped at the edge of the area); black dots: towns.")


def plot_localities(ax, labels=True, towns=True, color="k", fontsize=7.5):
    """The mines, gold occurrences, the Mincheri block and the towns on a map in km."""
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=2.2, foreground="white")]
    for w, e, s, n in KUMARASWAMY_BLOCKS.values():
        ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
            (w, s), e - w, n - s, fill=False, ec=color, lw=0.9, zorder=6))
    for name, x, y, *_ in MINES:
        ax.plot(x, y, "s", ms=6.5, mfc="white", mec=color, mew=1.2, zorder=7)
        if labels:
            ax.annotate(name, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=fontsize,
                        fontweight="bold", color=color, path_effects=halo, zorder=8)
    for name, x, y, *_ in GOLD:
        ax.plot(x, y, "*", ms=10, mfc="#f1c40f", mec=color, mew=0.8, zorder=7)
        if labels:
            ax.annotate(f"{name} Au", (x, y), xytext=(5, -2), textcoords="offset points", fontsize=fontsize,
                        color=color, path_effects=halo, zorder=8)
    w, e, s, n = MINCHERI
    ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
        (w, s), e - w, n - s, fill=False, ec=color, lw=1.1, ls="--", zorder=6))
    if labels:
        ax.annotate("Mincheri Cu", (w, s), xytext=(2, -10), textcoords="offset points", fontsize=fontsize - 0.5,
                    color=color, path_effects=halo, zorder=8)
    if towns:
        for name, x, y in TOWNS:
            ax.plot(x, y, "o", ms=3.5, color=color, zorder=7)
            if labels:   # to the right near the western edge, to the left elsewhere
                right = x < AOI_KM[0] + 15
                ax.annotate(name, (x, y), xytext=(4 if right else -4, 4), textcoords="offset points",
                            fontsize=fontsize - 0.5, style="italic", ha="left" if right else "right", color=color,
                            path_effects=halo, zorder=8)
    ax.set_xlim(AOI_KM[0], AOI_KM[1])
    ax.set_ylim(AOI_KM[2], AOI_KM[3])


# ---------------------------------------------------------------- the schematic geological map

@lru_cache(maxsize=1)
def schematic():
    """(x km, y km, elevation, ridges, belt, belts labelled): the DEM over the area, the ridges and
    the outline of the schist belts (boolean grids), and the belts labelled 1, 2, ... by size."""
    import rasterio
    from scipy import ndimage as ndi
    with rasterio.open(DEM) as d:
        z = d.read(1).astype(float)
        tr = d.transform
    ny, nx = z.shape
    xc = (tr.c + tr.a * (np.arange(nx) + 0.5)) / 1e3
    yc = (tr.f + tr.e * (np.arange(ny) + 0.5)) / 1e3
    sx = (xc >= AOI_KM[0] - 3) & (xc <= AOI_KM[1] + 3)
    sy = (yc >= AOI_KM[2] - 3) & (yc <= AOI_KM[3] + 3)
    z, xc, yc = z[np.ix_(sy, sx)], xc[sx], yc[sy]
    relief = z - ndi.median_filter(z, size=27)                        # 27 × 450 m ≈ 12 km
    ridge = ndi.binary_opening((relief > RIDGE_RELIEF_M) & (z > RIDGE_MIN_M), iterations=1)
    lab, n = ndi.label(ridge)
    sizes = ndi.sum(ridge, lab, range(1, n + 1))
    ridge = np.isin(lab, 1 + np.where(sizes >= 6)[0])                 # at least about 1.2 km²
    yy, xx = np.mgrid[-9:10, -9:10]
    disk = xx ** 2 + yy ** 2 <= 81                                    # radius about 4 km
    belt = ndi.binary_fill_holes(ndi.binary_closing(np.pad(ridge, 12), structure=disk))[12:-12, 12:-12]
    blab, bn = ndi.label(belt)
    bsizes = ndi.sum(belt, blab, range(1, bn + 1))
    order = np.argsort(bsizes)[::-1]
    keep = [i + 1 for i in order if bsizes[i] >= 150]                  # about 30 km² and more
    belts = np.zeros_like(blab)
    for k, i in enumerate(keep, 1):
        belts[blab == i] = k
    return xc, yc, z, ridge & (belts > 0), belts > 0, belts


def draw_schematic(ax, labels=True):
    """The schematic geological map: granite and gneiss, the schist belts and their ridges."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    xc, yc, z, ridge, belt, belts = schematic()
    units = np.where(ridge, 2, np.where(belt, 1, 0)).astype(float)
    cmap = ListedColormap(["#f3d9d0", "#a8d5a2", "#3f6e3a"])
    ax.pcolormesh(xc, yc, units, cmap=cmap, vmin=-0.5, vmax=2.5, shading="nearest", rasterized=True)
    ax.contour(xc, yc, belt.astype(float), levels=[0.5], colors="#2d4f2a", linewidths=0.9)
    if labels:
        import matplotlib.patheffects as pe
        halo = [pe.withStroke(linewidth=2.5, foreground="white")]
        for text, x, y, rot in (("western (Sandur) range", 650.8, 1669.0, -52), ("eastern range", 684.5, 1666.0, -52),
                                ("central metavolcanic\nterrane", 659.8, 1676.0, -52),
                                ("north-eastern belt", 692.5, 1675.5, -30),
                                ("granite and gneiss", 652.0, 1645.0, 0), ("granite and gneiss", 694.0, 1690.0, 0)):
            ax.text(x, y, text, rotation=rot, ha="center", va="center", fontsize=7, color="#1f3a1d",
                    style="italic", path_effects=halo, zorder=5)
    ax.set_aspect("equal")
    return [Patch(fc="#3f6e3a", label="ridges held up by iron formation (DEM)"),
            Patch(fc="#a8d5a2", ec="#2d4f2a", label="Sandur schist belt (envelope of the ridges)"),
            Patch(fc="#f3d9d0", label="granite and gneiss")]


def outline(ax, color="k", lw=0.9, ridges=True):
    """The schematic belt outline (and ridges) over another map, for comparison."""
    xc, yc, _, ridge, belt, _ = schematic()
    ax.contour(xc, yc, belt.astype(float), levels=[0.5], colors=color, linewidths=lw, linestyles="-")
    if ridges:
        ax.contour(xc, yc, ridge.astype(float), levels=[0.5], colors=color, linewidths=lw * 0.6, linestyles=":")


def belt_stats():
    """Length along strike, the largest width across it and the strike of the main belt (km, degrees)."""
    xc, yc, _, _, _, belts = schematic()
    X, Y = np.meshgrid(xc, yc)
    pts = np.c_[X[belts == 1], Y[belts == 1]]
    c = pts - pts.mean(axis=0)
    w, v = np.linalg.eigh(np.cov(c.T))
    along, across = v[:, 1], v[:, 0]
    s, t = c @ along, c @ across
    bins = np.arange(s.min(), s.max() + 1.0, 1.0)
    width = max((np.ptp(t[(s >= a) & (s < a + 1.0)]) if ((s >= a) & (s < a + 1.0)).sum() > 1 else 0.0) for a in bins)
    strike = (np.degrees(np.arctan2(along[0], along[1])) + 360) % 180
    return {"length_km": float(np.ptp(s)), "width_km": float(width), "strike_deg": float(strike),
            "area_km2": float(len(pts) * 0.45 ** 2)}


# ---------------------------------------------------------------- distances to the models

def distance_table(x_km, y_km, value, threshold, edge_km=5.0):
    """For each locality: the distance (km) to the nearest cell with ``value`` ≥ threshold, and the
    share of the area (an edge band of ``edge_km`` left out) that lies at least as close to one.
    ``value`` is a 2-D array on the cell centres ``x_km`` (columns), ``y_km`` (rows)."""
    from scipy import ndimage as ndi
    x_km, y_km, value = np.asarray(x_km), np.asarray(y_km), np.asarray(value)
    dx = float(np.median(np.diff(x_km)))
    dist = ndi.distance_transform_edt(~(value >= threshold)) * dx
    inner = ((x_km[None, :] >= x_km.min() + edge_km) & (x_km[None, :] <= x_km.max() - edge_km)
             & (y_km[:, None] >= y_km.min() + edge_km) & (y_km[:, None] <= y_km.max() - edge_km))
    rows = []
    for name, x, y, kind in LOCALITIES:
        i, j = int(np.argmin(np.abs(y_km - y))), int(np.argmin(np.abs(x_km - x)))
        d = float(dist[i, j])
        rows.append({"name": name, "kind": kind, "x": x, "y": y, "distance_km": d,
                     "area_as_close": float((dist[inner] <= d + 1e-9).mean())})
    return rows
