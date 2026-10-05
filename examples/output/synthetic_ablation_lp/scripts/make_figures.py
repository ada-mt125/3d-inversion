"""Figures and figures/numbers.json of the Lp ablation report.

    py examples/output/synthetic_ablation_lp/scripts/make_figures.py

Reads the inputs (``inputs/``, from ``make_synthetic.py``) and the 200 results
(``data/runs/<run>/``, from ``deploy/ec2_sweep.py``).  The measures are in ``metrics.py``.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

import metrics as mt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))      # the repository
from geoinv3d.viz.result_workflow import load_result, result_mesh  # noqa: E402
from geoinv3d.viz.sections import ResultModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures"
# the results: data/runs/ (or a folder given as the first argument, e.g. a part of the runs)
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "runs"
RUNS_JSON = json.loads((ROOT / "inputs" / "runs.json").read_text())
plt.rcParams.update({"font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 9,
                     "axes.titlesize": 9.5, "axes.unicode_minus": False, "savefig.dpi": 140})

GROUPS = {"A": "A: no prior", "Bw1": "B: reference, weight 1", "Bw10": "B: reference, weight 10",
          "Bw100": "B: reference, weight 100", "C": "C: boreholes, 150 m reach"}
GTICK = {"A": "A\nno prior", "Bw1": "B\nweight 1", "Bw10": "B\nweight 10", "Bw100": "B\nweight 100",
         "C": "C\nboreholes"}
GCOL = {"A": "#8a8f98", "Bw1": "#c4b0ff", "Bw10": "#7b4cff", "Bw100": "#3d1d99", "C": "#e07b39"}
NORMS = {"n0000": "(0,0,0,0)", "n0111": "(0,1,1,1)", "n0221": "(0,2,2,1)", "n0222": "(0,2,2,2)", "n1111": "(1,1,1,1)"}
BETAS = {"b10": "1", "b15": "1.5", "b20": "2", "b30": "3"}
LENGTHS = {"L1": "1", "L3": "3"}
SETTINGS = [f"{n}_{b}_{ell}" for n in NORMS for b in BETAS for ell in LENGTHS]
PURPLE = "#7b4cff"
CMAP = plt.get_cmap("RdYlBu_r").copy()
CMAP.set_bad("white")
VMAX = 1.0
YSEC = mt.CUBE["centre"][1]
X0, Y0 = 600000.0, 1600000.0
km = lambda v: (np.asarray(v) - X0) / 1000      # noqa: E731
kmy = lambda v: (np.asarray(v) - Y0) / 1000     # noqa: E731


def val(N, key, field="score"):
    """A run's measure, NaN for a run that is missing."""
    r = N["runs"].get(key)
    return float("nan") if r is None or r.get(field) is None else r[field]


def label(setting):
    n, b, ell = setting.split("_")
    return f"p={NORMS[n]}, β={BETAS[b]}, L={LENGTHS[ell]}"


def load(key):
    d = RUNS / key
    meta = load_result(d)
    with zipfile.ZipFile(d / "result.zip") as z:
        if "reference_model.npy" in z.namelist():
            meta["_reference"] = np.load(io.BytesIO(z.read("reference_model.npy")))
    return meta


# ── geometry for the drawings ──────────────────────────────────────────

def cube_box_section(ax, **kw):
    c, e = mt.CUBE, mt.CUBE["edge"] / 2
    ax.add_patch(Rectangle((km(c["centre"][0] - e), c["bottom_elev"]), e * 2 / 1000, c["edge"], fill=False,
                           **({"ec": "k", "lw": 0.8} | kw)))


def dyke_section(ax, d=mt.DYKE, **kw):
    hw = d["thickness"] / np.sin(np.radians(d["dip"])) / 2
    dd = np.linspace(d["top_depth"], d["bottom_depth"], 40)
    xc = d["top_x"] + (dd - d["top_depth"]) / np.tan(np.radians(d["dip"]))
    for sgn in (-1, 1):
        x = xc + sgn * hw
        ax.plot(km(x), mt.ground(x, np.full_like(x, YSEC)) - dd, **kw)
    for i in (0, -1):
        x = np.array([xc[i] - hw, xc[i] + hw])
        ax.plot(km(x), mt.ground(x, np.full_like(x, YSEC)) - dd[i], **kw)


def prior_section(ax):
    s0, s1 = mt.SPEC_B["sources"]
    p = np.array(s0["polygon"])
    kw = {"color": PURPLE, "ls": "--", "lw": 1.0}
    if p[:, 1].min() <= YSEC <= p[:, 1].max():
        x = np.linspace(p[:, 0].min(), p[:, 0].max(), 20)
        g = mt.ground(x, np.full_like(x, YSEC))
        ax.plot(km(x), g - s0["top_m"], **kw)
        ax.plot(km(x), g - s0["bottom_m"], **kw)
        for xe in (x[0], x[-1]):
            ge = float(mt.ground(xe, YSEC))
            ax.plot(km([xe, xe]), [ge - s0["top_m"], ge - s0["bottom_m"]], **kw)
    q = np.array(s1["polygon"])
    hw = (q[:, 0].max() - q[:, 0].min()) / 2
    dyke_section(ax, {"top_x": q[:, 0].mean(), "top_depth": s1["top_m"], "bottom_depth": s1["bottom_m"],
                      "dip": s1["dip"], "thickness": 2 * hw * np.sin(np.radians(s1["dip"]))}, **kw)


def holes_section(ax, traces):
    for name, (s, x, y, z, k) in traces.items():
        if abs(y[0] - YSEC) >= 100:
            continue
        ax.plot(km(x), z, "-", color="#222", lw=0.8)
        hit = k > 0
        ax.plot(km(np.where(hit, x, np.nan)), np.where(hit, z, np.nan), "-", color="#111", lw=3.0)
        ax.text(km(x[0]), z[0] + 30, name.split()[0], ha={"H1": "right", "H2": "left"}.get(name[:2], "center"),
                va="bottom", fontsize=7)


def section_points(nx=401, nz=201, top=800.0, bottom=-400.0):
    xs = np.linspace(X0, X0 + 5000, nx)
    zs = np.linspace(top, bottom, nz)
    X, Z = np.meshgrid(xs, zs)
    pts = np.column_stack([X.ravel(), np.full(X.size, YSEC), Z.ravel()])
    above = Z > mt.ground(xs, np.full_like(xs, YSEC))[None, :]
    return xs, zs, X.shape, pts, above


# ── figures ────────────────────────────────────────────────────────────

def fig_setup(traces):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.55]})
    ax = axes[0]
    xs = np.linspace(X0, X0 + 5000, 201)
    X, Y = np.meshgrid(xs, xs + (Y0 - X0))
    gz = mt.ground(X, Y)
    im = ax.imshow(gz, extent=[0, 5, 0, 5], origin="lower", cmap="gist_earth", vmin=420, vmax=820)
    ax.contour(km(X), kmy(Y), gz, levels=np.arange(500, 800, 50), colors="k", linewidths=0.3, alpha=0.5)
    c, e = mt.CUBE, mt.CUBE["edge"] / 2
    ax.add_patch(Rectangle((km(c["centre"][0] - e), kmy(c["centre"][1] - e)), 0.3, 0.3, fc="#d7301f", ec="k", lw=0.8))
    d = mt.DYKE
    hw = d["thickness"] / np.sin(np.radians(d["dip"])) / 2
    reach = (d["bottom_depth"] - d["top_depth"]) / np.tan(np.radians(d["dip"]))
    ax.add_patch(Rectangle((km(d["top_x"] - hw), kmy(d["y"][0])), (reach + 2 * hw) / 1000, (d["y"][1] - d["y"][0]) / 1000,
                           fc="#d7301f", alpha=0.3, ec="#d7301f", lw=0.8))
    s0, s1 = mt.SPEC_B["sources"]
    p = np.array(s0["polygon"])
    ax.add_patch(Rectangle((km(p[:, 0].min()), kmy(p[:, 1].min())), np.ptp(p[:, 0]) / 1000, np.ptp(p[:, 1]) / 1000,
                           fill=False, ec=PURPLE, ls="--", lw=1.1))
    q = np.array(s1["polygon"])
    reach_b = (s1["bottom_m"] - s1["top_m"]) / np.tan(np.radians(s1["dip"]))
    ax.add_patch(Rectangle((km(q[:, 0].min()), kmy(q[:, 1].min())), (np.ptp(q[:, 0]) + reach_b) / 1000,
                           np.ptp(q[:, 1]) / 1000, fill=False, ec=PURPLE, ls="--", lw=1.1))
    for name, (s, x, y, z, k) in traces.items():
        ax.plot(km(x), kmy(y), "-", color="k", lw=1.2)
        ax.plot(km(x[0]), kmy(y[0]), "^", color="k", ms=5)
        for xx, yy in zip(km(x[::40]), kmy(y[::40])):
            ax.add_patch(Circle((xx, yy), 0.15, fill=False, ec="#e07b39", lw=0.5, alpha=0.7))
        ax.annotate(name.split()[0], (km(x[0]), kmy(y[0])), xytext=(3, 4), textcoords="offset points", fontsize=7)
    ax.axhline(kmy(YSEC), color="k", ls=":", lw=0.8)
    ax.set(xlim=(0, 5), ylim=(0, 5), xlabel="easting from 600 km (km)", ylabel="northing from 1600 km (km)",
           title="(a) ground, bodies (red), B's reference (dashed), holes (▲, 150 m circles)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="ground elevation (m)")
    ax = axes[1]
    raw = np.loadtxt(mt.INPUTS / "synthetic_tmi_80m.csv", delimiter=",", skiprows=1)
    n = int(round(np.sqrt(len(raw))))
    tmi = raw[:, 3].reshape(n, n)
    v = np.percentile(np.abs(tmi), 99)
    im = ax.imshow(tmi, extent=[0, 5, 0, 5], origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.axhline(kmy(YSEC), color="k", ls=":", lw=0.8)
    ax.set(xlabel="easting (km)", title=f"(b) total-field anomaly 80 m above the ground, {len(raw):,} points")
    fig.colorbar(im, ax=ax, shrink=0.8, label="ΔT (nT)")
    ax = axes[2]
    xs, zs, shape, pts, above = section_points()
    t, *_ = mt.truth_at(pts)
    t = np.where(above, np.nan, t.reshape(shape))
    im = ax.imshow(t, extent=[0, 5, zs[-1], zs[0]], aspect="auto", cmap=CMAP, vmin=0, vmax=1.0,
                   interpolation="nearest")
    ax.plot(km(xs), mt.ground(xs, np.full_like(xs, YSEC)), "k-", lw=0.8)
    prior_section(ax)
    holes_section(ax, traces)
    ax.set(xlabel="easting (km)", ylabel="elevation (m)",
           title=f"(c) true model on the E–W section at northing {kmy(YSEC):.1f} km: cube 0.5 SI, intrusion 1 SI")
    fig.colorbar(im, ax=ax, shrink=0.8, label="susceptibility (SI)")
    fig.tight_layout()
    fig.savefig(FIGS / "setup.png", bbox_inches="tight")
    plt.close(fig)


def fig_tuning(N):
    """Heat maps of the score: norms × (β, L), one panel per group."""
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(17, 3.6), sharey=True)
    cols = [(b, ell) for b in BETAS for ell in LENGTHS]
    for ax, g in zip(axes, [g for g in GROUPS if g in N["best"]]):
        S = np.array([[val(N, f"{g}_{n}_{b}_{ell}") for b, ell in cols] for n in NORMS])
        im = ax.imshow(S, cmap="viridis", vmin=0, vmax=0.8, aspect="auto")
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                r = N["runs"].get(f"{g}_{list(NORMS)[i]}_{cols[j][0]}_{cols[j][1]}")
                if r is None:
                    continue
                ax.text(j, i, f"{S[i, j]:.2f}" + ("" if r["status"] == "converged" else "*"), ha="center",
                        va="center", fontsize=6.5, color="w" if S[i, j] < 0.5 else "k")
        i, j = np.unravel_index(np.nanargmax(S), S.shape)
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#ff3b30", lw=2))
        ax.set_xticks(range(len(cols)), [f"β{BETAS[b]}\nL{LENGTHS[ell]}" for b, ell in cols], fontsize=7)
        ax.set_yticks(range(len(NORMS)), list(NORMS.values()))
        ax.set_title(GROUPS[g], loc="left")
    axes[0].set_ylabel("norms (p_s, p_x, p_y, p_z)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="score: volume-matched overlap, mean of the two bodies", pad=0.01)
    fig.savefig(FIGS / "tuning.png", bbox_inches="tight")
    plt.close(fig)


def fig_effects(N):
    """Mean score by each parameter, per group."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
    for ax, (dim, values, title) in zip(axes, [(0, NORMS, "norms"), (1, BETAS, "depth weighting β"),
                                               (2, LENGTHS, "length scale L")]):
        for g in [g for g in GROUPS if g in N["best"]]:
            means = [np.nanmean([val(N, f"{g}_{s}") for s in SETTINGS if s.split("_")[dim] == v]) for v in values]
            best = [np.nanmax([val(N, f"{g}_{s}") for s in SETTINGS if s.split("_")[dim] == v]) for v in values]
            ax.plot(range(len(values)), means, "o-", color=GCOL[g], label=GROUPS[g], lw=1.4)
            ax.plot(range(len(values)), best, "v:", color=GCOL[g], lw=0.9, ms=4)
        ax.set_xticks(range(len(values)), list(values.values()))
        ax.set_title(f"by {title}: mean (solid), best (dotted)", loc="left")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("score")
    axes[-1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "effects.png", bbox_inches="tight")
    plt.close(fig)


def fig_paired(N):
    """Each setting: its score with the prior against without."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, key, title in ((axes[0], "overlap_cube", "cube"), (axes[1], "overlap_dyke", "intrusion")):
        a = np.array([val(N, f"A_{s}", key) for s in SETTINGS])
        for g in [g for g in GROUPS if g != "A" and g in N["best"]]:
            b = np.array([val(N, f"{g}_{s}", key) for s in SETTINGS])
            ax.scatter(a, b, s=14, color=GCOL[g], label=GROUPS[g], alpha=0.85)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="overlap without a prior (A)", ylabel="overlap with the prior",
               title=f"{title}: the same 40 settings with and without a prior")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "paired.png", bbox_inches="tight")
    plt.close(fig)


def fig_sections(models, N, traces, rows):
    xs, zs, shape, pts, above = section_points()
    t, *_ = mt.truth_at(pts)
    t = np.where(above, np.nan, t.reshape(shape))
    fig, axes = plt.subplots(len(rows) + 1, 1, figsize=(10.5, 1.9 * (len(rows) + 1) + 0.4), sharex=True,
                             gridspec_kw={"hspace": 0.35})
    ext = [0, 5, zs[-1], zs[0]]
    gline = mt.ground(xs, np.full_like(xs, YSEC))

    def draw(ax, v, title, g):
        im = ax.imshow(v, extent=ext, aspect="auto", cmap=CMAP, vmin=0, vmax=VMAX, interpolation="nearest")
        ax.plot(km(xs), gline, "k-", lw=0.6)
        cube_box_section(ax)
        dyke_section(ax, color="k", lw=0.6)
        if g.startswith("B"):
            prior_section(ax)
        if g == "C":
            holes_section(ax, traces)
        ax.set_title(title, loc="left")
        ax.set_ylabel("elevation (m)")
        return im

    im = draw(axes[0], t, "true model (dashed: B's reference model; black: C's holes)", "")
    prior_section(axes[0])
    holes_section(axes[0], traces)
    for ax, (key, title) in zip(axes[1:], rows):
        r = N["runs"][key]
        v = np.where(above, np.nan, ResultModel(models[key]).sample(pts).reshape(shape))
        draw(ax, v, f"{title} — {label(key.split('_', 1)[1])}: score {r['score']:.2f}, χ²/N {r['chi2_per_datum']:.2f}",
             key.split("_")[0])
    axes[-1].set_xlabel("easting from 600 km (km)")
    fig.colorbar(im, ax=axes, shrink=0.4, extend="max", label="susceptibility (SI)", pad=0.01)
    return fig


def fig_metrics(N, best):
    panels = [("overlap_cube", "cube overlap"), ("overlap_dyke", "intrusion overlap"),
              ("cube_depth", "depth of the cube's magnetization (m)"), ("dyke_base", "90 % depth of the intrusion's magnetization (m)"),
              ("dip", "apparent dip of the intrusion (°)"), ("k_dyke", "mean κ in the intrusion (SI)"),
              ("k_cube", "mean κ in the cube (SI)"), ("k_wrong", "mean κ where B's reference is wrong (SI)")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
    for ax, (key, title) in zip(axes.ravel(), panels):
        truth = N["truth_measures"][key]          # the true model, measured the same way
        gs = [g for g in GROUPS if g in best]
        vals = [N["runs"][best[g]][key] for g in gs]
        ax.bar(range(len(gs)), vals, color=[GCOL[g] for g in gs])
        ax.axhline(truth, color="k", ls="--", lw=1)
        ax.set_xticks(range(len(gs)), [GTICK[g] for g in gs], fontsize=7)
        ax.set_title(title, loc="left")
        if key in ("cube_depth", "dyke_base"):
            ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGS / "best_metrics.png", bbox_inches="tight")
    plt.close(fig)


def fig_boreholes(models, N, traces, setting):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.3, 1, 1, 1]})
    ax = axes[0]
    labels = ["0–50", "50–100", "100–150", "150–250", "250–500", ">500"]
    for s in N["boreholes"]["c_minus_a"]:
        ax.semilogy(range(len(labels)), N["boreholes"]["c_minus_a"][s], "-", color="#e07b39", lw=0.6, alpha=0.5)
    ax.semilogy(range(len(labels)), N["boreholes"]["c_minus_a"][setting], "o-", color="#b5501a", lw=2,
                label=f"C's best: {label(setting)}")
    ax.set_xticks(range(len(labels)), labels)
    ax.axvspan(2.5, 5.5, color="#eee", zorder=-1)
    ax.set(xlabel="distance from the nearest hole (m)", ylabel="mean |C − A| (SI)",
           title="(a) reach of the holes (40 settings)")
    ax.legend(fontsize=7, frameon=False)
    picks = [("H3 intrusion, inclined", "(b) H3, inclined through the intrusion"),
             ("H5 intrusion, deep", "(c) H5, the intrusion's deeper part"),
             ("V", f"(d) test hole V, {N['boreholes']['virtual']['nearest_hole_m']:.0f} m from any hole (not used)")]
    for ax, (name, title) in zip(axes[1:], picks):
        if name == "V":
            pts, depth, k = virtual_trace()
        else:
            s, x, y, z, k = traces[name]
            pts = np.column_stack([x, y, z])
            depth = mt.ground(x[0], y[0]) - z
        ax.plot(k, depth, "k-", lw=2.0, label="log (true)")
        top = 1.1
        for g in ("A", "C"):
            v = ResultModel(models[f"{g}_{setting}"]).sample(pts)
            top = max(top, np.nanmax(v) * 1.05)
            ax.plot(v, depth, "-", color=GCOL[g], lw=1.4, label=GROUPS[g])
        ax.set(ylim=(depth.max(), 0), xlim=(-0.03, top), xlabel="susceptibility (SI)", ylabel="depth (m)", title=title)
    axes[1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "boreholes.png", bbox_inches="tight")
    plt.close(fig)


# ── the boreholes across the settings ─────────────────────────────────

# settings shown side by side for A and C: C's best, others across the grid, A's best, the page's
# default and C's worst against A
C_ROWS = [("n0000_b20_L3", "C's best"), ("n0111_b15_L3", ""), ("n1111_b15_L1", "A's best"),
          ("n1111_b30_L1", "β = 3"), ("n0221_b15_L1", "the page's default"), ("n0222_b15_L1", "A's near-worst"),
          ("n0221_b20_L3", "C's worst against A")]
ROBUST = ["score", "overlap_cube", "overlap_dyke", "cube_depth", "dyke_base", "dip", "k_dyke", "k_cube"]


def fig_robust_scores(N):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1, 1.25]})
    ax = axes[0]
    gs = list(GROUPS)
    data = [[val(N, f"{g}_{s}") for s in SETTINGS] for g in gs]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops={"color": "k"})
    for patch, g in zip(bp["boxes"], gs):
        patch.set(facecolor=GCOL[g], alpha=0.3, edgecolor=GCOL[g])
    rng = np.random.default_rng(0)
    for i, (g, d) in enumerate(zip(gs, data), start=1):
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(d)), d, s=11, color=GCOL[g], zorder=3)
    ax.set_xticks(range(1, len(gs) + 1), [GTICK[g] for g in gs], fontsize=7.5)
    ax.set(ylim=(0, 1), ylabel="score", title="(a) all 40 settings in each group")
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    marker = {"n0000": "o", "n0111": "s", "n0221": "^", "n0222": "v", "n1111": "D"}
    for st in SETTINGS:
        n, b, ell = st.split("_")
        ax.scatter(val(N, f"A_{st}"), val(N, f"C_{st}"), marker=marker[n], s=30, lw=1, edgecolor=GCOL["C"],
                   facecolor=GCOL["C"] if ell == "L3" else "white", zorder=3)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="", marker=m, color=GCOL["C"], mfc="white", label=f"p = {NORMS[n]}")
               for n, m in marker.items()]
    handles += [Line2D([], [], ls="", marker="o", color=GCOL["C"], label="filled: L = 3, open: L = 1")]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="lower right")
    better = N["group_stats"]["C"]["better_than_A"]
    ax.set(xlim=(0.2, 0.92), ylim=(0.2, 0.92), xlabel="score without a prior (A)", ylabel="score with the holes (C)",
           title=f"(b) each setting with and without the holes: C higher in {better} of 40")
    ax.grid(alpha=0.3)
    ax = axes[2]
    cols = [(b, ell) for b in BETAS for ell in LENGTHS]
    D = np.array([[val(N, f"C_{n}_{b}_{ell}") - val(N, f"A_{n}_{b}_{ell}") for b, ell in cols] for n in NORMS])
    im = ax.imshow(D, cmap="RdBu", vmin=-0.4, vmax=0.4, aspect="auto")
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            ax.text(j, i, f"{D[i, j]:+.2f}", ha="center", va="center", fontsize=7, color="k")
    ax.set_xticks(range(len(cols)), [f"β{BETAS[b]}\nL{LENGTHS[ell]}" for b, ell in cols], fontsize=7)
    ax.set_yticks(range(len(NORMS)), list(NORMS.values()))
    ax.set_title("(c) gain from the holes (C − A score), per setting", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8, label="C − A")
    fig.tight_layout()
    fig.savefig(FIGS / "c_scores.png", bbox_inches="tight")
    plt.close(fig)


def fig_robust_measures(N):
    panels = [("overlap_cube", "cube overlap"), ("overlap_dyke", "intrusion overlap"),
              ("cube_depth", "depth of the cube's magnetization (m)"), ("dyke_base", "90 % depth of the intrusion's magnetization (m)"),
              ("dip", "apparent dip of the intrusion (°)"), ("k_dyke", "mean κ in the intrusion (SI)")]
    fig, axes = plt.subplots(1, 6, figsize=(17, 3.9))
    rng = np.random.default_rng(1)
    for ax, (key, title) in zip(axes, panels):
        data = [[val(N, f"{g}_{s}", key) for s in SETTINGS] for g in ("A", "C")]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops={"color": "k"})
        for patch, g in zip(bp["boxes"], ("A", "C")):
            patch.set(facecolor=GCOL[g], alpha=0.3, edgecolor=GCOL[g])
        for i, (g, d) in enumerate(zip(("A", "C"), data), start=1):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=9, color=GCOL[g], zorder=3)
        ax.axhline(N["truth_measures"][key], color="k", ls="--", lw=1)
        ax.set_xticks([1, 2], ["A\nno prior", "C\nboreholes"], fontsize=7.5)
        ax.set_title(title, loc="left", fontsize=8.5)
        if key in ("cube_depth", "dyke_base"):
            ax.invert_yaxis()
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "c_measures.png", bbox_inches="tight")
    plt.close(fig)


def fig_c_sections(models, N, traces):
    xs, zs, shape, pts, above = section_points()
    fig, axes = plt.subplots(len(C_ROWS), 2, figsize=(15.5, 1.85 * len(C_ROWS) + 0.5), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.42, "wspace": 0.05})
    ext = [0, 5, zs[-1], zs[0]]
    gline = mt.ground(xs, np.full_like(xs, YSEC))
    for i, (st, tag) in enumerate(C_ROWS):
        for j, g in enumerate(("A", "C")):
            ax = axes[i, j]
            r = N["runs"][f"{g}_{st}"]
            v = np.where(above, np.nan, ResultModel(models[f"{g}_{st}"]).sample(pts).reshape(shape))
            im = ax.imshow(v, extent=ext, aspect="auto", cmap=CMAP, vmin=0, vmax=VMAX, interpolation="nearest")
            ax.plot(km(xs), gline, "k-", lw=0.6)
            cube_box_section(ax)
            dyke_section(ax, color="k", lw=0.6)
            if g == "C":
                holes_section(ax, traces)
            head = f"{'A, no prior' if g == 'A' else 'C, boreholes'} — {label(st)}"
            ax.set_title(f"{head}{f'  ({tag})' if tag else ''}: score {r['score']:.2f}", loc="left", fontsize=8.5)
        axes[i, 0].set_ylabel("elevation (m)")
    for ax in axes[-1]:
        ax.set_xlabel("easting from 600 km (km)")
    fig.colorbar(im, ax=axes, shrink=0.35, extend="max", label="susceptibility (SI)", pad=0.01)
    fig.savefig(FIGS / "c_sections.png", bbox_inches="tight")
    plt.close(fig)


VIRTUAL = (603200.0, 1602700.0, 800.0)         # a hole not given to group C, over 200 m from every hole


def virtual_trace(step=5.0):
    x, y, L = VIRTUAL
    d = np.arange(0.0, L + 1e-9, step)
    pts = np.column_stack([np.full_like(d, x), np.full_like(d, y), float(mt.ground(x, y)) - d])
    return pts, d, mt.truth_at(pts)[0]


# ── numbers ────────────────────────────────────────────────────────────

def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    traces = mt.hole_traces()
    keys = [k for k in RUNS_JSON if (RUNS / k / "result.zip").exists()]
    missing = sorted(set(RUNS_JSON) - set(keys))
    first = load(keys[0])
    cells = mt.Cells(result_mesh(first), first["_active"])
    N = {"truth": {"cube": mt.CUBE, "dyke": mt.DYKE, **{k: mt.T[k] for k in (
             "field", "noise", "station_height", "station_spacing", "n_stations", "tmi_range", "ground_range",
             "seed", "true_cells", "true_volume_m3")}},
         "mesh": {**{k: first["mesh_design"]["used"].get(k) for k in
                     ("core_cell_m", "core_cell_z_m", "depth_core_m", "pad_distance_m", "octree_levels")},
                  "n_active": int(first["n_active_cells"]), "n_data": int(first["n_data"])},
         "runs": {}, "missing": missing}
    N["truth_measures"] = mt.measures(cells.t, cells)
    models = {}
    for i, key in enumerate(keys):
        meta = load(key)
        assert np.array_equal(meta["_active"], first["_active"])
        m = np.asarray(meta["_model"], float)
        conv = meta.get("convergence") or {}
        d = meta["_data"]
        info = RUNS_JSON[key]
        minutes = (RUNS / key / "minutes").read_text().strip() if (RUNS / key / "minutes").exists() else None
        N["runs"][key] = {"group": key.split("_")[0], "setting": key.split("_", 1)[1], "norms": info["norms"],
                          "beta": info["beta"], "length": info["length"], "weight": info["weight"],
                          "chi2_per_datum": conv.get("chi2_per_datum"), "status": conv.get("status"),
                          "iterations": meta.get("n_iterations"), "minutes": float(minutes) if minutes else None,
                          "rms_nT": float(np.sqrt(np.mean((d["observed"] - d["predicted"]) ** 2))),
                          **mt.measures(m, cells)}
        models[key] = meta
        if i % 20 == 0:
            print(f"{i}/{len(keys)} {key} score {N['runs'][key]['score']:.2f}", flush=True)
    present = [g for g in GROUPS if any(k.split("_")[0] == g for k in models)]
    for g in [g for g in present if g != "A"]:
        geo = models[next(k for k in models if k.split("_")[0] == g)]["geology"]
        N.setdefault("geology", {})[g] = {"n_constrained": geo["n_constrained"], "n_touched": geo.get("n_touched"),
                                          "units": {u["name"]: {k: u.get(k) for k in ("value", "lower", "upper", "weight", "n_cells")}
                                                    for u in geo["units"]}}
    best = {g: max((f"{g}_{s}" for s in SETTINGS if f"{g}_{s}" in models), key=lambda k: N["runs"][k]["score"])
            for g in present}
    N["best"] = best
    N["effects"] = {g: {dim: {v: float(np.nanmean([val(N, f"{g}_{s}") for s in SETTINGS
                                                if s.split("_")[i] == v])) for v in vals}
                        for i, (dim, vals) in enumerate([("norms", NORMS), ("beta", BETAS), ("length", LENGTHS)])}
                    for g in present}
    N["group_stats"] = {g: {"n": int(sum(f"{g}_{s}" in models for s in SETTINGS)),
                            "mean": float(np.nanmean([val(N, f"{g}_{s}") for s in SETTINGS])),
                            "median": float(np.nanmedian([val(N, f"{g}_{s}") for s in SETTINGS])),
                            "converged": int(sum(val(N, f"{g}_{s}", "status") == "converged" for s in SETTINGS)),
                            "better_than_A": int(sum(val(N, f"{g}_{s}") > val(N, f"A_{s}") for s in SETTINGS))}
                        for g in present}
    # how firmly B's reference held its cells: the model where the reference is a unit's full value
    N["pinned"] = {}
    for g in [g for g in ("Bw1", "Bw10", "Bw100") if g in best]:
        meta = models[best[g]]
        for v in (0.75, 0.5):
            s = np.isclose(meta["_reference"], v, atol=1e-6)
            N["pinned"].setdefault(g, {})[str(v)] = {"n": int(s.sum()), "mean": float(np.mean(meta["_model"][s]))}
    # the boreholes: C against A for each setting, by the distance from the holes
    b = N["boreholes"] = {"bands": [[lo, None if not np.isfinite(hi) else hi] for lo, hi in mt.BANDS],
                          "c_minus_a": {}, "cells_within": {str(r): int((cells.core & (cells.hole_dist <= r)).sum())
                                                            for r in (50, 150, 250)},
                          "core_cells": int(cells.core.sum())}
    for s in [s for s in SETTINGS if f"A_{s}" in models and f"C_{s}" in models]:
        ma, mc = (np.asarray(models[f"{g}_{s}"]["_model"], float) for g in ("A", "C"))
        b["c_minus_a"][s] = mt.by_distance(np.abs(mc - ma), cells)
    pts, depth, k = virtual_trace()
    hit = k > 0
    b["virtual"] = {"xy": list(VIRTUAL[:2]), "hit_m": [float(depth[hit].min()), float(depth[hit].max())],
                    "nearest_hole_m": float(min(np.hypot(t[1] - VIRTUAL[0], t[2] - VIRTUAL[1]).min()
                                                for t in traces.values()))}
    # C's best setting that A also has (all of them, once every run is in)
    sbest = max((s for s in SETTINGS if f"C_{s}" in models and f"A_{s}" in models), key=lambda s: N["runs"][f"C_{s}"]["score"])
    for g in ("A", "C"):
        v = ResultModel(models[f"{g}_{sbest}"]).sample(pts)
        b["virtual"][g] = float(np.nanmean(v[hit]))
    N["robust"] = {g: {m: [float(np.nanpercentile([val(N, f"{g}_{s}", m) for s in SETTINGS], q)) for q in (0, 25, 50, 75, 100)]
                       for m in ROBUST} for g in present}
    tm = N["truth_measures"]
    N["near_truth"] = {g: {"dip_10": int(sum(abs(val(N, f"{g}_{s}", "dip") - tm["dip"]) <= 10 for s in SETTINGS)),
                           "cube_depth_50": int(sum(abs(val(N, f"{g}_{s}", "cube_depth") - tm["cube_depth"]) <= 50 for s in SETTINGS)),
                           "dyke_base_100": int(sum(abs(val(N, f"{g}_{s}", "dyke_base") - tm["dyke_base"]) <= 100 for s in SETTINGS)),
                           "score_07": int(sum(val(N, f"{g}_{s}") >= 0.7 for s in SETTINGS))} for g in present}
    (FIGS / "numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False, default=float), encoding="utf-8")

    fig_setup(traces)
    fig_tuning(N)
    fig_effects(N)
    fig_paired(N)
    fig_metrics(N, best)
    rows = [(best[g], f"{GROUPS[g]}, best") for g in GROUPS if g in best]
    fig = fig_sections(models, N, traces, rows)
    fig.savefig(FIGS / "best_sections.png", bbox_inches="tight")
    plt.close(fig)
    sa = best["A"].split("_", 1)[1]
    fig = fig_sections(models, N, traces, [(f"{g}_{sa}", GROUPS[g]) for g in GROUPS if f"{g}_{sa}" in models])
    fig.savefig(FIGS / "same_setting_sections.png", bbox_inches="tight")
    plt.close(fig)
    fig_boreholes(models, N, traces, sbest)
    if all(f"{g}_{s}" in models for g in ("A", "C") for s in SETTINGS):
        fig_robust_scores(N)
        fig_robust_measures(N)
        fig_c_sections(models, N, traces)
    # the runs the report shows, kept in the repository (data/runs/ is not)
    import shutil
    shown = set(best.values()) | {f"{g}_{sa}" for g in GROUPS if f"{g}_{sa}" in models} | {f"A_{sbest}", f"C_{sbest}"}
    for key in sorted(shown) if len(sys.argv) == 1 else []:
        dest = ROOT / "data" / "best" / key
        dest.mkdir(parents=True, exist_ok=True)
        for f in ("result.zip", "minutes"):
            if (RUNS / key / f).exists():
                shutil.copy2(RUNS / key / f, dest / f)
    N["shown"] = sorted(shown)
    (FIGS / "numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    for g, k in best.items():
        r = N["runs"][k]
        print(f"best {g:6s} {label(k.split('_', 1)[1]):32s} score {r['score']:.2f} (cube {r['overlap_cube']:.2f}, "
              f"dyke {r['overlap_dyke']:.2f}) χ²/N {r['chi2_per_datum']:.2f} depth {r['cube_depth']:.0f} "
              f"base {r['dyke_base']:.0f} dip {r['dip']:.0f} κ {r['k_cube']:.2f}/{r['k_dyke']:.2f} wrong {r['k_wrong']:.3f}")
    print("missing:", missing)


if __name__ == "__main__":
    main()
