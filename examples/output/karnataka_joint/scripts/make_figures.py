"""Figures and numbers of the Karnataka joint gravity-magnetic report.

    py examples/output/karnataka_joint/scripts/make_figures.py [--set v2]

--set v2: the second series of runs (data/ec2_runs_fixed, data/lowres_runs_fixed; the group
lasso as group_lasso_depth), the magnetization-vector inversion of the magnetic data and the runs
with the rock-sample constraints (constraint_figures.py), into figures_v2/ (build_report_v2.py);
without it, the first series into figures/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, PowerNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "karnataka_inputs" / "shared"))
from kmodel import MAIN_HIGH, Grid, body, box_stats, data_fit, gridded, load  # noqa: E402

DATA, LOW, FIGS = ROOT / "data" / "ec2_runs", ROOT / "data" / "lowres_runs", ROOT / "figures"
SINGLE_G = ROOT.parent / "karnataka_gravity_terrain" / "data" / "ec2_runs" / "as1_beta1"
SINGLE_M = ROOT.parent / "karnataka_magnetic" / "data"
SINGLE_M = (SINGLE_M / "ec2_runs" if (SINGLE_M / "ec2_runs").exists() else SINGLE_M / "ec2_trials") / "beta1"
COUPLINGS = [
    # key, label, short label
    ("none", "No coupling", "none"),
    ("cross_gradient", "Cross-gradient", "cross-gradient"),
    ("joint_total_variation", "Joint total variation", "JTV"),
    ("linear_correspondence", "Linear correspondence", "linear"),
    ("pgi", "PGI (rock units)", "PGI"),
    ("group_lasso", "Group lasso", "group lasso"),
    ("group_lasso_uncoupled", "L1 + L2 by ADMM, no coupling", "L1+L2 ADMM"),
]
# the second series (build_report_v2.py)
COUPLINGS_V2 = [
    ("none", "No coupling", "none"),
    ("cross_gradient", "Cross-gradient", "cross-gradient"),
    ("joint_total_variation", "Joint total variation", "JTV"),
    ("linear_correspondence", "Linear correspondence", "linear"),
    ("group_lasso_depth", "Group lasso", "group lasso"),
    ("group_lasso_depth_uncoupled", "L1 + L2 by ADMM, no coupling", "L1+L2 ADMM"),
]
SANDUR_N = 1667500.0
BOX = (658000.0, 678000.0, 1657000.0, 1675000.0)
RHO = dict(cmap="RdBu_r", vmin=-0.3, vmax=0.3)
CHI_TICKS = [0, 0.05, 0.2, 0.5, 1.0]


def chi_style():
    return dict(cmap="Oranges", norm=PowerNorm(0.5, vmin=0.0, vmax=1.0))


def imshow(ax, g, x, y, **kw):
    if y[0] > y[-1]:
        g, y = g[::-1], y[::-1]
    dx, dy = (x[1] - x[0]) / 2, (y[1] - y[0]) / 2
    return ax.imshow(g, origin="lower", extent=(x[0] - dx, x[-1] + dx, y[0] - dy, y[-1] + dy), **kw)


def gradients(g):
    """Cell-centre gradients of a model in the core (air and the faces to it count as 0)."""
    m = np.where(g.active, g.m, 0.0)
    gx = np.gradient(m, g.mesh.cell_centers_x, axis=0)
    gy = np.gradient(m, g.mesh.cell_centers_y, axis=1)
    gz = np.gradient(m, g.mesh.cell_centers_z, axis=2)
    core = g.core_xy[:, :, None] & g.active & (g.zc >= g.z_core_base)[None, None, :]
    return np.stack([gx[core], gy[core], gz[core]], axis=1), core


def coupling_measures(gg, gm):
    """How alike the density and the susceptibility models are, in the core."""
    a, core = gradients(gg)
    b, _ = gradients(gm)
    na, nb = (a ** 2).sum(axis=1), (b ** 2).sum(axis=1)
    cross = (np.cross(a, b) ** 2).sum(axis=1)
    rho, chi = gg.m[core], gm.m[core]
    dense = np.abs(rho) > 0.1 * np.abs(rho).max()
    magnetic = np.abs(chi) > 0.1 * np.abs(chi).max()
    edge_r, edge_c = na > 0.01 * na.max(), nb > 0.01 * nb.max()
    return {
        # 0: the gradients are parallel wherever both exist; 1: perpendicular
        "cross_gradient": float(cross.sum() / max((na * nb).sum(), 1e-300)),
        # share of the susceptibility model's edges that lie on an edge of the density model
        "edges_shared": float((edge_r & edge_c).sum() / max(edge_c.sum(), 1)),
        "edge_cells": [int(edge_r.sum()), int(edge_c.sum())],
        "corr_cells": float(np.corrcoef(rho, chi)[0, 1]),
        "corr_abs_cells": float(np.corrcoef(np.abs(rho), np.abs(chi))[0, 1]),
        "support": {"dense": int(dense.sum()), "magnetic": int(magnetic.sum()), "both": int((dense & magnetic).sum()),
                    "magnetic_in_dense": float((dense & magnetic).sum() / max(magnetic.sum(), 1)),
                    "jaccard": float((dense & magnetic).sum() / max((dense | magnetic).sum(), 1))},
        "corr_integrated": float(np.corrcoef(gg.integrated()[0].ravel(), gm.integrated()[0].ravel())[0, 1]),
    }


def depth_sensitivity(path):
    """How fast the gravity and the magnetic data lose sight of a cell with depth: the median
    column norm of each error-weighted sensitivity matrix per km3 of cell, in depth bins below
    the ground (core columns), relative to 0-2 km.  From the 2 km run at ``path``."""
    from geoinv3d.datamodel.mesh import Mesh3D
    from geoinv3d.datamodel.survey import SurveyData
    from geoinv3d.methods.gravity import GravityMethod
    from geoinv3d.methods.magnetics import MagneticsMethod
    from geoinv3d.viz.result_workflow import result_mesh
    r, gg, _ = load_joint(path)
    tm = result_mesh(r)
    mesh, act = Mesh3D.from_discretize(tm), np.asarray(r["_active"], bool)
    from joint_params import FIELD
    norms = {}
    for name, method in (("gravity", GravityMethod()), ("magnetics", MagneticsMethod(inducing_field=tuple(FIELD)))):
        d = r["_datas"][name]
        sv = SurveyData(locations=d["locations"], observed=d["observed"], std=d["std"])
        G = np.asarray(method.make_simulation_active(mesh, sv, act).G, dtype=float) / d["std"][:, None]
        norms[name] = np.linalg.norm(G, axis=0) / (tm.cell_volumes[act] / 1e9)
    depth = gg.depth.reshape(-1, order="F")[act]
    core = np.broadcast_to(gg.core_xy[:, :, None], gg.shape).reshape(-1, order="F")[act]
    out, ref = [], None
    for lo, hi in ((0, 2), (2, 4), (4, 8), (8, 10), (10, 40)):
        sel = core & (depth >= lo * 1e3) & (depth < hi * 1e3)
        if not sel.any():
            continue
        v = {k: float(np.median(n[sel])) for k, n in norms.items()}
        ref = ref or v
        out.append({"from_km": lo, "to_km": hi, "gravity": v["gravity"] / ref["gravity"],
                    "magnetics": v["magnetics"] / ref["magnetics"]})
    return out


def load_joint(path):
    r = load(path)
    gg, gm = Grid(r, r["_models"]["gravity"]), Grid(r, r["_models"]["magnetics"])
    return r, gg, gm


def main():
    global DATA, LOW, FIGS, COUPLINGS
    v2 = "--set" in sys.argv and sys.argv[sys.argv.index("--set") + 1] == "v2"
    if v2:
        DATA, LOW = ROOT / "data" / "ec2_runs_fixed", ROOT / "data" / "lowres_runs_fixed"
        FIGS, COUPLINGS = ROOT / "figures_v2", COUPLINGS_V2
    FIGS.mkdir(exist_ok=True)
    runs = {}
    for k, *_ in COUPLINGS:
        if (DATA / k / "result.zip").exists():
            runs[k] = load_joint(DATA / k)
    keys = [k for k, *_ in COUPLINGS if k in runs]
    labels = {k: lab for k, lab, _ in COUPLINGS}
    short = {k: s for k, _, s in COUPLINGS}
    ref = runs["none"]
    numbers = {"keys": keys, "full": {}, "runs": {}, "low": {}, "single": {},
               "mesh": {"shape": list(ref[1].shape), "n_cells": int(ref[1].mesh.n_cells), "n_active": int(ref[1].active.sum())},
               "n_data": {k: int(len(v["observed"])) for k, v in ref[0]["_datas"].items()}}
    for k in keys:
        r, gg, gm = runs[k]
        z, v, dz = gg.column(MAIN_HIGH)
        numbers["full"][k] = {
            "gravity": {**gg.shares(), **data_fit(r["_datas"]["gravity"]), "main": body(z, v, dz),
                        "box_sandur": box_stats(Grid(r, np.clip(r["_models"]["gravity"], 0, None)), BOX),
                        "min": float(gg.m.min()), "max": float(gg.m.max())},
            "magnetics": {**gm.shares(), **data_fit(r["_datas"]["magnetics"]), "box_sandur": box_stats(gm, BOX),
                          "min": float(gm.m.min()), "max": float(gm.m.max())},
            "coupling": coupling_measures(gg, gm), "info": r.get("coupling"), "settings": r.get("settings"),
            "stopped": r.get("stopped_early"),
            "n_iterations": r.get("n_iterations"), "notes": r.get("notes")}
        # how far each model moved from the uncoupled one
        for name, g, g0 in (("gravity", gg, ref[1]), ("magnetics", gm, ref[2])):
            core = g.core_xy[:, :, None] & g.active
            numbers["full"][k][name]["corr_with_none"] = float(np.corrcoef(g.m[core], g0.m[core])[0, 1])
            numbers["full"][k][name]["corr_integrated_with_none"] = float(
                np.corrcoef(g.integrated()[0].ravel(), g0.integrated()[0].ravel())[0, 1])
        run = DATA / k / "run.json"
        if run.exists():
            numbers["runs"][k] = json.loads(run.read_text())
    # the single inversions with the same settings, against the uncoupled joint run
    sg, sm = load(SINGLE_G), load(SINGLE_M)
    gsg, gsm = Grid(sg), Grid(sm)
    core = ref[1].core_xy[:, :, None] & ref[1].active
    if gsg.shape == ref[1].shape:
        zs, vs, dzs = gsg.column(MAIN_HIGH)
        numbers["single"] = {"gravity_corr": float(np.corrcoef(gsg.m[core], ref[1].m[core])[0, 1]),
                             "magnetics_corr": float(np.corrcoef(gsm.m[core], ref[2].m[core])[0, 1]),
                             "gravity_chi2": data_fit(sg["_data"])["chi2"], "magnetics_chi2": data_fit(sm["_data"])["chi2"],
                             "gravity_main": body(zs, vs, dzs), "magnetics_box": box_stats(gsm, BOX),
                             "coupling": coupling_measures(gsg, gsm)}
    # the 2 km study (local), including the group lasso without bounds
    for d in sorted(LOW.iterdir()) if LOW.exists() else []:
        if (d / "result.zip").exists():
            r, gg, gm = load_joint(d)
            numbers["low"][d.name] = {
                "gravity": {**data_fit(r["_datas"]["gravity"]), **gg.shares(), "min": float(gg.m.min()), "max": float(gg.m.max())},
                "magnetics": {**data_fit(r["_datas"]["magnetics"]), **gm.shares(), "min": float(gm.m.min()), "max": float(gm.m.max())},
                "coupling": coupling_measures(gg, gm),
                "seconds": json.loads((d / "run.json").read_text())["seconds"] if (d / "run.json").exists() else None}

    plt.rcParams.update({"font.sans-serif": ["DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9,
                         "axes.titlesize": 10, "figure.dpi": 150, "savefig.bbox": "tight"})
    g0 = ref[1]
    xe = g0.mesh.nodes_x[g0.n_pad:g0.shape[0] - g0.n_pad + 1] / 1e3
    ye = g0.mesh.nodes_y[g0.n_pad:g0.shape[1] - g0.n_pad + 1] / 1e3
    n = len(keys)

    # ── sections: density and susceptibility of each coupling ──
    fig, axs = plt.subplots(n, 2, figsize=(14, 1.85 * n), sharex=True, layout="constrained")
    for r_, k in enumerate(keys):
        _, gg, gm = runs[k]
        for c, (g, style) in enumerate(((gg, RHO), (gm, chi_style()))):
            x, z, v, ground, northing = g.section(SANDUR_N)
            im = axs[r_, c].pcolormesh(x, z, v, **style)
            axs[r_, c].plot(np.repeat(x, 2)[1:-1], np.repeat(ground, 2), color="k", lw=0.9)
            axs[r_, c].axhline(g.z_core_base / 1e3, color="k", lw=0.7, ls="--")
            axs[r_, c].set_ylim(-10.5, 1.6)
            axs[r_, c].set_title(f"{labels[k]} · {'density contrast' if c == 0 else 'susceptibility'}", loc="left", fontsize=9)
            if c == 0:
                axs[r_, c].set_ylabel("Elevation (km)"); im_r = im
            else:
                im_c = im
    for ax in axs[-1]:
        ax.set_xlabel("Easting (km, UTM 43N)")
    fig.colorbar(im_r, ax=axs[:, 0], shrink=0.5, label="Density contrast (g/cc)", location="bottom", aspect=40)
    fig.colorbar(im_c, ax=axs[:, 1], shrink=0.5, label="Susceptibility (SI)", ticks=CHI_TICKS, location="bottom", aspect=40)
    fig.savefig(FIGS / "sections.png"); plt.close(fig)

    # ── slices at about 1.5 km and integrated maps ──
    fig, axs = plt.subplots(4, n, figsize=(2.5 * n, 10.4), sharex=True, sharey=True, layout="constrained")
    for c, k in enumerate(keys):
        _, gg, gm = runs[k]
        x, y, v, dk = gg.depth_slice(1.5)
        a = axs[0, c].pcolormesh(x, y, v.T, **RHO)
        x, y, v, _ = gm.depth_slice(1.5)
        b = axs[1, c].pcolormesh(x, y, v.T, **chi_style())
        x, y, v, dk2 = gg.depth_slice(4.0)
        axs[2, c].pcolormesh(x, y, v.T, **RHO)
        x, y, v, _ = gm.depth_slice(4.0)
        axs[3, c].pcolormesh(x, y, v.T, **chi_style())
        axs[0, c].set_title(short[k], fontsize=9)
        for ax in axs[:, c]:
            ax.set_aspect("equal"); ax.tick_params(labelsize=6.5)
    for r_, lab in enumerate([f"density, {dk:.1f} km", f"susceptibility, {dk:.1f} km", f"density, {dk2:.1f} km", f"susceptibility, {dk2:.1f} km"]):
        axs[r_, 0].set_ylabel(lab + "\nNorthing (km)", fontsize=8)
    fig.colorbar(a, ax=axs, shrink=0.35, label="Density contrast (g/cc)", location="bottom", aspect=35, pad=0.01)
    fig.colorbar(b, ax=axs, shrink=0.35, label="Susceptibility (SI)", ticks=CHI_TICKS, location="bottom", aspect=35, pad=0.01)
    fig.savefig(FIGS / "slices.png"); plt.close(fig)

    # ── residuals of both datasets ──
    fig, axs = plt.subplots(2, n, figsize=(2.5 * n, 5.6), sharex=True, sharey=True, layout="constrained")
    for c, k in enumerate(keys):
        r = runs[k][0]
        f = numbers["full"][k]
        dd = r["_datas"]["gravity"]
        x, y, g = gridded(dd["locations"], dd["observed"] - dd["predicted"])
        a = imshow(axs[0, c], g, x, y, cmap="RdBu_r", vmin=-3, vmax=3)
        axs[0, c].set_title(f"{short[k]}\nχ²/N {f['gravity']['chi2']:.2f} · {f['gravity']['rms']:.2f} mGal", fontsize=8.5)
        dd = r["_datas"]["magnetics"]
        x, y, g = gridded(dd["locations"], dd["observed"] - dd["predicted"])
        b = imshow(axs[1, c], g, x, y, cmap="RdBu_r", vmin=-150, vmax=150)
        axs[1, c].set_title(f"χ²/N {f['magnetics']['chi2']:.2f} · {f['magnetics']['rms']:.0f} nT", fontsize=8.5)
        for ax in axs[:, c]:
            ax.set_aspect("equal"); ax.tick_params(labelsize=6.5)
    fig.colorbar(a, ax=axs[0], shrink=0.8, label="gravity residual (mGal)", extend="both")
    fig.colorbar(b, ax=axs[1], shrink=0.8, label="magnetic residual (nT)", extend="both")
    fig.savefig(FIGS / "residuals.png"); plt.close(fig)

    # ── the relation between the two properties, cell by cell ──
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.3 * rows), sharex=True, sharey=True, layout="constrained")
    axs = np.atleast_2d(axs)
    for i, k in enumerate(keys):
        _, gg, gm = runs[k]
        core = gg.core_xy[:, :, None] & gg.active & (gg.zc >= gg.z_core_base)[None, None, :]
        ax = axs.ravel()[i]
        h = ax.hist2d(gm.m[core], gg.m[core], bins=[np.linspace(-0.05, 1.05, 56), np.linspace(-0.3, 0.6, 46)],
                      norm=LogNorm(vmin=1, vmax=3e5), cmap="viridis")
        ax.set_title(short[k], fontsize=9)
        ax.axhline(0, color="w", lw=0.4); ax.axvline(0, color="w", lw=0.4)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    for ax in axs[-1]:
        ax.set_xlabel("Susceptibility (SI)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Density contrast (g/cc)")
    fig.colorbar(h[3], ax=axs, shrink=0.6, label="core cells")
    fig.savefig(FIGS / "crossplot.png"); plt.close(fig)

    # ── integrated maps: density with the susceptibility as contours ──
    from scipy.ndimage import gaussian_filter
    fig, axs = plt.subplots(2, n, figsize=(2.5 * n, 5.4), sharex=True, sharey=True, layout="constrained")
    for c, k in enumerate(keys):
        _, gg, gm = runs[k]
        a = axs[0, c].pcolormesh(xe, ye, gg.integrated()[0].T, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
        b = axs[1, c].pcolormesh(xe, ye, gm.integrated()[0].T, cmap="Oranges", vmin=0, vmax=1.0)
        axs[0, c].set_title(short[k], fontsize=9)
        for ax in axs[:, c]:
            ax.set_aspect("equal"); ax.tick_params(labelsize=6.5)
    fig.colorbar(a, ax=axs[0], shrink=0.8, label="integrated density\n(g/cc·km)")
    fig.colorbar(b, ax=axs[1], shrink=0.8, label="integrated susceptibility\n(SI·km)", extend="max")
    fig.savefig(FIGS / "integrated.png"); plt.close(fig)

    # ── profiles with depth in the Sandur belt ──
    edges = np.arange(0, 12.01, 0.5)
    mid = (edges[:-1] + edges[1:]) / 2
    palette = ["#7f7f7f", "#1f5f8b", "#2e9c6a", "#d68910", "#8e44ad", "#c0392b", "#c0392b"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    cx, cy = g0.mesh.cell_centers_x, g0.mesh.cell_centers_y
    sel = (((cx >= BOX[0]) & (cx <= BOX[1]))[:, None] & ((cy >= BOX[2]) & (cy <= BOX[3]))[None, :])[:, :, None]
    for i, k in enumerate(keys):
        _, gg, gm = runs[k]
        for ax, g, positive in ((axs[0], gg, True), (axs[1], gm, False)):
            w = (np.clip(g.m, 0, None) if positive else np.abs(g.m)) * g.vol * g.active * sel
            d = g.depth / 1e3
            half = (g.dz / 2e3)[None, None, :]
            cum = np.array([(w * np.clip((e - (d - half)) / (2 * half), 0, 1)).sum() for e in edges])
            ax.plot(np.diff(cum) / 1e9 / np.diff(edges), mid, "--" if k.endswith("_uncoupled") else "-",
                    color=palette[i % len(palette)], lw=1.8, label=short[k])
    axs[0].set_xlabel("excess mass per km of depth (g/cc·km³ per km)"); axs[1].set_xlabel("susceptibility × volume per km of depth (SI·km³ per km)")
    axs[0].set_ylabel("Depth below the ground (km)"); axs[0].set_ylim(12, 0)
    axs[0].set_title("Dense rock in the Sandur belt"); axs[1].set_title("Magnetic rock in the Sandur belt")
    for ax in axs:
        ax.grid(alpha=0.3)
    axs[1].legend(frameon=False, fontsize=8)
    fig.savefig(FIGS / "profiles.png"); plt.close(fig)

    if v2:   # the magnetic data with a magnetization vector per cell (remanence)
        from fixes_figures import mvi
        from fixes_summary import magnetic_pair
        numbers["magnetic"] = magnetic_pair()
        # the stations the magnetic single inversion underfits by more than 150 nT, in each joint run
        from fixes_summary import MAGNETIC, UNDERFIT_NT
        single = load(MAGNETIC / "beta1")["_data"]
        bad = (single["observed"] - single["predicted"]) > UNDERFIT_NT
        under = {}
        for k in keys:
            dd = runs[k][0]["_datas"]["magnetics"]
            if np.allclose(dd["locations"][:, :2], single["locations"][:, :2]):
                rr = dd["observed"] - dd["predicted"]
                under[k] = {"n_over": int((rr > UNDERFIT_NT).sum()), "mean_at_single": float(rr[bad].mean()),
                            "rms_at_single": float(np.sqrt(np.mean(rr[bad] ** 2)))}
        numbers["magnetic"]["joint_underfit"] = under
        numbers["depth_sensitivity"] = depth_sensitivity(LOW / "none")
        numbers["gl_info"] = {k: runs[k][0].get("group_lasso") and {
            key: runs[k][0]["group_lasso"].get(key) for key in (
                "criterion", "lambda1", "lambda1_max", "lambda2", "data_weights", "weighting",
                "balance", "admm_iterations")} for k in keys if k.startswith("group_lasso")}
        mvi(FIGS, name="mvi")
        # the constraints from the rock samples (Section 5)
        from constraint_figures import constraints
        numbers["constraints"] = constraints(FIGS)
        # the assessment against the published geology (Section 7)
        from literature_figures import literature
        numbers["literature"] = literature(FIGS, numbers["constraints"])
    (FIGS / "numbers.json").write_text(json.dumps(numbers, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("figures and numbers written to", FIGS, "for", keys)


if __name__ == "__main__":
    main()
