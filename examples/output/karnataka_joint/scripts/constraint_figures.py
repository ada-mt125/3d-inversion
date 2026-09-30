"""Figures and numbers of the rock-property constraints (Section 5 of the second joint report).

Called by make_figures.py --set v2.  Inputs: data/rock_properties/measured_by_rock_type.csv (the
measured densities and susceptibilities per rock family, inside the 70 km block and in the
region; from the density report of 30 September 2026), the 2 km runs of data/lowres_runs_fixed
and data/lowres_bounds/<key>/ (joint_params.VARIANTS), and the full-resolution runs of
data/ec2_runs_fixed and data/ec2_runs_bounds/<coupling>_rho35_gb05.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROCKS = ROOT / "data" / "rock_properties" / "measured_by_rock_type.csv"
BACKGROUND = 2.66           # g/cc: the background of the density contrast
COUPLINGS = ("none", "joint_total_variation")
FULL_KEY = "rho35_gb05"     # the constrained full-resolution runs
ORDER = ["Banded iron formation", "Dolerite / gabbro", "Metabasalt / amphibolite",
         "Schist, phyllite, quartzite", "Gneiss / TTG", "Granite", "Felsic volcanic"]


def rocks():
    """{family: {scope: {n, rho_min, rho_median, rho_max, k_p05, k_median, k_p95}}}."""
    out = {}
    with open(ROCKS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["family"], {})[r["scope"]] = {
                "n": int(r["n"]), "rho_min": float(r["rho_min"]), "rho_median": float(r["rho_median"]),
                "rho_max": float(r["rho_max"]), "k_p05": float(r["k_SI_p05"]),
                "k_median": float(r["k_SI_median"]), "k_p95": float(r["k_SI_p95"])}
    return out


def _paths(full):
    from compare_bounds import VARIANTS
    base = ROOT / "data" / ("ec2_runs_fixed" if full else "lowres_runs_fixed")
    var = ROOT / "data" / ("ec2_runs_bounds" if full else "lowres_bounds")
    out = {}
    for c in COUPLINGS:
        for key in ["rho50"] + list(VARIANTS):
            p = base / c if key == "rho50" else (var / f"{c}_{key}" if full else var / key / c)
            if (p / "result.zip").exists():
                out[(key, c)] = p
    return out


def _corr(a, b):
    core = a.core_xy[:, :, None] & a.active
    return float(np.corrcoef(a.m[core], b.m[core])[0, 1])


def numbers():
    from compare_bounds import measures, settings
    from kmodel import data_fit
    from make_figures import load_joint
    out = {"rocks": rocks(), "background": BACKGROUND, "low": {}, "full": {}, "full_key": FULL_KEY,
           "settings": {}}
    for full, dest in ((False, out["low"]), (True, out["full"])):
        for (key, c), p in _paths(full).items():
            dest.setdefault(key, {})[c] = measures(p, key)
            (lo, hi), betas = settings(key)
            out["settings"][key] = {"bounds": [lo, hi], "betas": list(betas)}
            if full:
                r = load_joint(p)[0]
                dest[key][c]["rms"] = {k: data_fit(r["_datas"][k])["rms"] for k in ("gravity", "magnetics")}
                run = p / "run.json"
                if run.exists():
                    dest[key][c]["run"] = json.loads(run.read_text())
    # how far the constraints move the models, against how far the coupling does
    paths = _paths(True)
    models = {kc: load_joint(p) for kc, p in paths.items() if kc[0] in ("rho50", FULL_KEY)}
    corr = {}
    for c in COUPLINGS:
        a, b = models.get(("rho50", c)), models.get((FULL_KEY, c))
        if a and b:
            corr[f"{c}: constrained vs not"] = {"density": _corr(b[1], a[1]), "susceptibility": _corr(b[2], a[2])}
    for key in ("rho50", FULL_KEY):
        a, b = models.get((key, "joint_total_variation")), models.get((key, "none"))
        if a and b:
            corr[f"{key}: JTV vs none"] = {"density": _corr(a[1], b[1]), "susceptibility": _corr(a[2], b[2])}
    out["corr"] = corr
    return out


def figures(figs, nums):
    import matplotlib.pyplot as plt
    from compare_bounds import title
    from kmodel import MAIN_HIGH
    from make_figures import RHO, SANDUR_N, chi_style, load_joint

    # ── the measured densities against the bounds ──
    R = nums["rocks"]
    fams = [f for f in ORDER if f in R]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    for i, f in enumerate(fams):
        y = len(fams) - 1 - i
        for scope, dy, lw, col in (("region", -0.14, 2.0, "#9aa5b1"), ("block", 0.14, 4.5, "#1f5f8b")):
            s = R[f].get(scope)
            if not s:
                continue
            ax.plot([s["rho_min"] - BACKGROUND, s["rho_max"] - BACKGROUND], [y + dy] * 2, color=col, lw=lw,
                    solid_capstyle="butt", label=f"{scope} (range, median)" if i == 0 else None)
            ax.plot(s["rho_median"] - BACKGROUND, y + dy, "o", color="k", ms=3.5)
            ax.text(s["rho_max"] - BACKGROUND + 0.012, y + dy, f"n = {s['n']}", va="center", fontsize=6.5,
                    color="#555")
    for x, ls, lab in ((-0.2, "--", "bounds of Sections 2–4"), (0.5, "--", None),
                       (-0.15, "-", "bounds from the samples"), (0.35, "-", None)):
        ax.axvline(x, color="#c0392b" if ls == "-" else "#7f7f7f", ls=ls, lw=1.2, label=lab)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_yticks(range(len(fams)), fams[::-1], fontsize=8)
    ax.set_xlabel(f"Density contrast to {BACKGROUND} g/cc")
    top = ax.secondary_xaxis("top", functions=(lambda x: x + BACKGROUND, lambda x: x - BACKGROUND))
    top.set_xlabel("Density (g/cc)")
    ax.set_xlim(-0.25, 0.9)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(figs / "constraints_rocks.png")
    plt.close(fig)

    # ── the column under the main high: 2 km variants, and the full-resolution pair ──
    low, full = _paths(False), _paths(True)
    keys = ["rho50", "rho40", "rho35", "rho30", "rho35_gb05", "rho35_b05"]
    style = {"rho50": ("#7f7f7f", "-"), "rho40": ("#f0a35e", "-"), "rho35": ("#d35400", "-"),
             "rho30": ("#7b2d00", "-"), "rho35_gb05": ("#1f5f8b", "-"), "rho35_b05": ("#1f5f8b", ":")}
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.3), sharey=True, layout="constrained")
    for ax, c in zip(axs[:2], COUPLINGS):
        for key in keys:
            if (key, c) in low:
                z, v, _ = load_joint(low[(key, c)])[1].column(MAIN_HIGH)
                col, ls = style[key]
                ax.step(v, z, where="mid", color=col, ls=ls, lw=1.6, label=title(key))
        ax.set_title(f"2 km mesh · {'no coupling' if c == 'none' else 'joint total variation'}", loc="left")
    for c, ls in zip(COUPLINGS, ("-", "--")):
        for key in ("rho50", FULL_KEY):
            if (key, c) in full:
                z, v, _ = load_joint(full[(key, c)])[1].column(MAIN_HIGH)
                axs[2].step(v, z, where="mid", color=style[key][0], ls=ls, lw=1.6,
                            label=f"{'none' if c == 'none' else 'JTV'} · {title(key)}")
    axs[2].set_title("1 km mesh (full resolution)", loc="left")
    for ax in axs:
        ax.set_xlabel("Density contrast (g/cc)")
        ax.grid(alpha=0.3)
    axs[0].set_ylim(11, 0)
    axs[0].set_ylabel("Depth below the ground (km)")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=3, fontsize=7.5, frameon=False,
               title="2 km mesh: density bounds, β gravity / magnetics", title_fontsize=7.5)
    axs[2].legend(fontsize=7, frameon=False)
    fig.savefig(figs / "constraints_column.png")
    plt.close(fig)

    # ── full-resolution sections, without and with the constraints ──
    rows = [(key, c) for c in COUPLINGS for key in ("rho50", FULL_KEY) if (key, c) in full]
    fig, axs = plt.subplots(len(rows), 2, figsize=(14, 1.95 * len(rows)), sharex=True, layout="constrained",
                            squeeze=False)
    rho = {**RHO, "vmin": -0.5, "vmax": 0.5}
    for r_, (key, c) in enumerate(rows):
        _, gg, gm = load_joint(full[(key, c)])
        for k_, (g, st) in enumerate(((gg, rho), (gm, chi_style()))):
            x, z, v, ground, _ = g.section(SANDUR_N)
            im = axs[r_, k_].pcolormesh(x, z, v, **st)
            axs[r_, k_].plot(np.repeat(x, 2)[1:-1], np.repeat(ground, 2), color="k", lw=0.9)
            axs[r_, k_].set_ylim(-10.5, 1.6)
            axs[r_, k_].set_title(f"{'No coupling' if c == 'none' else 'Joint total variation'} · {title(key)} · "
                                  f"{'density contrast' if k_ == 0 else 'susceptibility'}", loc="left", fontsize=9)
            if k_ == 0:
                axs[r_, k_].set_ylabel("Elevation (km)")
                im_r = im
            else:
                im_c = im
    for ax in axs[-1]:
        ax.set_xlabel("Easting (km, UTM 43N)")
    fig.colorbar(im_r, ax=axs[:, 0], shrink=0.5, label="Density contrast (g/cc)", location="bottom", aspect=40)
    fig.colorbar(im_c, ax=axs[:, 1], shrink=0.5, label="Susceptibility (SI)", ticks=[0, 0.05, 0.2, 0.5, 1.0],
                 location="bottom", aspect=40)
    fig.savefig(figs / "constraints_sections.png")
    plt.close(fig)


def constraints(figs):
    nums = numbers()
    figures(figs, nums)
    return nums
