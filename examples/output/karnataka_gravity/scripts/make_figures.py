"""Figures (Chinese and English) and numbers for the Karnataka gravity comparison report.

    py examples/output/karnataka_gravity/scripts/make_figures.py
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from common import (AOI, DATA, FIGS, FULL, LOWRES, MAIN_HIGH, NW_HIGH, body, column, data_fit,  # noqa: E402
                    grids, load_full, load_lowres, metrics, simpeg_8km)

L = {
    "zh": {"font": ["Microsoft YaHei", "DejaVu Sans"], "E": "东向 (km)", "N": "北向 (km)", "EU": "东向 (km, UTM 43N)",
           "depth": "深度 (km)", "rho": "密度差 (g/cc)", "raw": "布格重力异常（1 km 抽稀）", "reg": "二阶趋势面（区域场）",
           "res": "剩余异常（反演输入）", "resid": "残差 观测 − 预测 (mGal)", "at": "深度 {d:.1f} km",
           "share": "|质量| 占比（%/km，核心区下方）", "core": "核心网格底界 10 km", "sec": "北向 {n:.1f} km",
           "integ": "垂向积分密度 (g/cc·km)", "cent": "质心深度 (km)", "prof_full": "全分辨率（1 km 网格）",
           "prof_low": "2 km 参数实验", "spec_t": "剩余异常的径向平均功率谱（Spector & Grant）",
           "wn": "波数 (周期/km)", "spec_pts": "径向平均功率谱", "deep": "深部", "mid": "中部", "shallow": "浅部",
           "centre_t": "主布格高值正下方的密度", "deep8": "SimPEG 8 km 深网格（9 月 24 日，α_s=0.05）",
           "floor": "8 km 网格的最深单元（{z:.1f} km）", "lv_t": "主高值下方的半峰值深度范围：2 km 实验与 1 km 结果",
           "lv_2": "2 km 网格（1,295 个数据）", "lv_1": "1 km 网格（5,040 个数据）"},
    "en": {"font": ["DejaVu Sans"], "E": "Easting (km)", "N": "Northing (km)", "EU": "Easting (km, UTM 43N)",
           "depth": "Depth (km)", "rho": "Density contrast (g/cc)", "raw": "Bouguer anomaly (1 km)",
           "reg": "Second-order trend surface (regional)", "res": "Residual anomaly (inverted)",
           "resid": "Residual, observed − predicted (mGal)", "at": "Depth {d:.1f} km",
           "share": "|mass| share (%/km, below the core area)", "core": "Base of the core mesh, 10 km",
           "sec": "northing {n:.1f} km", "integ": "Vertically integrated density (g/cc·km)",
           "cent": "Centroid depth (km)", "prof_full": "Full resolution (1 km mesh)", "prof_low": "2 km study",
           "spec_t": "Radially averaged power spectrum of the residual (Spector & Grant)", "wn": "Wavenumber (cycles/km)",
           "spec_pts": "Radially averaged spectrum", "deep": "Deep", "mid": "Intermediate", "shallow": "Shallow",
           "centre_t": "Density under the main Bouguer high", "deep8": "SimPEG, 8 km deep mesh (24 Sep, α_s=0.05)",
           "floor": "deepest cell of the 8 km mesh ({z:.1f} km)",
           "lv_t": "Half-maximum depth range under the main high: 2 km study and 1 km runs",
           "lv_2": "2 km mesh (1,295 data)", "lv_1": "1 km mesh (5,040 data)"},
}
SLICE_RUNS = ["original_sparse", "l1l2_irls", "as1_beta0.5", "as1_beta1", "as1_beta1.5"]
LOW_PROFILES = [("base", "α_s=1e-4, [0,2,2,1], sens", "#7f7f7f", "-"), ("as1_p0222", "α_s=1, [0,2,2,2], sens", "#d68910", "-"),
                ("as1_p0222_dw2", "α_s=1, β=2", "#8e44ad", "-"), ("as1_p0222_dw1", "α_s=1, β=1", "#2e9c6a", "-"),
                ("as1_p0222_dw0.5", "α_s=1, β=0.5", "#1f5f8b", "-")]


def gridded(locs, values):
    xs, ys = np.unique(locs[:, 0]), np.unique(locs[:, 1])
    g = np.full((len(ys), len(xs)), np.nan)
    g[np.searchsorted(ys, locs[:, 1]), np.searchsorted(xs, locs[:, 0])] = values
    return xs / 1e3, ys / 1e3, g


def label(key, lang):
    return next(zh if lang == "zh" else en for k, zh, en, *_ in FULL if k == key)


def core_view(meta):
    """Core-area model and axes for maps and sections."""
    mesh, m, vol, depth, n_pad, n_core_z = grids(meta)
    nx, ny, _ = m.shape
    sx, sy = slice(n_pad, nx - n_pad), slice(n_pad, ny - n_pad)
    return mesh, m, depth, sx, sy, n_pad


def spectrum():
    import rasterio
    from geoinv3d.methods.regional import fit_trend
    with rasterio.open(DATA / "NGPM_BA.tiff") as src:
        a = src.read(1).astype(float); nod = src.nodata; tr = src.transform
    if nod is not None:
        a[a == nod] = np.nan
    X = tr.c + tr.a * (np.arange(a.shape[1]) + 0.5)
    Y = tr.f + tr.e * (np.arange(a.shape[0]) + 0.5)
    ix = (X >= AOI[0]) & (X <= AOI[1]); iy = (Y >= AOI[2]) & (Y <= AOI[3])
    sub = a[np.ix_(iy, ix)]
    XX, YY = np.meshgrid(X[ix], Y[iy])
    ok = np.isfinite(sub)
    res = np.zeros_like(sub)
    res[ok] = sub[ok] - fit_trend(np.column_stack([XX[ok], YY[ok]]), sub[ok], 2)
    ny_, nx_ = res.shape
    P = np.abs(np.fft.fftshift(np.fft.fft2(res * np.outer(np.hanning(ny_), np.hanning(nx_))))) ** 2
    dk = abs(tr.a) / 1e3
    K = np.hypot(*np.meshgrid(np.fft.fftshift(np.fft.fftfreq(nx_, dk)), np.fft.fftshift(np.fft.fftfreq(ny_, dk))))
    step = 1 / (nx_ * dk)
    kb, pb = [], []
    for lo in np.arange(step / 2, K.max() - step, step):
        s = (K >= lo) & (K < lo + step)
        if s.any():
            kb.append(K[s].mean()); pb.append(P[s].mean())
    kb, lnp = np.array(kb), np.log(np.array(pb))
    fits = {}
    for name, (k0, k1) in {"deep": (0.03, 0.12), "mid": (0.12, 0.3), "shallow": (0.3, 0.6)}.items():
        s = (kb >= k0) & (kb <= k1)
        p = np.polyfit(kb[s], lnp[s], 1)
        fits[name] = {"k": [k0, k1], "depth_km": float(-p[0] / (4 * np.pi)), "fit": p.tolist()}
    return kb, lnp, fits


def crop_gravity(path, title):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    g = im.crop((0, int(0.075 * h), int(0.505 * w), h))
    canvas = Image.new("RGB", (g.width, g.height + 44), "white")
    canvas.paste(g, (0, 44))
    try:
        font = ImageFont.truetype("arialbd.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(canvas).text((16, 8), title, fill="black", font=font)
    return canvas


def main():
    runs = {k: load_full(k) for k, *_ in FULL}
    low = {k: load_lowres(k) for k in LOWRES}
    orig = runs["original_sparse"]
    d = orig["_data"]
    kb, lnp, spec = spectrum()
    numbers = {"data": {"n": int(len(d["observed"])),
                        "raw": [float((d["observed"] + d["regional"]).min()), float((d["observed"] + d["regional"]).max())],
                        "regional": [float(d["regional"].min()), float(d["regional"].max())],
                        "residual": [float(d["observed"].min()), float(d["observed"].max())]},
               "spectrum": {k: round(v["depth_km"], 2) for k, v in spec.items()}, "full": {}, "low": {}}

    # numbers: full-resolution runs
    integ, cent = {}, {}
    for k, *_ in FULL:
        meta = runs[k]
        mesh, m, depth, sx, sy, _ = core_view(meta)
        dz = mesh.h[2]
        I = (m[sx, sy, :] * dz[None, None, :] / 1e3).sum(axis=2)          # g/cc km
        A = (np.abs(m[sx, sy, :]) * dz[None, None, :]).sum(axis=2)
        Z = (np.abs(m[sx, sy, :]) * dz[None, None, :] * depth[None, None, :]).sum(axis=2) / np.maximum(A, 1e-12) / 1e3
        Z[A < 0.2 * A.max()] = np.nan
        integ[k], cent[k] = I, Z
        xc, yc = mesh.cell_centers_x[sx], mesh.cell_centers_y[sy]
        box = (np.abs(xc[:, None] - MAIN_HIGH[0]) <= 5000) & (np.abs(yc[None, :] - MAIN_HIGH[1]) <= 5000)
        numbers["full"][k] = {**metrics(meta), **data_fit(meta), "settings": meta["settings"],
                              "main": body(*column(meta, MAIN_HIGH)), "nw": body(*column(meta, NW_HIGH)),
                              "centroid_main_box_km": float(np.nanmedian(np.where(box, Z, np.nan)))}
    keys = [k for k, *_ in FULL]
    C = np.corrcoef(np.array([integ[k].ravel() for k in keys]))
    numbers["integrated_corr"] = {"keys": keys, "matrix": np.round(C, 3).tolist(),
                                  "min_offdiag": float(C[~np.eye(len(keys), dtype=bool)].min())}
    xs, ys, g_res = gridded(d["locations"], d["observed"])
    for k in keys:   # the integrated density against the residual anomaly (the data)
        mesh = grids(runs[k])[0]
        numbers["full"][k]["corr_integrated_vs_data"] = float(np.corrcoef(
            integ[k].T[:g_res.shape[0], :g_res.shape[1]][np.isfinite(g_res)], g_res[np.isfinite(g_res)])[0, 1]) \
            if integ[k].T.shape == g_res.shape else None
    for k in LOWRES:
        meta = low[k]
        numbers["low"][k] = {**metrics(meta), **data_fit(meta), "settings": meta["settings"],
                             "main": body(*column(meta, MAIN_HIGH))}
    s8 = simpeg_8km()
    z8, v8 = np.array(s8["profiles"]["main"]["depth_km"]), np.array(s8["profiles"]["main"]["density"])
    numbers["simpeg_8km"] = {"chi": s8["chi"], "n_data": s8["n_data"], "time_s": s8["inversion_time_s"],
                             "deepest_km": s8["deepest_cell_km"], "main": body(z8, v8),
                             "nw": body(np.array(s8["profiles"]["nw"]["depth_km"]), np.array(s8["profiles"]["nw"]["density"])),
                             "at_9km": float(np.interp(9.2, z8, v8))}

    for lang in ("zh", "en"):
        t = L[lang]
        out = FIGS / lang
        out.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"font.sans-serif": t["font"], "axes.unicode_minus": False, "font.size": 9,
                             "axes.titlesize": 10, "figure.dpi": 150, "savefig.bbox": "tight"})

        # data and regional field
        _, _, g_reg = gridded(d["locations"], d["regional"])
        _, _, g_raw = gridded(d["locations"], d["observed"] + d["regional"])
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"wspace": 0.45})
        v = np.nanmax(np.abs(g_res))
        for ax, g, tt, cm, lim in [(axs[0], g_raw, t["raw"], "viridis", None), (axs[1], g_reg, t["reg"], "viridis", None),
                                   (axs[2], g_res, t["res"], "RdBu_r", (-v, v))]:
            im = ax.imshow(g, origin="lower", extent=(xs[0] - .5, xs[-1] + .5, ys[0] - .5, ys[-1] + .5), cmap=cm,
                           vmin=lim[0] if lim else None, vmax=lim[1] if lim else None)
            ax.set_title(tt); ax.set_xlabel(t["EU"]); ax.set_aspect("equal")
            plt.colorbar(im, ax=ax, shrink=0.78, label="mGal")
        axs[0].set_ylabel(t["N"])
        fig.savefig(out / "data.png"); plt.close(fig)

        # spectrum
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(kb, lnp, "o", ms=3, color="#5b7a99", label=t["spec_pts"])
        for name, col in [("deep", "#c0392b"), ("mid", "#d68910"), ("shallow", "#27ae60")]:
            k0, k1 = spec[name]["k"]; s = (kb >= k0) & (kb <= k1)
            ax.plot(kb[s], np.polyval(spec[name]["fit"], kb[s]), "-", color=col, lw=2,
                    label=f"{t[name]}: h ≈ {spec[name]['depth_km']:.1f} km")
        ax.set_xlim(0, 0.8); ax.set_xlabel(t["wn"]); ax.set_ylabel("ln P"); ax.set_title(t["spec_t"])
        ax.legend(frameon=False); ax.grid(alpha=0.3)
        fig.savefig(out / "spectrum.png"); plt.close(fig)

        # residual maps of the six runs
        fig, axs = plt.subplots(2, 3, figsize=(12.5, 8), gridspec_kw={"wspace": 0.25, "hspace": 0.3})
        for ax, (k, *_) in zip(axs.ravel(), FULL):
            dd = runs[k]["_data"]
            _, _, g = gridded(dd["locations"], dd["observed"] - dd["predicted"])
            im = ax.imshow(g, origin="lower", extent=(xs[0] - .5, xs[-1] + .5, ys[0] - .5, ys[-1] + .5),
                           cmap="RdBu_r", vmin=-3, vmax=3)
            f = numbers["full"][k]
            ax.set_title(f"{label(k, lang)}\nχ²/N {f['chi2']:.2f} · RMS {f['rms']:.2f} mGal", fontsize=9)
            ax.set_aspect("equal"); ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=axs, shrink=0.6, label=t["resid"])
        fig.savefig(out / "residuals.png"); plt.close(fig)

        # depth slices
        fig, axs = plt.subplots(len(SLICE_RUNS), 3, figsize=(10, 3.0 * len(SLICE_RUNS)), sharex=True, sharey=True,
                                gridspec_kw={"wspace": 0.08, "hspace": 0.25})
        for r, k in enumerate(SLICE_RUNS):
            mesh, m, depth, sx, sy, n_pad = core_view(runs[k])
            nx, ny, _ = m.shape
            for c, dz_km in enumerate([2.0, 5.0, 8.0]):
                kk = int(np.argmin(np.abs(depth / 1e3 - dz_km)))
                ax = axs[r, c]
                im = ax.pcolormesh(mesh.nodes_x[n_pad:nx - n_pad + 1] / 1e3, mesh.nodes_y[n_pad:ny - n_pad + 1] / 1e3,
                                   m[sx, sy, kk].T, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
                ax.set_aspect("equal"); ax.tick_params(labelsize=7)
                ax.set_title((label(k, lang) if c == 0 else "") + "\n" + t["at"].format(d=depth[kk] / 1e3), loc="left", fontsize=8.5)
        fig.colorbar(im, ax=axs, shrink=0.4, label=t["rho"])
        fig.savefig(out / "slices.png"); plt.close(fig)

        # E-W sections through the main high
        fig, axs = plt.subplots(len(FULL), 1, figsize=(10, 1.95 * len(FULL)), sharex=True)
        for ax, (k, *_) in zip(axs, FULL):
            mesh, m, depth, sx, sy, n_pad = core_view(runs[k])
            nx = m.shape[0]
            j = int(np.argmin(np.abs(mesh.cell_centers_y - MAIN_HIGH[1])))
            im = ax.pcolormesh(mesh.nodes_x[n_pad:nx - n_pad + 1] / 1e3, -mesh.nodes_z / 1e3, m[sx, j, :].T,
                               cmap="RdBu_r", vmin=-0.3, vmax=0.3)
            ax.axhline(10, color="k", lw=0.7, ls="--"); ax.axvline(MAIN_HIGH[0] / 1e3, color="k", lw=0.5, ls=":")
            ax.set_ylim(-mesh.nodes_z.min() / 1e3, 0); ax.set_ylabel(t["depth"])
            ax.set_title(f"{label(k, lang)} — {t['sec'].format(n=mesh.cell_centers_y[j] / 1e3)}", loc="left")
        axs[-1].set_xlabel(t["EU"])
        fig.colorbar(im, ax=axs, shrink=0.5, label=t["rho"])
        fig.savefig(out / "sections.png"); plt.close(fig)

        # robust (integrated) vs not robust (centroid depth)
        fig, axs = plt.subplots(2, len(SLICE_RUNS), figsize=(16, 6.4), sharex=True, sharey=True,
                                gridspec_kw={"wspace": 0.08, "hspace": 0.2})
        for c, k in enumerate(SLICE_RUNS):
            mesh, m, depth, sx, sy, n_pad = core_view(runs[k])
            nx, ny, _ = m.shape
            xe, ye = mesh.nodes_x[n_pad:nx - n_pad + 1] / 1e3, mesh.nodes_y[n_pad:ny - n_pad + 1] / 1e3
            a = axs[0, c].pcolormesh(xe, ye, integ[k].T, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
            b = axs[1, c].pcolormesh(xe, ye, cent[k].T, cmap="viridis_r", vmin=0, vmax=15)
            axs[0, c].set_title(label(k, lang), fontsize=8.5)
            for ax in axs[:, c]:
                ax.set_aspect("equal"); ax.tick_params(labelsize=7)
        fig.colorbar(a, ax=axs[0], shrink=0.8, label=t["integ"])
        fig.colorbar(b, ax=axs[1], shrink=0.8, label=t["cent"])
        fig.savefig(out / "robust.png"); plt.close(fig)

        # |mass| with depth: full resolution and the 2 km study
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
        for k, zh, en, colr, ls in FULL:
            mesh, m, vol, depth, n_pad, _ = grids(runs[k])
            nx, ny, _ = m.shape
            a = np.abs(m * vol)[n_pad:nx - n_pad, n_pad:ny - n_pad, :].sum(axis=(0, 1))
            o = np.argsort(depth)
            axs[0].plot((a / (mesh.h[2] / 1e3) / a.sum())[o] * 100, depth[o] / 1e3, ls, color=colr, lw=1.8,
                        label=zh if lang == "zh" else en)
        for k, lab, colr, ls in LOW_PROFILES:
            mesh, m, vol, depth, n_pad, _ = grids(low[k])
            nx, ny, _ = m.shape
            a = np.abs(m * vol)[n_pad:nx - n_pad, n_pad:ny - n_pad, :].sum(axis=(0, 1))
            o = np.argsort(depth)
            axs[1].plot((a / (mesh.h[2] / 1e3) / a.sum())[o] * 100, depth[o] / 1e3, ls, color=colr, lw=1.8, label=lab)
        for ax, tt in zip(axs, [t["prof_full"], t["prof_low"]]):
            ax.axhline(10, color="k", lw=0.8, ls="--"); ax.set_title(tt); ax.set_xlabel(t["share"])
            ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=7.5)
        axs[0].set_ylabel(t["depth"]); axs[0].set_ylim(22, 0)
        fig.savefig(out / "mass_profiles.png"); plt.close(fig)

        # density under the main high, with the SimPEG 8 km mesh model
        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.plot(v8, z8, "-", color="#b03a2e", lw=2.4, label=t["deep8"])
        for k, zh, en, colr, ls in FULL:
            z, vv = column(runs[k], MAIN_HIGH)
            ax.plot(vv, z, ls, color=colr, lw=1.6, label=zh if lang == "zh" else en)
        ax.axhline(s8["deepest_cell_km"], color="#b03a2e", lw=0.8, ls="--")
        ax.text(0.02, s8["deepest_cell_km"] - 0.3, t["floor"].format(z=s8["deepest_cell_km"]), fontsize=8, color="#b03a2e")
        ax.axvline(0, color="k", lw=0.5); ax.set_ylim(22, 0)
        ax.set_xlabel(t["rho"]); ax.set_ylabel(t["depth"]); ax.set_title(t["centre_t"])
        ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=7.5, loc="lower right")
        fig.savefig(out / "centre_profiles.png"); plt.close(fig)

        # the 2 km study against the 1 km runs (same place)
        pairs = [("β=0.5", "as1_p0222_dw0.5", "as1_beta0.5"), ("β=1", "as1_p0222_dw1", "as1_beta1"),
                 ("β=1.5", "as1_p0222_dw1.5", "as1_beta1.5"), ("α_s=0.1, β=1", "as0.1_p0222_dw1", "as0.1_beta1")]
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for i, (lab, lk, fk) in enumerate(pairs):
            for off, src, colr, name in [(-0.15, numbers["low"][lk]["main"], "#9aa7b3", t["lv_2"]),
                                         (0.15, numbers["full"][fk]["main"], "#1f5f8b", t["lv_1"])]:
                ax.plot([i + off, i + off], [src["top_km"], src["bottom_km"]], "-", color=colr, lw=7, solid_capstyle="butt",
                        label=name if i == 0 else None)
                ax.plot(i + off, src["centroid_km"], "o", color="white", ms=4, mec="k", mew=0.6)
        ax.set_xticks(range(len(pairs))); ax.set_xticklabels([p[0] for p in pairs])
        ax.set_ylim(12, 0); ax.set_ylabel(t["depth"]); ax.set_title(t["lv_t"], fontsize=9.5)
        ax.grid(alpha=0.3, axis="y"); ax.legend(frameon=False, fontsize=8, loc="lower left")
        fig.savefig(out / "lowres_vs_full.png"); plt.close(fig)

        # Tomofast-x vs SimPEG sections (gravity half of the original figures)
        a = crop_gravity(DATA / "tomofastx" / "simpeg_deepmesh_v2_cross_sections.png", "SimPEG (chi 1.04)")
        b = crop_gravity(DATA / "tomofastx" / "tomofastx_deepmesh_v2_cross_sections.png", "Tomofast-x (chi 0.65)")
        both = Image.new("RGB", (a.width + b.width + 30, max(a.height, b.height)), "white")
        both.paste(a, (0, 0)); both.paste(b, (a.width + 30, 0))
        both.save(out / "tomofastx.png")

    (FIGS / "numbers.json").write_text(json.dumps(numbers, indent=1, ensure_ascii=False), encoding="utf-8")
    print("figures and numbers written to", FIGS)


if __name__ == "__main__":
    main()
