"""Regularization trials on the Block-8 5 km window: which regularization, and which settings,
for the magnetic inversion of a strongly magnetic window.

    py examples/output/block8_regularization/scripts/trials.py run [NAME ...]
    py examples/output/block8_regularization/scripts/trials.py score [NAME ...]
    py examples/output/block8_regularization/scripts/trials.py list

Every trial inverts every second node of the 37.5 m grid in the window (75 m, 4,489 data) on the
same octree (75 m x 37.5 m cells, GLO-90 ground, stations 80 m above it), with the automatic errors
(2 % + 1.5 % of the data's 5-95 % spread) unless the trial says otherwise.  "score" then predicts
the nodes left out (the other three of each 2 x 2 block, 13,467) from the recovered model: a
model that fits noise fits the nodes it saw better than those it did not.  It also measures where
the model sits: below the ground, beside and below the core.

Trials named "syn-..." invert the synthetic data of synthetic.py (a known model on the same
stations) instead, and are scored against the true model as well.

Results go to data/runs/<name>/: result.zip and result.json (as the worker writes them, for the
viewer), run.json (time), score.json and model_grid.npz (the model sampled on a regular grid
below the ground, for the figures).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parents[2]))
sys.path.insert(0, str(HERE))

INPUTS = ROOT / "data" / "inputs"
RUNS = ROOT / "data" / "runs_new_defaults"   # data/runs: the trials of 2 October, with the former defaults
CRS = "EPSG:32643"
WINDOW = [659650.0, 664650.0, 1664550.0, 1669550.0]
FLIGHT = 80.0
SPACING = 75.0
MESH = {"mesh_type": "octree", "core_cell_m": 75.0, "core_cell_z_m": 37.5,
        "depth_core_m": 4000.0, "pad_distance_m": 2000.0}
BOUNDS = {"bounds_lower": 0.0, "bounds_upper": 3.0}   # as the runs of 5 October and the Lp ablation
# the model sampled for the figures: 75 m columns over the window, 25 m steps below the ground
GRID_DEPTHS = np.arange(12.5, 2000.0, 25.0)


CACHE = ROOT / "data" / "cache"
N_PROCESSES = 12


def fast_sensitivities():
    """Every trial has the same mesh and stations, so the magnetic simulation's sensitivity
    matrix is computed once (in N_PROCESSES processes: without choclo SimPEG uses one) and
    read from data/cache/ by the later trials (keyed by the stations, the active cells, the
    field and the mesh)."""
    import hashlib

    from simpeg.potential_fields.magnetics import simulation as msim
    cls = msim.Simulation3DIntegral
    if getattr(cls, "_trials_patched", False):
        return
    init, operator = cls.__init__, cls.linear_operator

    def __init__(self, *a, **kw):
        if kw.get("engine", "geoana") == "geoana":
            kw.setdefault("n_processes", N_PROCESSES)
        init(self, *a, **kw)

    def linear_operator(self):
        if self.store_sensitivities == "forward_only" or getattr(self, "_no_cache", False):
            return operator(self)
        h = hashlib.sha1()
        for arr in (*(rx.locations for src in [self.survey.source_field] for rx in src.receiver_list),
                    np.asarray(self.active_cells), self.mesh.cell_centers):
            h.update(np.ascontiguousarray(arr).tobytes())
        src = self.survey.source_field
        h.update(repr((src.amplitude, src.inclination, src.declination, self.model_type,
                       [rx.components for rx in src.receiver_list])).encode())
        f = CACHE / f"G_{h.hexdigest()[:16]}.npy"
        if f.exists():
            G = np.load(f)
            print(f"[trials] sensitivities from {f.name} {G.shape}", flush=True)
            return G
        t = time.time()
        G = operator(self)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(f, G)
        print(f"[trials] sensitivities {G.shape} in {time.time() - t:.0f} s -> {f.name}", flush=True)
        return G

    cls.__init__, cls.linear_operator, cls._trials_patched = __init__, linear_operator, True


def inducing_field():
    """IGRF-14 at the window's centre for 2020.0 (the survey date is not known), as the
    worker computes it for "igrf": {} (80 m above the ground)."""
    from geoinv3d.methods.igrf import inducing_field as igrf
    f = igrf(float(np.mean(WINDOW[:2])), float(np.mean(WINDOW[2:])), CRS, alt_m=FLIGHT, date="2020-01-01")
    return [float(v) for v in f["inducing_field"]]


# The page's magnetic data card removes an order-2 trend surface by default (since 5 October)
TREND = {"method": "polynomial", "order": 2}


def base(data: str = "real", **over) -> dict:
    """The job every trial starts from, as the page submits it with its defaults; ``over``
    replaces its settings (dataset keys too).

    The regularization's settings are left to the worker, which since 5 October gives a single
    magnetic inversion Lp norms p = (1, 1, 1, 1) at length scale 3 and depth weighting β 1.5
    when the job does not set them (worker.MAG_LP_NORMS, MAG_LP_LENGTH, MAG_DEPTH_BETA)."""
    ds = {"method": "magnetics", "component": "tmi", "noise_pct": "auto", "noise_floor": "auto",
          "method_kwargs": {"inducing_field": inducing_field()}, "regional": TREND}
    if data == "real":
        ds.update(files=[str(INPUTS / "tmi_window.tif")], station_height=FLIGHT,
                  decimate_spacing_m=SPACING)
    else:
        ds.update(files=[str(INPUTS / f"{data}.csv")])
    for k in ("noise_pct", "noise_floor", "regional"):
        if k in over:
            ds[k] = over.pop(k)
    if ds["regional"] is None:
        ds.pop("regional")
    params = {"inversion_mode": "single", "param_mode": "manual", "crs": CRS, "aoi": WINDOW,
              "datasets": [ds], "topography": {"file": str(INPUTS / "dem_glo90.tif")},
              **MESH, **BOUNDS, "regularization_type": "sparse",
              "max_iter": 60, "max_irls_iterations": 40}
    params.update(over)
    return {k: v for k, v in params.items() if v is not None}


def L(n: float) -> dict:
    """Length scales (alpha_x/y/z) of n."""
    return {"alpha_x": float(n), "alpha_y": float(n), "alpha_z": float(n)}


SENS = {"depth_weighting": "sensitivity"}

# The trials: name -> (what it tests, settings).  "default" is the job with the worker's and the
# page's defaults (Lp p = (1,1,1,1), L = 3, depth weighting β 1.5, an order-2 trend removed).
# A: other regularizations and norms, with the same weighting; B: the default's settings, one at
# a time; C: the errors and the trend.
TRIALS = {
    "default":       ("default: Lp (1,1,1,1), L 3, β 1.5", {}),
    # A: the regularizations and norms (β 1.5 throughout, as the worker gives them)
    "lp0221_L1":     ("Lp (0,2,2,1), L 1 (the former default)", {"norms": [0, 2, 2, 1], **L(1)}),
    "lp0111":        ("Lp (0,1,1,1), L 3", {"norms": [0, 1, 1, 1]}),
    "lp0000":        ("Lp (0,0,0,0), L 3", {"norms": [0, 0, 0, 0]}),
    "lp0222":        ("Lp (0,2,2,2), L 3", {"norms": [0, 2, 2, 2]}),
    "l2":            ("smooth L2", {"regularization_type": "l2"}),
    "mgs":           ("minimum gradient support", {"regularization_type": "mgs"}),
    "tv":            ("total variation", {"regularization_type": "tv"}),
    "l1l2_irls08":   ("L1-L2 IRLS, l1_ratio 0.8", {"regularization_type": "l1l2", "l1l2_solver": "irls",
                                                    "l1_ratio": 0.8}),
    # B: the default's settings
    "beta1":         ("β 1", {"depth_weighting_exponent": 1.0}),
    "beta2":         ("β 2", {"depth_weighting_exponent": 2.0}),
    "beta3":         ("β 3", {"depth_weighting_exponent": 3.0}),
    "sens":          ("sensitivity weighting", SENS),
    "L1":            ("length scale 1", L(1)),
    "L6":            ("length scale 6", L(6)),
    "as0.1":         ("α_s 0.1", {"alpha_s": 0.1}),
    "as10":          ("α_s 10", {"alpha_s": 10.0}),
    "b0inf":         ("bounds [0, ∞)", {"bounds_upper": None}),
    "free":          ("no bounds", {"bounds_lower": None, "bounds_upper": None}),
    # C: the errors (floor as a share of the 5-95 % spread of the detrended data: auto is 1.5 %)
    # and the trend
    "f0.75":         ("floor 0.75 % of the spread", {"noise_floor": "share:0.0075"}),
    "f3":            ("floor 3 % of the spread", {"noise_floor": "share:0.03"}),
    "notrend":       ("no trend removed", {"regional": None}),
}


def detrend(xy, values, regional):
    """``values`` less the job's regional field fitted to them (None: unchanged)."""
    from geoinv3d.methods.regional import remove_regional
    return values if not regional else remove_regional(xy, values, regional)[0]


def trend_at(fit_xy, fit_raw, xy, order):
    """The order-``order`` trend surface fitted to (fit_xy, fit_raw), as the worker fits it
    (methods/regional.fit_trend), evaluated at ``xy``."""
    from geoinv3d.methods.regional import _terms
    centre = fit_xy.mean(axis=0)
    scale = max(float(np.abs(fit_xy - centre).max()), 1e-12)
    u, v = ((fit_xy - centre) / scale).T
    coeffs, *_ = np.linalg.lstsq(_terms(u, v, order), fit_raw, rcond=None)
    u, v = ((xy - centre) / scale).T
    return _terms(u, v, order) @ coeffs


def spread(regional=TREND) -> float:
    """The 5-95 % spread of the data the trials invert (every second node in the window, less
    the trend), from which the automatic floor is taken."""
    from geoinv3d.cloud.worker import grid_window
    x, y, v = grid_window(str(INPUTS / "tmi_window.tif"), CRS, WINDOW)
    xx, yy = np.meshgrid(x[::2], y[::2])
    v = v[::2, ::2].ravel()
    ok = np.isfinite(v)
    d = detrend(np.column_stack([xx.ravel()[ok], yy.ravel()[ok]]), v[ok], regional)
    p5, p95 = np.percentile(d, [5, 95])
    return float(p95 - p5)


def synthetic_spread(regional=TREND) -> float:
    """The same for the synthetic data (synthetic.csv: x, y, z, tmi)."""
    a = np.loadtxt(INPUTS / "synthetic.csv", delimiter=",", skiprows=1)
    d = detrend(a[:, :2], a[:, 3], regional)
    p5, p95 = np.percentile(d, [5, 95])
    return float(p95 - p5)


def params_of(name: str) -> dict:
    synthetic = name.startswith("syn-")
    key = name[4:] if synthetic else name
    over = dict(TRIALS[key][1])
    floor = over.get("noise_floor")
    if isinstance(floor, str) and floor.startswith("share:"):
        over["noise_floor"] = float(floor[6:]) * spread()
    p = base("synthetic" if synthetic else "real", **over)
    p["task_id"] = f"block8-{name}"
    return p


def run(name: str) -> None:
    from geoinv3d.cloud.worker import pack_result, result_metadata_json, run_data_pipeline
    dest = RUNS / name
    if (dest / "result.zip").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    t = time.time()
    params = params_of(name)
    (dest / "params.json").write_text(json.dumps(params, indent=1))
    result = run_data_pipeline(params, str(INPUTS))
    pack_result(result, str(dest / "result.zip"))
    (dest / "result.json").write_text(result_metadata_json(result), encoding="utf-8")
    (dest / "run.json").write_text(json.dumps({"seconds": round(time.time() - t)}))
    print(f"== {name}: {time.time() - t:.0f} s", flush=True)


# ── scoring ──────────────────────────────────────────────────────────────────

def load(name: str) -> dict:
    import io
    import zipfile
    with zipfile.ZipFile(RUNS / name / "result.zip") as zf:
        meta = json.loads(zf.read("result.json"))
        npy = lambda n: np.load(io.BytesIO(zf.read(n)))
        data = dict(np.load(io.BytesIO(zf.read("data.npz"))))
        return {"meta": meta, "model": npy("recovered_model.npy"), "active": npy("active_cells.npy"),
                "data": data}


def mesh_of(meta: dict):
    import discretize
    return discretize.TreeMesh.deserialize(meta["mesh"]) if hasattr(discretize.TreeMesh, "deserialize") \
        else discretize.base.BaseMesh.deserialize(meta["mesh"])


def surface():
    from geoinv3d.cloud.worker import _load_topography
    return _load_topography({"topography": {"file": str(INPUTS / "dem_glo90.tif")}}, str(INPUTS), CRS)[0]


def grid_nodes():
    """Every node of the 37.5 m grid in the window: (x, y, value), and the stations 80 m above
    the ground."""
    from geoinv3d.cloud.worker import grid_window
    x, y, v = grid_window(str(INPUTS / "tmi_window.tif"), CRS, WINDOW)
    xx, yy = np.meshgrid(x, y)
    return xx.ravel(), yy.ravel(), v.ravel()


def sensitivity(dmesh, active, locs, field, chunk=4500, cache=False) -> np.ndarray:
    """The TMI sensitivities (float32) of the active cells at ``locs``, built in chunks (or
    at once, through the cache of fast_sensitivities, with ``cache``)."""
    from simpeg import maps
    from simpeg.potential_fields import magnetics
    fast_sensitivities()
    n = int(active.sum())
    if cache:
        chunk = len(locs)
    out = np.empty((len(locs), n), dtype=np.float32)
    for i in range(0, len(locs), chunk):
        rx = magnetics.receivers.Point(locs[i:i + chunk], components=["tmi"])
        src = magnetics.sources.UniformBackgroundField(receiver_list=[rx], amplitude=field[0],
                                                       inclination=field[1], declination=field[2])
        sim = magnetics.simulation.Simulation3DIntegral(
            survey=magnetics.survey.Survey(source_field=src), mesh=dmesh,
            chiMap=maps.IdentityMap(nP=n), active_cells=active, store_sensitivities="ram")
        sim._no_cache = not cache
        out[i:i + chunk] = np.asarray(sim.G, dtype=np.float32)
        del sim
    return out


_HOLDOUT = {}


def holdout(r: dict, synthetic: bool):
    """(locations, observed, G) of the nodes the trials leave out, for this run's mesh (cached
    on disk by the mesh's active-cell count: every trial has the same mesh)."""
    from geoinv3d.cloud.worker import _lift_buried_stations
    n_act = int(r["active"].sum())
    key = (n_act, synthetic)
    if key in _HOLDOUT:
        return _HOLDOUT[key]
    cache = ROOT / "data" / f"holdout_{n_act}.npz"
    gfile = ROOT / "data" / f"holdout_G_{n_act}.npy"
    if cache.exists() and gfile.exists():
        h = dict(np.load(cache))
        G = np.load(gfile, mmap_mode="r")
    else:
        x, y, v = grid_nodes()
        fx, fy = r["data"]["locations"][:, 0], r["data"]["locations"][:, 1]
        ix = lambda a: np.round((a - WINDOW[0]) / 37.5).astype(int)
        iy = lambda b: np.round((b - WINDOW[2]) / 37.5).astype(int)
        px, py = ix(fx[0]) % 2, iy(fy[0]) % 2       # the fitted nodes' parity in each direction
        fitted = (ix(x) % 2 == px) & (iy(y) % 2 == py)
        keep = ~fitted & np.isfinite(v)
        locs = np.column_stack([x[keep], y[keep], surface()(x[keep], y[keep]) + FLIGHT])
        dmesh = mesh_of(r["meta"])
        _lift_buried_stations(dmesh, r["active"], locs)
        # the nodes farthest from any fitted one (53 m): off in both directions
        diag = (ix(x[keep]) % 2 != px) & (iy(y[keep]) % 2 != py)
        h = {"locations": locs, "observed": v[keep], "diag_flag": diag}
        t = time.time()
        G = sensitivity(dmesh, r["active"], locs, inducing_field())
        np.save(gfile, G)
        np.savez(cache, **h)
        print(f"holdout sensitivities: {G.shape} in {time.time() - t:.0f} s", flush=True)
        G = np.load(gfile, mmap_mode="r")
    if synthetic:
        h = {**h, "observed": synthetic_holdout(h, G)}
    _HOLDOUT[key] = (h, G)
    return _HOLDOUT[key]


def synthetic_holdout(h, G):
    """The synthetic data at the left-out nodes (synthetic.py's true model, its noise)."""
    f = INPUTS / "synthetic_holdout.npy"
    if f.exists():
        return np.load(f)
    raise FileNotFoundError("run synthetic.py first")


def dot(G, m, chunk=2000) -> np.ndarray:
    m32 = np.asarray(m, dtype=np.float32)
    return np.concatenate([np.asarray(G[i:i + chunk]) @ m32 for i in range(0, G.shape[0], chunk)]).astype(float)


def model_grid(dmesh, active, model, surf):
    """The model on 75 m columns over the window, at GRID_DEPTHS below the ground (NaN above)."""
    xs = np.arange(WINDOW[0] + 37.5, WINDOW[1], 75.0)
    ys = np.arange(WINDOW[2] + 37.5, WINDOW[3], 75.0)
    xx, yy = np.meshgrid(xs, ys)
    g = surf(xx, yy)
    act_index = np.full(dmesh.n_cells, -1)
    act_index[active] = np.arange(int(active.sum()))
    out = np.full((len(GRID_DEPTHS),) + xx.shape, np.nan, dtype=np.float32)
    for k, d in enumerate(GRID_DEPTHS):
        pts = np.column_stack([xx.ravel(), yy.ravel(), (g - d).ravel()])
        idx = act_index[dmesh.point2index(pts)]
        vals = np.where(idx >= 0, model[np.maximum(idx, 0)], np.nan)
        out[k] = vals.reshape(xx.shape)
    return xs, ys, g, out


def score(name: str) -> dict:
    from geoinv3d.cloud.worker import _outside_core_parts
    r = load(name)
    meta, m, active = r["meta"], r["model"], r["active"]
    synthetic = name.startswith("syn-")
    dmesh = mesh_of(meta)
    surf = surface()
    d = r["data"]
    ds = meta["datasets"][0]
    floor, pct = float(ds["noise_floor"]), float(ds["noise_pct"])
    fit_res = d["predicted"] - d["observed"]
    s = {"name": name, "label": TRIALS[name[4:] if synthetic else name][0],
         "seconds": json.loads((RUNS / name / "run.json").read_text())["seconds"],
         "n_data": int(len(d["observed"])), "noise_floor": floor, "noise_pct": pct,
         "n_iterations": meta.get("n_iterations"), "converged": meta.get("converged"),
         "convergence": meta.get("convergence"),
         "chi2_fit": float(np.mean((fit_res / d["std"]) ** 2)), "rms_fit": float(np.sqrt(np.mean(fit_res ** 2)))}
    # the nodes left out, with the errors of the auto rule (so trials with other errors compare)
    h, G = holdout(r, synthetic)
    pred = dot(G, m)
    obs = h["observed"]
    reg = ds.get("regional") or {}
    if reg.get("method") == "polynomial":
        # the trend the run removed, from its fitted nodes, off the nodes left out too
        fit_xy = d["locations"][:, :2]
        obs = obs - trend_at(fit_xy, d["observed"] + d["regional"], h["locations"][:, :2], int(reg["order"]))
    elif reg:
        raise ValueError(f"{name}: scoring handles polynomial trends only, not {reg.get('method')}")
    # one error model for every trial: 2 % + 1.5 % of the spread of the default's data (detrended)
    auto_floor = 0.015 * (synthetic_spread() if synthetic else spread())
    std_h = 0.02 * np.abs(obs) + auto_floor
    std_f = 0.02 * np.abs(d["observed"]) + auto_floor
    res = pred - obs
    diag = h["diag_flag"].astype(bool)
    s.update(chi2_fit_auto=float(np.mean((fit_res / std_f) ** 2)),
             chi2_holdout=float(np.mean((res / std_h) ** 2)), rms_holdout=float(np.sqrt(np.mean(res ** 2))),
             rms_holdout_diag=float(np.sqrt(np.mean(res[diag] ** 2))),
             chi2_holdout_diag=float(np.mean((res[diag] / std_h[diag]) ** 2)))
    # the fit to the strongest anomalies (|d| > 2,000 nT) and to the quiet part
    strong = np.abs(obs) > 2000
    s["rms_holdout_strong"] = float(np.sqrt(np.mean(res[strong] ** 2))) if strong.any() else None
    s["rms_holdout_quiet"] = float(np.sqrt(np.mean(res[~strong] ** 2)))
    # where the model is
    cc, vol = dmesh.cell_centers[active], dmesh.cell_volumes[active]
    depth = surf(cc[:, 0], cc[:, 1]) - cc[:, 2]
    inside = ((cc[:, 0] >= WINDOW[0]) & (cc[:, 0] <= WINDOW[1]) & (cc[:, 1] >= WINDOW[2])
              & (cc[:, 1] <= WINDOW[3]))
    mom = np.abs(m) * vol
    gx, gy = np.meshgrid(np.linspace(WINDOW[0], WINDOW[1], 20), np.linspace(WINDOW[2], WINDOW[3], 20))
    z_bottom = float(np.min(surf(gx, gy))) - MESH["depth_core_m"]
    beside, below = _outside_core_parts(dmesh, active, m, tuple(WINDOW), MESH["core_cell_m"], z_bottom)
    s.update(outside_core=float(meta.get("outside_core_share", {}).get("magnetics", np.nan)),
             beside=beside, below=below)
    mi = mom[inside]
    di = depth[inside]
    order = np.argsort(di)
    cum = np.cumsum(mi[order]) / max(mi.sum(), 1e-30)
    s["depth_p50"] = float(di[order][np.searchsorted(cum, 0.5)])
    s["depth_p90"] = float(di[order][np.searchsorted(cum, 0.9)])
    for a, b in ((0, 200), (200, 600), (600, 1500), (1500, 1e9)):
        s[f"share_{a}_{int(min(b, 9999))}"] = float(mi[(di >= a) & (di < b)].sum() / max(mi.sum(), 1e-30))
    # what explains the data: the cells under the window, or those around and below it (the volume
    # share above weighs the large padding cells much; this weighs them by their effect at the stations)
    gfit = next((f for f in sorted(CACHE.glob("G_*.npy"))
                 if np.load(f, mmap_mode="r").shape == (s["n_data"], len(m))), None)
    if gfit is not None:
        Gf = np.load(gfit, mmap_mode="r")
        win = inside & (depth <= MESH["depth_core_m"])
        d_in, d_out = dot(Gf, np.where(win, m, 0.0)), dot(Gf, np.where(win, 0.0, m))
        e_in, e_out = float(d_in @ d_in), float(d_out @ d_out)
        s["data_from_window"] = e_in / (e_in + e_out)
        # and from the top 300 m under the window
        top = win & (depth <= 300.0)
        d_top = dot(Gf, np.where(top, m, 0.0))
        s["data_from_top300"] = float(d_top @ d_top) / (e_in + e_out)
    s.update(chi_max=float(m.max()), chi_min=float(m.min()),
             n_at_upper=int(np.sum(m >= 0.999 * float(meta.get("settings", {}).get("bounds_upper") or np.inf))) if meta.get("settings", {}).get("bounds_upper") else 0,
             chi_p99_window=float(np.percentile(m[inside], 99)))
    if synthetic:
        s.update(synthetic_scores(dmesh, active, m, depth, inside, vol))
    xs, ys, g, mg = model_grid(dmesh, active, m, surf)
    np.savez_compressed(RUNS / name / "model_grid.npz", x=xs, y=ys, ground=g, depths=GRID_DEPTHS, model=mg)
    np.save(RUNS / name / "holdout_predicted.npy", pred.astype(np.float32))
    (RUNS / name / "score.json").write_text(json.dumps(s, indent=1))
    return s


def synthetic_scores(dmesh, active, m, depth, inside, vol) -> dict:
    """How close the model is to synthetic.py's true model, in the window."""
    true = np.load(INPUTS / "synthetic_model.npy")
    a, b = m[inside], true[inside]
    w = vol[inside]
    corr = float(np.corrcoef(a, b)[0, 1])
    err = float(np.sum(w * np.abs(a - b)) / np.sum(w * np.abs(b)))
    # the moment found in the true bodies' cells, and outside them
    body = b > 0
    s = {"syn_corr": corr, "syn_rel_l1": err,
         "syn_moment_in_bodies": float(np.sum(w[body] * a[body]) / max(np.sum(w * np.abs(a)), 1e-30)),
         "syn_moment_ratio": float(np.sum(w * a) / np.sum(w * b))}
    labels = np.load(INPUTS / "synthetic_labels.npy")[inside]
    for k in np.unique(labels[labels > 0]):
        sel = labels == k
        s[f"syn_body{k}_recovered"] = float(np.sum(w[sel] * a[sel]) / np.sum(w[sel] * b[sel]))
    return s


def main():
    args = sys.argv[1:]
    cmd, names = (args[0], args[1:]) if args else ("list", [])
    if cmd == "list":
        for k, (label, over) in TRIALS.items():
            print(f"{k:16s} {label:40s} {over}")
        return
    names = names or list(TRIALS)
    fast_sensitivities()
    for name in names:
        if cmd == "run":
            run(name)
        elif cmd == "score":
            if (RUNS / name / "result.zip").exists():
                s = score(name)
                print(f"{name:16s} chi2 fit {s['chi2_fit']:.2f}  holdout {s['chi2_holdout']:.2f} "
                      f"(rms {s['rms_fit']:.0f}/{s['rms_holdout']:.0f} nT)  depth50 {s['depth_p50']:.0f} m  "
                      f"outside {s['outside_core']:.2f} (below {s['below']:.2f})  {s['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
