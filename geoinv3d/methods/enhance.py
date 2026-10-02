"""Enhancement of gridded potential-field data: the views interpreters use before inverting.

The processing of e.g. Yang et al. (2026, Ore Geology Reviews 198, 107581) on a ground
magnetic survey: the field reduced to the pole, continued upwards to steady the
derivatives, its vertical derivative, edge detectors built from ratios of derivatives,
and the separation of a regional from a residual field.

* Wavenumber filters (``upward``, ``vertical_derivative``, ``rtp``, ``butterworth``,
  ``bandpass``) work on a grid padded by reflection and tapered towards its mean, so the
  FFT sees no jump at the edges; gaps are filled from their nearest values first and
  are gaps again afterwards.
* Edge detectors (``edges``): the total horizontal derivative (THDR), the analytic
  signal amplitude (ASA), the tilt angle (Miller and Singh 1994), the theta map (Wijns et
  al. 2005: THDR / ASA), the horizontal derivative of the tilt (TDR-THDR, Verduzco et al.
  2004) and the normalized standard deviation (NSTD, Cooper and Cowan 2008).  ``nvdr``
  takes the vertical derivative of a detector, keeps its positive part and scales it to
  its maximum, which sharpens its maxima onto the edges (the idea of the normalized
  vertical derivative of Wang et al. 2009 and Zhu et al. 2021; their exact scaling may
  differ).

Grids are 2-D arrays with rows along y (north, increasing) and columns along x (east),
spacings dx, dy in metres; z is positive down, so the vertical derivative of a field
measured above its sources is |k| times its spectrum.
"""

from __future__ import annotations

import numpy as np


# ── grids ──
def to_grid(x, y, v, spacing: float | None = None, max_nodes: int = 4_000_000):
    """Values at stations on a regular grid: (xg, yg, grid) with NaN away from data.

    Stations already on a lattice (a thinned grid file) are placed as they are;
    scattered ones are interpolated linearly at ``spacing`` (default: their median
    nearest-neighbour distance), and nodes farther than two spacings from any station
    are left empty.
    """
    from scipy.spatial import cKDTree
    x, y, v = (np.asarray(a, dtype=float).ravel() for a in (x, y, v))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[ok], y[ok], v[ok]
    if len(v) < 9:
        raise ValueError("At least 9 stations are needed to grid the data")
    # a lattice (a grid file, perhaps thinned): as many distinct x and y as the stations need;
    # found without a neighbour search, which is slow for the millions of nodes of a survey grid
    if spacing is None:
        ux, uy = np.unique(np.round(x, 3)), np.unique(np.round(y, 3))
        if len(ux) * len(uy) <= 1.05 * len(v) and len(ux) > 2 and len(uy) > 2:
            dxs, dys = np.diff(ux), np.diff(uy)
            dx, dy = np.median(dxs), np.median(dys)
            if np.allclose(dxs, dx, rtol=0.02) and np.allclose(dys, dy, rtol=0.02):
                g = np.full((len(uy), len(ux)), np.nan)
                g[np.searchsorted(uy, np.round(y, 3)), np.searchsorted(ux, np.round(x, 3))] = v
                return ux, uy, g
    tree = cKDTree(np.c_[x, y])
    pick = np.random.default_rng(0).choice(len(x), size=min(len(x), 20000), replace=False)
    nn = tree.query(np.c_[x[pick], y[pick]], k=2)[0][:, 1]
    sp = float(spacing or np.median(nn[nn > 0]))
    from scipy.interpolate import griddata
    nx, ny = int(np.ptp(x) / sp) + 1, int(np.ptp(y) / sp) + 1
    while nx * ny > max_nodes:
        sp *= 1.25
        nx, ny = int(np.ptp(x) / sp) + 1, int(np.ptp(y) / sp) + 1
    xg = x.min() + sp * np.arange(nx)
    yg = y.min() + sp * np.arange(ny)
    X, Y = np.meshgrid(xg, yg)
    g = griddata((x, y), v, (X, Y), method="linear")
    far = tree.query(np.c_[X.ravel(), Y.ravel()])[0].reshape(X.shape) > 2 * sp
    g[far] = np.nan
    return xg, yg, g


def coarsen(xg, yg, g, factor: int):
    """Block averages of ``factor`` x ``factor`` nodes (gaps left out): a grid that
    many times coarser, at the blocks' centres.  Averaging, unlike taking every k-th node,
    keeps the shorter wavelengths from aliasing into the longer ones."""
    k = int(factor)
    if k <= 1:
        return np.asarray(xg), np.asarray(yg), g
    ny, nx = (g.shape[0] // k) * k, (g.shape[1] // k) * k
    blocks = np.asarray(g, dtype=float)[:ny, :nx].reshape(ny // k, k, nx // k, k)
    with np.errstate(invalid="ignore"):
        out = np.nanmean(blocks, axis=(1, 3))
    xs = np.asarray(xg)[:nx].reshape(-1, k).mean(axis=1)
    ys = np.asarray(yg)[:ny].reshape(-1, k).mean(axis=1)
    return xs, ys, out


def sample(xg, yg, grid, x, y):
    """Bilinear values of a grid (NaN filled from neighbours) at points."""
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((np.asarray(yg), np.asarray(xg)), _filled(grid),
                                bounds_error=False, fill_value=None)
    return f(np.c_[np.asarray(y, dtype=float).ravel(), np.asarray(x, dtype=float).ravel()])


def _filled(g):
    """NaNs replaced by their nearest value."""
    from scipy import ndimage as ndi
    g = np.asarray(g, dtype=float)
    bad = ~np.isfinite(g)
    if not bad.any():
        return g
    if bad.all():
        raise ValueError("The grid holds no values")
    idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return g[tuple(idx)]


def _spectrum(g, pad: float = 0.5):
    """The FFT of a grid padded by reflection and tapered towards its mean; returns
    (spectrum, (kx, ky, k) in rad/m, crop) for ``_back``."""
    z = _filled(g)
    ny, nx = z.shape
    py, px = max(8, int(pad * ny)), max(8, int(pad * nx))
    mean = float(z.mean())
    zp = np.pad(z - mean, ((py, py), (px, px)), mode="reflect")
    # a cosine taper over the padding: no jump where the FFT wraps around
    wy = np.ones(zp.shape[0]); wx = np.ones(zp.shape[1])
    ty, tx = 0.5 * (1 - np.cos(np.pi * np.arange(py) / py)), 0.5 * (1 - np.cos(np.pi * np.arange(px) / px))
    wy[:py], wy[-py:] = ty, ty[::-1]
    wx[:px], wx[-px:] = tx, tx[::-1]
    zp = zp * np.outer(wy, wx)
    return np.fft.fft2(zp), (py, px, ny, nx, mean)


def _waves(shape, dx, dy):
    ky = 2 * np.pi * np.fft.fftfreq(shape[0], d=dy)
    kx = 2 * np.pi * np.fft.fftfreq(shape[1], d=dx)
    KX, KY = np.meshgrid(kx, ky)
    return KX, KY, np.hypot(KX, KY)


def _back(F, crop, mask=None, keep_mean=False):
    py, px, ny, nx, mean = crop
    out = np.real(np.fft.ifft2(F))[py:py + ny, px:px + nx]
    if keep_mean:
        out = out + mean
    if mask is not None:
        out[mask] = np.nan
    return out


def _filter(g, dx, dy, response, keep_mean=False):
    """A grid through a wavenumber response (a function of kx, ky, k)."""
    mask = ~np.isfinite(np.asarray(g, dtype=float))
    F, crop = _spectrum(g)
    KX, KY, K = _waves(F.shape, dx, dy)
    return _back(F * response(KX, KY, K), crop, mask, keep_mean)


# ── wavenumber filters ──
def upward(g, dx, dy, height: float):
    """The field continued ``height`` metres upwards (downwards if negative: unstable)."""
    return _filter(g, dx, dy, lambda kx, ky, k: np.exp(-k * height), keep_mean=True)


def vertical_derivative(g, dx, dy, order: int = 1):
    """d^n/dz^n (z down), per metre^n."""
    return _filter(g, dx, dy, lambda kx, ky, k: k ** order)


def horizontal_derivatives(g, dx, dy):
    """(d/dx, d/dy) per metre, by FFT."""
    return (_filter(g, dx, dy, lambda kx, ky, k: 1j * kx),
            _filter(g, dx, dy, lambda kx, ky, k: 1j * ky))


def _theta(kx, ky, k, inc, dec):
    """Blakely's Theta for a unit vector of inclination ``inc`` (down) and declination
    ``dec`` (from the grid's y axis, clockwise): its z part plus i times its projection
    on the wavenumber direction."""
    i, d = np.radians(inc), np.radians(dec)
    fx, fy, fz = np.cos(i) * np.sin(d), np.cos(i) * np.cos(d), np.sin(i)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = fz + 1j * (fx * kx + fy * ky) / k
    t[k == 0] = fz
    return t


def rtp(g, dx, dy, inc: float, dec: float, mag_inc=None, mag_dec=None, damping=None):
    """The total-field anomaly reduced to the pole: as if field and magnetization were
    vertical.  Magnetization along the field unless ``mag_inc``/``mag_dec`` are given
    (induced only; remanence in another direction needs them).  Near the magnetic
    equator the filter blows up along the declination; it is damped there:
    1/Theta becomes conj(Theta) / (|Theta|^2 + damping^2), with ``damping`` by default
    max(0, sin^2 30° - sin^2 I) (none from |I| = 30° up)."""
    mi = inc if mag_inc is None else mag_inc
    md = dec if mag_dec is None else mag_dec
    if damping is None:
        damping = max(0.0, np.sin(np.radians(30)) ** 2 - np.sin(np.radians(min(abs(inc), abs(mi)))) ** 2)

    def resp(kx, ky, k):
        t = _theta(kx, ky, k, inc, dec) * _theta(kx, ky, k, mi, md)
        return np.conj(t) / (np.abs(t) ** 2 + damping ** 2)

    return _filter(g, dx, dy, resp, keep_mean=True)


def butterworth(g, dx, dy, cutoff_m: float, order: int = 4, high: bool = False):
    """Low-pass (or high-pass) Butterworth filter at the wavelength ``cutoff_m``."""
    kc = 2 * np.pi / float(cutoff_m)

    def resp(kx, ky, k):
        lp = 1.0 / np.sqrt(1.0 + (k / kc) ** (2 * order))
        return 1.0 - lp if high else lp

    return _filter(g, dx, dy, resp, keep_mean=not high)


def bandpass(g, dx, dy, short_m: float, long_m: float, order: int = 4):
    """The wavelengths between ``short_m`` and ``long_m`` (Butterworth edges)."""
    k1, k2 = 2 * np.pi / float(long_m), 2 * np.pi / float(short_m)

    def resp(kx, ky, k):
        return (1.0 - 1.0 / np.sqrt(1.0 + (k / k1) ** (2 * order))) / np.sqrt(1.0 + (k / k2) ** (2 * order))

    return _filter(g, dx, dy, resp)


# ── edge detectors ──
def _window_std(a, w):
    from scipy import ndimage as ndi
    m = ndi.uniform_filter(a, size=w, mode="nearest")
    m2 = ndi.uniform_filter(a * a, size=w, mode="nearest")
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def edges(g, dx, dy, window: int = 5) -> dict:
    """The derivative maps and edge detectors of a grid (see the module docstring):
    vdr (per m), thdr (per m), asa (per m), tilt (degrees), theta (0-1),
    tdr_thdr (degrees per km), nstd (0-1)."""
    mask = ~np.isfinite(np.asarray(g, dtype=float))
    gx, gy = horizontal_derivatives(g, dx, dy)
    gz = vertical_derivative(g, dx, dy)
    thdr = np.hypot(gx, gy)
    asa = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        tilt = np.degrees(np.arctan2(gz, thdr))
        theta = thdr / asa
    t = np.radians(tilt)
    tx, ty = np.gradient(_filled(t), dy, dx)[::-1]
    tdr_thdr = np.degrees(np.hypot(tx, ty)) * 1000.0
    sx, sy, sz = (_window_std(_filled(a), window) for a in (gx, gy, gz))
    with np.errstate(invalid="ignore", divide="ignore"):
        nstd = sz / (sx + sy + sz)
    out = {"vdr": gz, "thdr": thdr, "asa": asa, "tilt": tilt, "theta": theta,
           "tdr_thdr": tdr_thdr, "nstd": nstd}
    for a in out.values():
        a[mask] = np.nan
    return out


def nvdr(edge, dx, dy):
    """The positive part of a detector's vertical derivative, scaled to its maximum (0-1)."""
    d = vertical_derivative(edge, dx, dy)
    top = np.nanmax(d)
    out = np.where(d > 0, d / top if top > 0 else 0.0, 0.0)
    out[~np.isfinite(np.asarray(edge, dtype=float))] = np.nan
    return out
