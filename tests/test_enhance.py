"""Enhancement filters against analytic fields: reduction to the pole, continuation,
derivatives, edge detectors, separations, gridding."""

import numpy as np
import pytest

pytest.importorskip("scipy")
from geoinv3d.methods import enhance as E   # noqa: E402

DX = 50.0
X1 = np.arange(-6000, 6001, DX)
X, Y = np.meshgrid(X1, X1)


def dipole_tmi(depth, inc, dec, x0=0.0, y0=0.0):
    """Total-field anomaly (nT, arbitrary moment) of an induced point dipole at depth."""
    i, d = np.radians(inc), np.radians(dec)
    f = np.array([np.cos(i) * np.sin(d), np.cos(i) * np.cos(d), np.sin(i)])   # x east, y north, z down
    rx, ry, rz = X - x0, Y - y0, np.full_like(X, -depth)
    r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    mr = f[0] * rx + f[1] * ry + f[2] * rz
    B = [(3 * mr * c / r ** 2 - fc) / r ** 3 for c, fc in zip((rx, ry, rz), f)]
    return 1e9 * sum(fc * b for fc, b in zip(f, B))


def point_gz(depth):
    r = np.sqrt(X ** 2 + Y ** 2 + depth ** 2)
    return depth / r ** 3


@pytest.mark.parametrize("inc, dec", [(60, -10), (45, 20)])
def test_reduction_to_the_pole(inc, dec):
    pole = dipole_tmi(500, 90, 0)
    r = E.rtp(dipole_tmi(500, inc, dec), DX, DX, inc, dec)
    assert np.corrcoef(r.ravel(), pole.ravel())[0, 1] > 0.999
    assert np.unravel_index(np.argmax(r), r.shape) == np.unravel_index(np.argmax(pole), pole.shape)
    assert r.max() == pytest.approx(pole.max(), rel=0.02)


def test_reduction_to_the_pole_is_damped_at_low_latitude():
    pole = dipole_tmi(500, 90, 0)
    r = E.rtp(dipole_tmi(500, 20, 5), DX, DX, 20, 5)
    assert np.isfinite(r).all() and np.corrcoef(r.ravel(), pole.ravel())[0, 1] > 0.97


def test_upward_continuation_and_vertical_derivative():
    g = point_gz(800)
    up = E.upward(g, DX, DX, 300)
    assert np.max(np.abs(up - point_gz(1100))) < 0.01 * point_gz(1100).max()
    R2 = X ** 2 + Y ** 2 + 800 ** 2
    analytic = -(1 / R2 ** 1.5 - 3 * 800 ** 2 / R2 ** 2.5)     # d/dz (z down) of the point mass
    v = E.vertical_derivative(g, DX, DX)
    assert np.corrcoef(v.ravel(), analytic.ravel())[0, 1] > 0.9999
    assert v.max() == pytest.approx(analytic.max(), rel=0.01)


def test_edges_of_a_source():
    out = E.edges(point_gz(600), DX, DX)
    centre = np.unravel_index(np.argmin(X ** 2 + Y ** 2), X.shape)
    assert out["tilt"][centre] == pytest.approx(90, abs=1)      # straight over the source
    assert np.nanmin(out["tilt"]) >= -90 and np.nanmax(out["tilt"]) <= 90
    assert np.nanmin(out["theta"]) >= 0 and np.nanmax(out["theta"]) <= 1 + 1e-9
    assert np.nanmin(out["nstd"]) >= 0 and np.nanmax(out["nstd"]) <= 1
    # the tilt crosses zero where the vertical derivative changes sign: for a point mass
    # at depth d, at r = sqrt(2) d
    row = out["tilt"][centre[0]]
    xs = X1[np.where(np.diff(np.sign(row)) != 0)[0]]
    assert np.min(np.abs(xs)) == pytest.approx(600 * np.sqrt(2), abs=2 * DX)
    n = E.nvdr(out["thdr"], DX, DX)
    assert np.nanmin(n) == 0 and np.nanmax(n) == pytest.approx(1)


def test_separations():
    """Each separation recovers the broad field and leaves the local peak where it is; upward
    continuation also weakens the broad field itself (its known drawback as a regional)."""
    broad = 50 * np.exp(-((X - 2000) ** 2 + Y ** 2) / (2 * 5000 ** 2))
    local = point_gz(300) / point_gz(300).max() * 20
    g = broad + local
    centre = np.unravel_index(np.argmax(local), local.shape)
    regs = {"butterworth": E.butterworth(g, DX, DX, 3000), "upward": E.upward(g, DX, DX, 1500),
            "bandpass": g - E.bandpass(g, DX, DX, 2 * DX, 3000)}
    for name, reg in regs.items():
        assert np.corrcoef(reg.ravel(), broad.ravel())[0, 1] > 0.98, name
        assert np.unravel_index(np.argmax(g - reg), g.shape) == centre, name
    err = {k: np.sqrt(np.mean((r - broad) ** 2)) / broad.std() for k, r in regs.items()}
    assert err["butterworth"] < 0.1 and err["upward"] > 0.3


def test_gaps_stay_gaps():
    g = point_gz(800)
    g[:20, :20] = np.nan
    v = E.vertical_derivative(g, DX, DX)
    assert np.isnan(v[:20, :20]).all() and np.isfinite(v[30:, 30:]).all()


def test_to_grid():
    xs, ys = np.meshgrid(np.arange(0, 1000, 100.0), np.arange(0, 800, 100.0))
    v = xs + 2 * ys
    xg, yg, g = E.to_grid(xs.ravel(), ys.ravel(), v.ravel())        # a lattice: placed as it is
    assert g.shape == (8, 10) and np.allclose(g, v)
    rng = np.random.default_rng(0)
    px, py = rng.uniform(0, 1000, 400), rng.uniform(0, 1000, 400)
    xg, yg, g = E.to_grid(px, py, px + 2 * py)                       # scattered: interpolated
    X2, Y2 = np.meshgrid(xg, yg)
    ok = np.isfinite(g)
    assert ok.mean() > 0.8 and np.allclose(g[ok], (X2 + 2 * Y2)[ok], atol=1e-6)
    s = E.sample(xg, yg, g, [500.0], [500.0])
    assert s[0] == pytest.approx(1500, abs=1)
