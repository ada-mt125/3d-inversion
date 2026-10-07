"""An MT dataset from EDI files: stations on one frequency list, in the mesh's frame.

The EDI files are in the geographic frame of MT (x north, y east, z down); the mesh and
SimPEG are x east, y north, z up.  Swapping the horizontal axes and turning z over is a proper
rotation, so the impedance keeps its values with its indices swapped and the tipper changes sign:

    EDI Zxy -> SimPEG Zyx      EDI Zyx -> SimPEG Zxy      EDI Zxx <-> SimPEG Zyy
    EDI Tzx -> -SimPEG Tzy     EDI Tzy -> -SimPEG Tzx

(both use e^{+i omega t}: over a halfspace SimPEG's Zyx is in the first quadrant, as EDI's Zxy).

The data are ``values[f, c, s]`` (frequency, component, station) with NaN where a station has
no datum, in SimPEG's component names (``"xy_real"``, ``"yx_rho"``, ``"zx_imag"``...).
"""

from __future__ import annotations

import numpy as np

from .edi import MU0, EDIStation, read_edi

# MT (EDI) element -> (row, column) in the EDI tensor, for each SimPEG orientation
_SIMPEG_FROM_EDI = {"xy": (1, 0), "yx": (0, 1), "xx": (1, 1), "yy": (0, 0)}
_TIPPER_FROM_EDI = {"zx": 1, "zy": 0}                 # SimPEG Tzx = -EDI Tzy, Tzy = -EDI Tzx


def common_frequencies(freq_lists, per_decade: float | None = None, tol: float = 0.02):
    """One frequency list for every station: the distinct frequencies (within ``tol``,
    relative), or ``per_decade`` log-spaced ones over the whole band."""
    allf = np.sort(np.concatenate([np.asarray(f, float) for f in freq_lists]))
    allf = allf[np.isfinite(allf) & (allf > 0)]
    if allf.size == 0:
        raise ValueError("No frequencies in the EDI files")
    if per_decade:
        lo, hi = np.log10(allf[0]), np.log10(allf[-1])
        n = max(int(round((hi - lo) * per_decade)) + 1, 1)
        return np.logspace(lo, hi, n)
    out = [allf[0]]
    for f in allf[1:]:
        if np.log(f / out[-1]) > np.log1p(tol):
            out.append(f)
    return np.array(out)


def _match(target, freq, per_decade):
    """Index into ``freq`` of the station's frequency for each target (-1: none near)."""
    half = 0.5 / per_decade if per_decade else np.log10(1.02)
    lf, lt = np.log10(freq), np.log10(target)
    idx = np.abs(lt[:, None] - lf[None, :]).argmin(axis=1)
    ok = np.abs(lt - lf[idx]) <= half + 1e-12
    return np.where(ok, idx, -1)


def mt_dataset(stations: list, xy: np.ndarray, *, impedance: str = "offdiagonal", tipper: bool = True,
               data_type: str = "impedance", error_floor: float = 0.05, tipper_floor: float = 0.03,
               frequency_range=None, per_decade: float | None = None) -> dict:
    """Arrays of an MT dataset from ``stations`` (EDIStation) at ``xy`` (n, 2, the mesh's
    easting and northing).  ``impedance``: "offdiagonal", "full" or "none"; ``data_type``:
    "impedance" (real and imaginary parts) or "rho_phase" (apparent resistivity in ohm m,
    phase in degrees); ``error_floor``: share of sqrt|Zxy Zyx| below which no impedance error
    goes (and its equivalents for rho and phase); ``tipper_floor``: the tipper's, absolute."""
    if impedance not in ("offdiagonal", "full", "none"):
        raise ValueError(f"impedance must be offdiagonal, full or none, got {impedance!r}")
    if data_type not in ("impedance", "rho_phase"):
        raise ValueError(f"data_type must be impedance or rho_phase, got {data_type!r}")
    freqs = common_frequencies([s.freq for s in stations], per_decade)
    if frequency_range is not None:
        lo, hi = (float(v) for v in frequency_range)
        freqs = freqs[(freqs >= lo * 0.999) & (freqs <= hi * 1.001)]
    if freqs.size == 0:
        raise ValueError("No frequencies left in the frequency range")
    has_tipper = tipper and any(s.t is not None and np.isfinite(s.t).any() for s in stations)
    orients = {"offdiagonal": ["xy", "yx"], "full": ["xx", "xy", "yx", "yy"], "none": []}[impedance]
    parts = ("real", "imag") if data_type == "impedance" else ("rho", "phase")
    comps = [f"{o}_{p}" for o in orients for p in parts]
    if has_tipper:
        comps += [f"{o}_{p}" for o in ("zx", "zy") for p in ("real", "imag")]
    if not comps:
        raise ValueError("Neither impedance nor tipper data chosen")
    nf, nc, ns = freqs.size, len(comps), len(stations)
    values = np.full((nf, nc, ns), np.nan)
    std = np.full((nf, nc, ns), np.nan)
    w = 2 * np.pi * freqs
    for k, st in enumerate(stations):
        idx = _match(freqs, st.freq, per_decade)
        have = idx >= 0
        z = np.full((nf, 2, 2), complex(np.nan, np.nan))
        z[have] = st.z[idx[have]]
        zs = np.full((nf, 2, 2), np.nan)
        if st.z_std is not None:
            zs[have] = st.z_std[idx[have]]
        floor_z = error_floor * np.sqrt(np.abs(z[:, 0, 1] * z[:, 1, 0]))          # per frequency
        for c, name in enumerate(comps):
            o, p = name.split("_", 1)
            if o in _SIMPEG_FROM_EDI:
                i, j = _SIMPEG_FROM_EDI[o]
                zij = z[:, i, j]
                s = np.fmax(zs[:, i, j], floor_z)                    # NaN std: the floor
                if p in ("real", "imag"):
                    values[:, c, k] = zij.real if p == "real" else zij.imag
                    std[:, c, k] = s
                else:
                    rel = s / np.abs(zij)
                    if p == "rho":
                        rho = np.abs(zij) ** 2 / (w * MU0)
                        values[:, c, k], std[:, c, k] = rho, 2 * rho * rel
                    else:
                        values[:, c, k] = np.degrees(np.angle(zij))
                        std[:, c, k] = np.degrees(np.minimum(rel, 1.0))
            else:
                if st.t is None:
                    continue
                tt = np.full(nf, complex(np.nan, np.nan))
                tt[have] = -st.t[idx[have], _TIPPER_FROM_EDI[o]]
                ts = np.full(nf, np.nan)
                if st.t_std is not None:
                    ts[have] = st.t_std[idx[have], _TIPPER_FROM_EDI[o]]
                values[:, c, k] = tt.real if p == "real" else tt.imag
                std[:, c, k] = np.fmax(ts, tipper_floor)
    bad = ~np.isfinite(values) | ~np.isfinite(std) | (std <= 0)
    values[bad] = np.nan
    std[bad] = np.nan
    locations = np.column_stack([np.asarray(xy, float), [s.elev for s in stations]])
    return {"locations": locations, "names": [s.name for s in stations], "frequencies": freqs,
            "components": comps, "values": values, "std": std}


def read_edi_files(paths, units: str = "field") -> list[EDIStation]:
    stations = [read_edi(p, units=units) for p in paths]
    if len({s.name for s in stations}) < len(stations):         # repeated DATAIDs: the file names
        from pathlib import Path
        for s, p in zip(stations, paths):
            s.name = Path(p).stem
    return stations
