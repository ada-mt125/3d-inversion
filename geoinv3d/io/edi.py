"""EDI files (SEG's exchange format for MT): the impedance tensor and the tipper per station.

``read_edi`` returns one station: its position, frequencies, impedance Z (2 x 2, complex, in
ohms) and tipper T (Tzx, Tzy), with their standard deviations, in the geographic frame of MT:
x north, y east, z down, and the e^{+i omega t} convention of the files (Zxy in the first
quadrant over a layered earth).  Rotated data (ZROT, TROT) are rotated back to north.  The
impedance is in the files' field units, mV/km per nT, unless ``units="ohm"``.

Only impedance sections (``>=MTSECT``) are read: spectra (``>=SPECTRASECT``) must be turned into
impedances first, e.g. by the processing software.  ``write_edi`` writes the same fields back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

MU0 = 4e-7 * np.pi
FIELD_TO_OHM = 4e-4 * np.pi        # mV/km/nT -> ohm (V/m per A/m)
EMPTY = 1.0e32


@dataclass
class EDIStation:
    name: str
    lat: float
    lon: float
    elev: float
    freq: np.ndarray                  # (n,) Hz
    z: np.ndarray                     # (n, 2, 2) complex, ohm; NaN where missing
    z_std: Optional[np.ndarray]       # (n, 2, 2) ohm, or None
    t: Optional[np.ndarray] = None    # (n, 2) complex (Tzx, Tzy), or None
    t_std: Optional[np.ndarray] = None

    def rho_phase(self):
        """Apparent resistivity (ohm m) and phase (degrees) of each element of Z."""
        w = 2 * np.pi * self.freq[:, None, None]
        return np.abs(self.z) ** 2 / (w * MU0), np.degrees(np.angle(self.z))


def _angle(text: str) -> float:
    """Degrees from ``dd:mm:ss.s`` (sign on the degrees) or a decimal number."""
    text = text.strip().strip('"')
    if ":" not in text:
        return float(text)
    parts = [float(p) for p in text.split(":")]
    sign = -1.0 if text.startswith("-") else 1.0
    return sign * (abs(parts[0]) + (parts[1] if len(parts) > 1 else 0) / 60 + (parts[2] if len(parts) > 2 else 0) / 3600)


def _blocks(text: str):
    """(header line, the numbers or key=value text after it) for each '>' block."""
    out, head, body = [], None, []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(">"):
            if head is not None:
                out.append((head, body))
            head, body = s[1:].strip(), []
        elif head is not None and s:
            body.append(s)
    if head is not None:
        out.append((head, body))
    return out


def _keyvals(lines) -> dict:
    kv = {}
    for line in lines:
        for m in re.finditer(r"([A-Za-z_][\w.]*)\s*=\s*(\"[^\"]*\"|\S+)", line):
            kv[m.group(1).upper()] = m.group(2).strip('"')
    return kv


def _rotation(theta_deg):
    """R(theta) of a frame turned clockwise (north towards east) by theta: v' = R v."""
    t = np.radians(np.asarray(theta_deg, dtype=float))
    c, s = np.cos(t), np.sin(t)
    return np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -2)     # (..., 2, 2)


def rotate(z: np.ndarray, t: Optional[np.ndarray], theta_deg) -> tuple:
    """Z and T in a frame turned clockwise by theta: Z' = R Z R^T, T' = T R^T."""
    r = _rotation(np.broadcast_to(theta_deg, z.shape[:-2]))
    z2 = r @ z @ np.swapaxes(r, -1, -2)
    t2 = None if t is None else np.einsum("...j,...ij->...i", t, r)
    return z2, t2


def read_edi(path, units: str = "field") -> EDIStation:
    """One station of an EDI file (see the module docstring)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = _blocks(text)
    sections = {h.split()[0].upper(): b for h, b in blocks}
    if "=MTSECT" not in sections and "=SPECTRASECT" in sections:
        raise ValueError(f"{path.name}: spectra (>=SPECTRASECT) only; export the impedances first")
    head = _keyvals(sections.get("HEAD", []))
    meas = _keyvals(sections.get("=DEFINEMEAS", []))
    empty = float(head.get("EMPTY", EMPTY))

    def coord(*keys):
        for k in keys:
            for kv in (head, meas):
                if k in kv:
                    return _angle(kv[k])
        return np.nan

    lat, lon = coord("LAT", "REFLAT"), coord("LONG", "LON", "REFLONG", "REFLON")
    elev = coord("ELEV", "REFELEV")
    name = head.get("DATAID") or path.stem

    data = {}
    for h, body in blocks:
        key = h.split()[0].upper()
        if key.startswith("=") or key in ("HEAD", "INFO", "END"):
            continue
        nums = np.array([float(v) for line in body for v in line.replace(",", " ").split()
                         if re.match(r"^[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?$", v)], dtype=float)
        nums[np.abs(nums) >= 0.99 * empty] = np.nan
        data[key] = nums
    if "FREQ" not in data:
        raise ValueError(f"{path.name}: no >FREQ block")
    freq = data["FREQ"]
    n = freq.size
    scale = FIELD_TO_OHM if units == "field" else 1.0

    def arr(key):
        a = data.get(key)
        return None if a is None or a.size != n else a

    z = np.full((n, 2, 2), complex(np.nan, np.nan))
    z_var = np.full((n, 2, 2), np.nan)
    have_var = False
    for i, a in enumerate("XY"):
        for j, b in enumerate("XY"):
            re_, im_ = arr(f"Z{a}{b}R"), arr(f"Z{a}{b}I")
            if re_ is not None and im_ is not None:
                z[:, i, j] = (re_ + 1j * im_) * scale
            var = arr(f"Z{a}{b}.VAR")
            if var is not None:
                z_var[:, i, j] = var * scale ** 2
                have_var = True
    t = t_var = None
    if arr("TXR.EXP") is not None or arr("TYR.EXP") is not None:
        t = np.full((n, 2), complex(np.nan, np.nan))
        t_var = np.full((n, 2), np.nan)
        for k, a in enumerate("XY"):
            re_, im_ = arr(f"T{a}R.EXP"), arr(f"T{a}I.EXP")
            if re_ is not None and im_ is not None:
                t[:, k] = re_ + 1j * im_
            var = arr(f"T{a}VAR.EXP")
            if var is not None:
                t_var[:, k] = var
    # an element missing either part is missing
    z[~(np.isfinite(z.real) & np.isfinite(z.imag))] = complex(np.nan, np.nan)
    if t is not None:
        t[~(np.isfinite(t.real) & np.isfinite(t.imag))] = complex(np.nan, np.nan)
    # data in a turned frame: back to north (each frequency its own angle)
    zrot = arr("ZROT")
    if zrot is not None and np.any(np.nan_to_num(zrot) != 0):
        back = -np.nan_to_num(zrot)
        z, _ = rotate(z, None, back)
        z_var, _ = rotate(z_var.astype(complex), None, back)      # approximate: the largest stays
        z_var = np.abs(z_var.real)
    trot = arr("TROT") if arr("TROT") is not None else zrot
    if t is not None and trot is not None and np.any(np.nan_to_num(trot) != 0):
        _, t = rotate(np.zeros((n, 2, 2), complex), t, -np.nan_to_num(trot))
    order = np.argsort(freq)
    pick = lambda a: None if a is None else a[order]        # noqa: E731
    return EDIStation(name=str(name), lat=float(lat), lon=float(lon), elev=float(elev),
                      freq=freq[order], z=z[order],
                      z_std=np.sqrt(z_var[order]) if have_var else None,
                      t=pick(t), t_std=None if t_var is None else np.sqrt(t_var[order]))


def _fmt_block(values) -> str:
    v = np.where(np.isfinite(values), values, EMPTY)
    return "\n".join(" ".join(f"{x: .6E}" for x in v[i:i + 6]) for i in range(0, len(v), 6))


def _dms(deg: float) -> str:
    sign = "-" if deg < 0 else ""
    d = abs(deg)
    return f"{sign}{int(d)}:{int(d * 60) % 60:02d}:{(d * 3600) % 60:05.2f}"


def write_edi(path, st: EDIStation, units: str = "field") -> None:
    """Write a station as an EDI file (impedance and, if any, tipper; north-east frame)."""
    scale = 1 / FIELD_TO_OHM if units == "field" else 1.0
    n = st.freq.size
    lines = [">HEAD", f'  DATAID="{st.name}"', f"  LAT={_dms(st.lat)}", f"  LONG={_dms(st.lon)}",
             f"  ELEV={st.elev:.2f}", f"  EMPTY={EMPTY:.1E}", "", ">=DEFINEMEAS", "  MAXCHAN=5",
             f"  REFLAT={_dms(st.lat)}", f"  REFLONG={_dms(st.lon)}", f"  REFELEV={st.elev:.2f}", "",
             ">=MTSECT", f'  SECTID="{st.name}"', f"  NFREQ={n}", "", f">FREQ //{n}", _fmt_block(st.freq),
             f">ZROT //{n}", _fmt_block(np.zeros(n))]
    for i, a in enumerate("XY"):
        for j, b in enumerate("XY"):
            zz = st.z[:, i, j] * scale
            lines += [f">Z{a}{b}R ROT=ZROT //{n}", _fmt_block(zz.real), f">Z{a}{b}I ROT=ZROT //{n}",
                      _fmt_block(zz.imag)]
            if st.z_std is not None:
                lines += [f">Z{a}{b}.VAR ROT=ZROT //{n}", _fmt_block((st.z_std[:, i, j] * scale) ** 2)]
    if st.t is not None:
        lines += [f">TROT //{n}", _fmt_block(np.zeros(n))]
        for k, a in enumerate("XY"):
            lines += [f">T{a}R.EXP ROT=TROT //{n}", _fmt_block(st.t[:, k].real),
                      f">T{a}I.EXP ROT=TROT //{n}", _fmt_block(st.t[:, k].imag)]
            if st.t_std is not None:
                lines += [f">T{a}VAR.EXP ROT=TROT //{n}", _fmt_block(st.t_std[:, k] ** 2)]
    lines += [">END", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
