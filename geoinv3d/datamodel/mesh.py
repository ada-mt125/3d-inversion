"""3D rectilinear mesh definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Mesh3D:
    """A 3D rectilinear (tensor) mesh.

    Wraps discretize.TensorMesh but keeps geometry as plain arrays
    so the datamodel stays independent of discretize at import time.

    Attributes:
        hx: Cell widths in x (Northing). Shape (nx,).
        hy: Cell widths in y (Easting).  Shape (ny,).
        hz: Cell widths in z (Depth, positive down). Shape (nz,).
        origin: (x0, y0, z0) corner of the mesh.
        name: Optional label.
    """

    hx: NDArray[np.float64]
    hy: NDArray[np.float64]
    hz: NDArray[np.float64]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "hx", np.asarray(self.hx, dtype=np.float64))
        object.__setattr__(self, "hy", np.asarray(self.hy, dtype=np.float64))
        object.__setattr__(self, "hz", np.asarray(self.hz, dtype=np.float64))
        self.hx.setflags(write=False)
        self.hy.setflags(write=False)
        self.hz.setflags(write=False)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.hx), len(self.hy), len(self.hz))

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.shape))

    def to_discretize(self):
        """Convert to a discretize.TensorMesh."""
        from discretize import TensorMesh
        return TensorMesh([self.hx.copy(), self.hy.copy(), self.hz.copy()],
                          origin=self.origin)

    @classmethod
    def from_discretize(cls, mesh, name: str = "") -> Mesh3D:
        """Build from a discretize.TensorMesh."""
        return cls(
            hx=np.array(mesh.h[0]),
            hy=np.array(mesh.h[1]),
            hz=np.array(mesh.h[2]),
            origin=tuple(mesh.origin),
            name=name,
        )

    @classmethod
    def uniform(
        cls,
        nx: int, ny: int, nz: int,
        dx: float = 100.0, dy: float = 100.0, dz: float = 100.0,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        name: str = "",
    ) -> Mesh3D:
        """Create a uniform mesh with constant cell sizes."""
        return cls(
            hx=np.full(nx, dx),
            hy=np.full(ny, dy),
            hz=np.full(nz, dz),
            origin=origin,
            name=name,
        )

    @classmethod
    def padded(
        cls,
        nx_core: int, ny_core: int, nz_core: int,
        dx: float, dy: float, dz: float,
        n_pad: int = 5,
        pad_factor: float = 1.3,
        x0: float = 0.0,
        y0: float = 0.0,
        name: str = "",
    ) -> Mesh3D:
        """Create a mesh with uniform core + expanding padding cells.

        Padding cells expand outward on all horizontal sides and the
        bottom. Origin is computed so the core starts at (x0, y0) and
        the mesh bottom is below z=0.

        Args:
            nx_core, ny_core, nz_core: Core cell counts.
            dx, dy, dz: Core cell sizes.
            n_pad: Number of padding cells per side.
            pad_factor: Geometric expansion factor.
            x0, y0: South-west corner of the core domain.
        """
        def _padded_h(n_core, dh, n_pad, factor):
            core = np.full(n_core, dh)
            pad = dh * factor ** np.arange(1, n_pad + 1)
            return np.concatenate([pad[::-1], core, pad])

        hx = _padded_h(nx_core, dx, n_pad, pad_factor)
        hy = _padded_h(ny_core, dy, n_pad, pad_factor)

        core_z = np.full(nz_core, dz)
        pad_z = dz * pad_factor ** np.arange(1, n_pad + 1)
        hz = np.concatenate([pad_z[::-1], core_z])

        ox = x0 - np.sum(hx[:n_pad])
        oy = y0 - np.sum(hy[:n_pad])
        oz = -np.sum(hz)

        return cls(hx=hx, hy=hy, hz=hz, origin=(ox, oy, oz), name=name)


class DiscretizeMesh:
    """Adapter for a prebuilt discretize mesh (e.g. an octree ``TreeMesh``).

    Exposes the parts of the Mesh3D interface the methods rely on
    (``to_discretize``, ``n_cells``), so non-tensor meshes can be used
    wherever a Mesh3D is expected.
    """

    def __init__(self, mesh, name: str = "") -> None:
        self._mesh = mesh
        self.name = name

    @property
    def n_cells(self) -> int:
        return int(self._mesh.n_cells)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.n_cells,)

    @property
    def origin(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self._mesh.origin)

    def to_discretize(self):
        """Return the wrapped discretize mesh."""
        return self._mesh
