"""3D model visualization and convergence plots."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..datamodel.mesh import Mesh3D
from ..datamodel.model import PhysicalModel
from ..datamodel.result import InversionResult


def plot_model_slice(
    model: PhysicalModel,
    axis: str = "z",
    index: int = 0,
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "viridis",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Plot a 2D slice through a 3D model.

    Args:
        model: The PhysicalModel to visualize.
        axis: Slice axis — "x", "y", or "z".
        index: Cell index along the slice axis.
        cmap: Matplotlib colormap name.
        title: Plot title (defaults to model name).
        save_path: If given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    vals = model.values_3d()
    mesh = model.mesh

    axis_map = {"x": 0, "y": 1, "z": 2}
    ax_idx = axis_map[axis.lower()]

    if ax_idx == 0:
        slice_data = vals[index, :, :]
        extent = [0, np.sum(mesh.hy), 0, np.sum(mesh.hz)]
        xlabel, ylabel = "Easting (m)", "Depth (m)"
    elif ax_idx == 1:
        slice_data = vals[:, index, :]
        extent = [0, np.sum(mesh.hx), 0, np.sum(mesh.hz)]
        xlabel, ylabel = "Northing (m)", "Depth (m)"
    else:
        slice_data = vals[:, :, index]
        extent = [0, np.sum(mesh.hx), 0, np.sum(mesh.hy)]
        xlabel, ylabel = "Northing (m)", "Easting (m)"

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(
        slice_data.T, origin="lower", aspect="auto",
        extent=extent, cmap=cmap,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"{model.prop.value} ({model.units})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{model.name} — {axis.upper()} slice #{index}")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_convergence(
    result: InversionResult,
    figsize: tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
):
    """Plot convergence curves (phi_d, phi_m) for an inversion result.

    Returns matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    iters = np.arange(result.n_iterations)

    ax1.semilogy(iters, result.phi_d_history, "o-", color="#F44336", label="φ_d")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Data Misfit (φ_d)")
    ax1.set_title("Data Misfit Convergence")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.semilogy(iters, result.phi_m_history, "s-", color="#2196F3", label="φ_m")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Model Norm (φ_m)")
    ax2.set_title("Regularization Convergence")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle(f"{result.method} Inversion Convergence", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_model_3d(model: PhysicalModel, threshold: Optional[float] = None):
    """Interactive 3D visualization using PyVista.

    Args:
        model: The PhysicalModel to visualize.
        threshold: If given, show only cells above this value.

    Returns:
        PyVista Plotter.
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista required for 3D viz: pip install pyvista")

    mesh = model.mesh
    dmesh = mesh.to_discretize()

    # Convert discretize mesh to PyVista
    grid = dmesh.to_vtk()
    grid.cell_data[model.prop.value] = model.values

    plotter = pv.Plotter()
    if threshold is not None:
        threshed = grid.threshold(threshold, scalars=model.prop.value)
        plotter.add_mesh(threshed, scalars=model.prop.value, cmap="viridis",
                         show_scalar_bar=True)
    else:
        plotter.add_mesh(grid, scalars=model.prop.value, cmap="viridis",
                         opacity=0.7, show_scalar_bar=True)
    plotter.add_axes()
    plotter.show_grid()

    return plotter
