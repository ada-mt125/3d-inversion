"""Additional regularizations for SimPEG inversions.

L1–L2 (elastic net) regularization
----------------------------------
Utsugi (2019, Earth Planets Space 71:73) regularizes magnetic inversion with
the elastic net of Zou & Hastie (2005), as restated by Nwosu & Becken (2025,
GJI 243, ggaf390):

    phi_m = (1 - a)/2 * ||x~||_2^2 + a * ||x~||_1,
    x~ = W^-1 (m - m_ref),   W = diag(||g_j||^(-1/2))

where ``a`` (``l1_ratio``) mixes sparsity (a = 1, pure L1: compact bodies)
and stability (a = 0, pure L2 smallness: blurred) and ``g_j`` is column j of
the sensitivity matrix, so ``x~_j = s_j (m_j - m_ref_j)`` with
``s_j = ||g_j||^(1/2)`` (depth weighting).

In SimPEG's convention (no 1/2 factor, cell volumes ``v_j``) the objective
implemented here is

    phi_m = sum_j v_j [ 2a s_j sqrt(f_j^2 + eps^2) + (1 - a) s_j^2 f_j^2 ],
    f = m - m_ref,

i.e. twice the paper's form, which only rescales beta.  The L1 term is
minimised by majorize-minimize / IRLS: at the current model f_k the L1 term
is replaced by the quadratic ``a s_j f_j^2 / sqrt(f_kj^2 + eps^2)``, which
touches it at f_k and has the same gradient.  SimPEG's ``UpdateIRLS``
directive drives the reweighting and cools ``eps`` (``irls_threshold``);
``ElasticNetSensitivityWeights`` supplies ``s_j^2 = ||g_j||``.

The balance set by ``a`` depends on the size of ``x~``: the L1 and L2 terms
are equal where ``|x~| = 2a / (1 - a)``.  s_j is left unnormalized (data
units per model unit), as in the paper.
"""

from __future__ import annotations

import numpy as np
from simpeg.regularization import RegularizationMesh, Sparse, SparseSmallness


class ElasticNetSmallness(SparseSmallness):
    """Elastic-net (L1–L2) smallness term; see the module docstring.

    Cell weights other than ``volume`` and ``irls`` (e.g. ``sensitivity``)
    multiply to ``s_j^2``.  The IRLS weights set by :meth:`update_weights`
    turn SimPEG's least-squares form ``||W f||^2`` into the majorizer of the
    elastic net at the current model.

    ``norm`` is 1 for the elastic net; ``UpdateIRLS`` temporarily sets it to
    2 for its L2 warm-up stage, during which the term is plain L2 smallness.
    """

    def __init__(self, mesh, l1_ratio: float = 0.5, **kwargs):
        kwargs.setdefault("norm", 1.0)
        super().__init__(mesh, **kwargs)
        self.l1_ratio = l1_ratio

    @property
    def l1_ratio(self) -> float:
        """Mixing parameter a in [0, 1]: 1 = pure L1, 0 = pure L2."""
        return self._l1_ratio

    @l1_ratio.setter
    def l1_ratio(self, value: float):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"l1_ratio must be in [0, 1], got {value}")
        self._l1_ratio = value

    @property
    def norm(self):
        return self._norm

    @norm.setter
    def norm(self, value):
        if value is None:
            value = 1.0
        value = np.atleast_1d(np.asarray(value, dtype=float))
        if not (np.all(value == 1.0) or np.all(value == 2.0)):
            raise ValueError("ElasticNetSmallness norm must be 1 (elastic net) or 2 (L2 stage)")
        self._norm = np.full(self.regularization_mesh.nC, value[0])

    def _weight_product(self, exclude: tuple[str, ...]) -> np.ndarray:
        weights = [w for k, w in self._weights.items() if k not in exclude]
        if not weights:
            return np.ones(self.regularization_mesh.nC)
        return np.prod(weights, axis=0)

    def depth_scaling(self) -> np.ndarray:
        """s_j: square root of all cell weights except volume and IRLS."""
        s = np.sqrt(self._weight_product(exclude=("volume", "irls")))
        return np.maximum(s, 1e-12 * max(float(s.max()), 1e-300))

    def update_weights(self, m):
        """Set IRLS weights so ||W f||^2 majorizes the elastic net at ``m``."""
        f = self.f_m(m)
        if np.all(self.norm == 2.0):
            q = np.ones_like(f)
        else:
            a = self.l1_ratio
            s = self.depth_scaling()
            q = a / (s * np.sqrt(f**2 + self.irls_threshold**2)) + (1.0 - a)
        self.set_weights(irls=q)

    def elastic_net_value(self, m) -> float:
        """The elastic-net objective itself (not its quadratic majorizer)."""
        f = self.f_m(m)
        a, s = self.l1_ratio, self.depth_scaling()
        v = self._weights.get("volume", np.ones_like(f))
        l1 = np.sqrt(f**2 + self.irls_threshold**2)
        return float(np.sum(v * (2.0 * a * s * l1 + (1.0 - a) * s**2 * f**2)))


class ElasticNet(Sparse):
    """L1–L2 (elastic net) regularization of the model values (Utsugi 2019).

    A :class:`simpeg.regularization.Sparse` with a single
    :class:`ElasticNetSmallness` term, so SimPEG's ``UpdateIRLS`` directive
    runs it: an L2 warm-up to ``chifact_start``, then IRLS steps that cool
    ``irls_threshold`` (eps).  There are no smoothness terms, as in the paper.

    Args:
        mesh: discretize mesh or RegularizationMesh.
        l1_ratio: Mixing parameter a in [0, 1].
        active_cells: Optional active-cell mask.
        **kwargs: Passed to ``Sparse`` (e.g. reference_model, mapping).
    """

    def __init__(self, mesh, l1_ratio: float = 0.5, active_cells=None, **kwargs):
        if not isinstance(mesh, RegularizationMesh):
            mesh = RegularizationMesh(mesh)
        if active_cells is not None:
            mesh.active_cells = active_cells
        smallness = ElasticNetSmallness(mesh, l1_ratio=l1_ratio)
        kwargs.setdefault("irls_scaled", False)
        super().__init__(mesh, objfcts=[smallness], norms=[1.0], **kwargs)

    @property
    def l1_ratio(self) -> float:
        return self.objfcts[0].l1_ratio

    @l1_ratio.setter
    def l1_ratio(self, value: float):
        self.objfcts[0].l1_ratio = value

    def elastic_net_value(self, m) -> float:
        return self.objfcts[0].elastic_net_value(m)


# ── Focusing: minimum gradient support and total variation ──────────────


class Focusing(Sparse):
    """Minimum gradient support (MGS) or total variation (TV) regularization.

    Both stabilizers act on the isotropic gradient magnitude at cell centres,
    ``s_c = sum_axis A_axis (G_axis m)^2`` (face gradients squared and
    averaged to cells):

    * MGS (Portniaguine & Zhdanov 1999, 2002):
      ``S(m) = sum_c v_c e^2 s_c / (s_c + e^2)``, which counts the cells where
      the gradient exceeds the focusing parameter e: blocky, sharp-edged
      models.
    * TV (Rudin et al. 1992): ``S(m) = sum_c v_c 2 e sqrt(s_c + e^2)``,
      an L1 norm of the gradient: piecewise-constant models, less aggressive.

    (Scaled so that both equal ``sum v s`` for small gradients, like L2
    smoothness.)  Both are concave in ``s``, so linearizing in ``s`` at the
    current model gives a quadratic majorizer, i.e. first-order smoothness
    with cell weights ``w = S'(s)``: ``e^4 / (s + e^2)^2`` (MGS) or
    ``e / sqrt(s + e^2)`` (TV).  They are passed to the x/y/z smoothness terms
    as face weights ``A^T (v w) / (A_cc2f v)``, so the quadratic form is
    exactly ``sum_c v_c w_c s_c`` (times alpha).  SimPEG's ``UpdateIRLS``
    (or :class:`DampedUpdateIRLS`) drives it: an L2 warm-up, then reweighting.

    The focusing parameter e is set once, when IRLS starts, to
    ``threshold_scale`` times the ``threshold_percentile`` percentile of
    ``sqrt(s)`` of the L2 model (scale 1 for MGS, where e sets which
    gradients count as edges; 0.1 for TV, where e is only a smoothing
    constant and TV acts as an L1 norm for gradients above it), and then
    kept fixed (as in Zhdanov's
    method; the directive's eps cooling does not apply).  The smallness term
    stays L2 (weight ``alpha_s``).

    Args:
        mesh: discretize mesh or RegularizationMesh.
        stabilizer: "mgs" or "tv".
        threshold_percentile, threshold_scale: set e (see above);
            ``threshold_scale=None`` takes the stabilizer's default.
        **kwargs: Passed to ``Sparse`` (active_cells, alpha_s, length scales,
            reference_model, ...); ``norms`` is managed here.
    """

    STABILIZERS = ("mgs", "tv")
    DEFAULT_SCALE = {"mgs": 1.0, "tv": 0.1}

    def __init__(self, mesh, stabilizer: str = "mgs", threshold_percentile: float = 95.0,
                 threshold_scale: float | None = None, **kwargs):
        if stabilizer not in self.STABILIZERS:
            raise ValueError(f"stabilizer must be one of {self.STABILIZERS}, "
                             f"got {stabilizer!r}")
        # The smoothness "norm" only marks the stage: 2 = L2 warm-up, 1 = focusing
        kwargs["norms"] = [2.0, 1.0, 1.0, 1.0]
        kwargs.setdefault("irls_scaled", False)
        super().__init__(mesh, **kwargs)
        self.stabilizer = stabilizer
        self.threshold_percentile = threshold_percentile
        self.threshold_scale = (self.DEFAULT_SCALE[stabilizer] if threshold_scale is None
                                else float(threshold_scale))
        self.focusing_threshold: float | None = None

    @property
    def smoothness_terms(self) -> list:
        from simpeg.regularization import SmoothnessFirstOrder
        return [o for o in self.objfcts if isinstance(o, SmoothnessFirstOrder)]

    def _cell_average(self, orientation: str):
        return getattr(self.regularization_mesh, f"aveF{orientation}2CC")

    def gradient_magnitude2(self, m) -> np.ndarray:
        """s_c: squared isotropic gradient magnitude at the cell centres."""
        s = np.zeros(self.regularization_mesh.nC)
        for term in self.smoothness_terms:
            s += self._cell_average(term.orientation) @ term.f_m(m) ** 2
        return s

    def _rho(self, s, e2):
        if self.stabilizer == "mgs":
            return e2 * s / (s + e2)
        return 2.0 * np.sqrt(e2) * np.sqrt(s + e2)

    def cell_weights(self, s, e2) -> np.ndarray:
        """w = dS/ds per cell (1 for a vanishing gradient)."""
        if self.stabilizer == "mgs":
            return e2**2 / (s + e2) ** 2
        return np.sqrt(e2 / (s + e2))

    def stabilizer_value(self, m) -> float:
        """The MGS / TV stabilizer itself (not its quadratic majorizer).

        Includes each smoothness term's alpha (equal for all axes when the
        length scales are equal).
        """
        if self.focusing_threshold is None:
            raise ValueError("The focusing threshold is set when IRLS starts")
        e2 = self.focusing_threshold**2
        v = self.regularization_mesh.vol
        return float(self.alpha_x * np.sum(v * self._rho(self.gradient_magnitude2(m), e2)))

    def update_weights(self, model):
        for term in self.objfcts:
            if term not in self.smoothness_terms:
                term.update_weights(model)
        terms = self.smoothness_terms
        if all(np.all(t.norm == 2.0) for t in terms):  # L2 warm-up
            for t in terms:
                t.set_weights(irls=np.ones(self.regularization_mesh.nC))
            return
        s = self.gradient_magnitude2(model)
        if self.focusing_threshold is None:
            e = self.threshold_scale * np.percentile(np.sqrt(s), self.threshold_percentile)
            self.focusing_threshold = float(max(e, 1e-12 * max(np.sqrt(s).max(), 1e-300)))
        w = self.cell_weights(s, self.focusing_threshold**2)
        vw = self.regularization_mesh.vol * w
        v = self.regularization_mesh.vol
        for t in terms:
            A = self._cell_average(t.orientation)            # faces -> cells
            face_vol = getattr(self.regularization_mesh, f"aveCC2F{t.orientation}") @ v
            t.set_weights(irls=(A.T @ vw) / face_vol)
