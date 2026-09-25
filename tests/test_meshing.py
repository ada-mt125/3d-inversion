"""Tests for the data-driven mesh design (geoinv3d.cloud.meshing).

The upload page ports these rules to JavaScript; the canonical cases below
are also checked there (see LOGBOOK), so change both together.
"""

import numpy as np
import pytest

from geoinv3d.cloud.meshing import (
    nice_number, next_nice_number, padding_cells, recommend_mesh, station_spacing,
    tensor_shape,
)


def _grid(n, h):
    xx, yy = np.meshgrid(np.arange(n) * h, np.arange(n) * h)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _extent(xy):
    return (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())


class TestNiceNumbers:
    @pytest.mark.parametrize("x, nice", [(97, 100), (37, 50), (0.3, 0.25), (150, 200),
                                         (1800, 2000), (230, 250), (7000, 5000)])
    def test_nice_number(self, x, nice):
        assert nice_number(x) == pytest.approx(nice)

    @pytest.mark.parametrize("x, nxt", [(100, 200), (200, 250), (250, 500), (500, 1000),
                                        (0.25, 0.5), (130, 200)])
    def test_next_nice_number(self, x, nxt):
        assert next_nice_number(x) == pytest.approx(nxt)


class TestPadding:
    def test_padding_reaches_the_distance_with_expanding_cells(self):
        n = padding_cells(100, 2000)
        widths = 100 * 1.3 ** np.arange(1, n + 1)
        assert widths.sum() >= 2000 > widths[:-1].sum()
        assert n == 7   # counting 2000 / 100 = 20 cells would reach ~80 km

    def test_minimum_three_cells(self):
        assert padding_cells(100, 50) == 3


class TestStationSpacing:
    def test_regular_grid(self):
        s = station_spacing(_grid(7, 100.0))
        assert s["area_spacing"] == pytest.approx(100.0)
        assert s["nn_spacing"] == pytest.approx(100.0)
        assert s["spacing"] == pytest.approx(100.0)

    def test_single_line(self):
        s = station_spacing(np.column_stack([np.arange(11) * 100.0, np.zeros(11)]))
        assert s["spacing"] == pytest.approx(100.0)

    def test_random_scatter(self):
        xy = np.random.default_rng(0).uniform(0, 1e4, (10000, 2))
        assert station_spacing(xy)["area_spacing"] == pytest.approx(100.0, rel=0.05)

    def test_line_survey_uses_half_the_line_spacing(self):
        lines = [np.column_stack([np.arange(0, 10001, 5.0), np.full(2001, 1000.0 * k)])
                 for k in range(11)]
        s = station_spacing(np.vstack(lines))
        assert s["nn_spacing"] == pytest.approx(5.0)
        assert s["across_spacing"] == pytest.approx(1000.0, rel=0.1)
        assert s["spacing"] == pytest.approx(500.0, rel=0.1)

    def test_duplicates_and_too_few(self):
        xy = np.vstack([_grid(5, 50.0)] * 2)   # every station twice
        assert station_spacing(xy)["nn_spacing"] == pytest.approx(50.0)
        with pytest.raises(ValueError):
            station_spacing(np.array([[0.0, 0.0]]))


class TestRecommendMesh:
    def test_grid_survey(self):
        xy = _grid(7, 100.0)
        r = recommend_mesh(_extent(xy), 100.0, len(xy))
        assert (r["core_cell_m"], r["core_cell_z_m"]) == (100.0, 50.0)
        assert r["depth_core_m"] == 300.0 and r["pad_distance_m"] == 250.0
        assert r["shape"] == list(tensor_shape(_extent(xy), 100.0, 50.0, 300.0, 250.0))
        assert r["coarsened_from"] is None

    def test_paper_scale_survey(self):
        xy = _grid(26, 2000.0)
        r = recommend_mesh(_extent(xy), 2000.0, len(xy))
        assert (r["core_cell_m"], r["core_cell_z_m"], r["depth_core_m"]) == (2000, 1000, 25000)

    def test_budget_coarsens_cells(self):
        xy = np.random.default_rng(0).uniform(0, 1e4, (10000, 2))
        r = recommend_mesh(_extent(xy), 100.0, len(xy))
        assert r["coarsened_from"] == 100.0
        assert r["core_cell_m"] > 100.0
        assert r["n_cells"] <= 500_000 and r["sensitivity_bytes"] <= 8e9
        # a looser budget keeps the spacing-based cells
        loose = recommend_mesh(_extent(xy), 100.0, len(xy), max_cells=10**8,
                               max_sensitivity_bytes=1e12)
        assert loose["core_cell_m"] == 100.0 and loose["coarsened_from"] is None
