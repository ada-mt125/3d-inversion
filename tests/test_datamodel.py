"""Tests for the immutable data model."""

import pytest
import numpy as np

from geoinv3d.datamodel.mesh import Mesh3D
from geoinv3d.datamodel.model import PhysicalModel, PhysicalProperty
from geoinv3d.datamodel.survey import SurveyData
from geoinv3d.datamodel.result import InversionResult, IterationSnapshot


class TestMesh3D:
    def test_uniform(self):
        m = Mesh3D.uniform(10, 10, 5, dx=100, dy=100, dz=50)
        assert m.shape == (10, 10, 5)
        assert m.n_cells == 500

    def test_immutable(self):
        m = Mesh3D.uniform(5, 5, 3)
        with pytest.raises(ValueError):
            m.hx[0] = 999

    def test_origin(self):
        m = Mesh3D.uniform(5, 5, 3, origin=(100, 200, 300))
        assert m.origin == (100, 200, 300)


class TestPhysicalModel:
    def test_constant(self):
        m = Mesh3D.uniform(5, 5, 3)
        pm = PhysicalModel.constant(m, 2.67, PhysicalProperty.DENSITY)
        assert len(pm.values) == m.n_cells
        assert np.all(pm.values == 2.67)
        assert pm.units == "g/cm³"

    def test_values_3d(self):
        m = Mesh3D.uniform(5, 6, 7)
        pm = PhysicalModel.constant(m, 1.0, PhysicalProperty.DENSITY)
        v3d = pm.values_3d()
        assert v3d.shape == (5, 6, 7)

    def test_wrong_size_raises(self):
        m = Mesh3D.uniform(5, 5, 3)
        with pytest.raises(ValueError, match="n_cells"):
            PhysicalModel(m, np.zeros(10), PhysicalProperty.DENSITY)

    def test_immutable_values(self):
        m = Mesh3D.uniform(5, 5, 3)
        pm = PhysicalModel.constant(m, 1.0, PhysicalProperty.DENSITY)
        with pytest.raises(ValueError):
            pm.values[0] = 999


class TestSurveyData:
    def test_creation(self):
        locs = np.array([[0, 0, 0], [1, 0, 0]])
        obs = np.array([1.0, 2.0])
        std = np.array([0.1, 0.1])
        s = SurveyData(locs, obs, std, method="gravity")
        assert s.n_stations == 2
        assert s.n_data == 2

    def test_immutable(self):
        locs = np.array([[0, 0, 0]])
        obs = np.array([1.0])
        std = np.array([0.1])
        s = SurveyData(locs, obs, std)
        with pytest.raises(ValueError):
            s.observed[0] = 999


class TestInversionResult:
    def test_add_iterations(self):
        r = InversionResult(method="gravity")
        snap = IterationSnapshot(
            iteration=0, model_values=np.array([1.0, 2.0]),
            phi_d=100.0, phi_m=10.0, phi_total=110.0, beta=1.0,
        )
        r.add_iteration(snap)
        assert r.n_iterations == 1
        assert r.phi_d_history[0] == 100.0
