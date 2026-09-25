"""Tests for cloud task serialization and result packing."""

import json
import os
import tempfile
import zipfile

import numpy as np
import pytest

from geoinv3d.cloud.task import InversionTask, pack_task, unpack_task
from geoinv3d.cloud.worker import pack_result


class TestInversionTask:
    def test_to_meta(self):
        task = InversionTask(
            task_id="t1", nx=5, ny=5, nz=3,
            dx=10.0, dy=10.0, dz=5.0,
            method_type="gravity", max_iter=10,
        )
        meta = task.to_meta()
        assert meta["task_id"] == "t1"
        assert meta["nx"] == 5
        assert meta["method_type"] == "gravity"
        assert "initial_model" not in meta

    def test_to_meta_joint(self):
        task = InversionTask(
            task_id="j1", nx=5, ny=5, nz=3,
            dx=10.0, dy=10.0, dz=5.0,
            joint_methods=["gravity", "magnetics"],
            joint_weights=[1.0, 0.5],
        )
        meta = task.to_meta()
        assert meta["joint_methods"] == ["gravity", "magnetics"]
        assert meta["joint_weights"] == [1.0, 0.5]


class TestPackUnpack:
    def test_roundtrip_single(self, tmp_path):
        model = np.random.randn(75)
        obs = np.random.randn(36)
        std = np.abs(obs) * 0.1
        locs = np.random.randn(36, 3)

        task = InversionTask(
            task_id="rt1", nx=5, ny=5, nz=3,
            dx=10.0, dy=10.0, dz=5.0,
            method_type="gravity",
            initial_model=model,
            observed_data=obs,
            data_std=std,
            station_locations=locs,
        )

        path = str(tmp_path / "test.zip")
        pack_task(task, path)
        assert os.path.exists(path)

        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            assert "meta.json" in names
            assert "initial_model.npy" in names
            assert "observed.npy" in names

        task2 = unpack_task(path)
        assert task2.task_id == "rt1"
        assert task2.nx == 5
        assert task2.method_type == "gravity"
        np.testing.assert_array_almost_equal(task2.initial_model, model)
        np.testing.assert_array_almost_equal(task2.observed_data, obs)
        np.testing.assert_array_almost_equal(task2.data_std, std)
        np.testing.assert_array_almost_equal(task2.station_locations, locs)

    def test_roundtrip_joint(self, tmp_path):
        n_cells = 75
        surveys = [
            {
                "observed": np.random.randn(36),
                "std": np.ones(36) * 0.1,
                "locations": np.random.randn(36, 3),
            },
            {
                "observed": np.random.randn(36),
                "std": np.ones(36) * 0.05,
                "locations": np.random.randn(36, 3),
            },
        ]
        models = [np.zeros(n_cells), np.zeros(n_cells)]

        task = InversionTask(
            task_id="jrt1", nx=5, ny=5, nz=3,
            dx=10.0, dy=10.0, dz=5.0,
            joint_methods=["gravity", "magnetics"],
            joint_weights=[1.0, 1.0],
            joint_surveys=surveys,
            joint_initial_models=models,
        )

        path = str(tmp_path / "joint.zip")
        pack_task(task, path)

        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            assert "joint_obs_0.npy" in names
            assert "joint_obs_1.npy" in names
            assert "joint_model_0.npy" in names

        task2 = unpack_task(path)
        assert task2.joint_methods == ["gravity", "magnetics"]
        assert len(task2.joint_surveys) == 2
        assert len(task2.joint_initial_models) == 2
        np.testing.assert_array_almost_equal(
            task2.joint_surveys[0]["observed"], surveys[0]["observed"],
        )

    def test_empty_arrays(self, tmp_path):
        task = InversionTask(
            task_id="empty", nx=5, ny=5, nz=3,
            dx=10.0, dy=10.0, dz=5.0,
        )
        path = str(tmp_path / "empty.zip")
        pack_task(task, path)

        task2 = unpack_task(path)
        assert task2.initial_model is None
        assert task2.observed_data is None


class TestSparseTask:
    def test_roundtrip_sparse_fields(self, tmp_path):
        hx = np.concatenate([np.full(3, 200.0), np.full(10, 100.0), np.full(3, 200.0)])
        hy = hx.copy()
        hz = np.concatenate([np.full(2, 100.0), np.full(5, 50.0)])
        active = np.ones(len(hx) * len(hy) * len(hz), dtype=bool)
        active[:100] = False

        task = InversionTask(
            task_id="sparse1",
            hx=hx, hy=hy, hz=hz,
            origin=(1000.0, 2000.0, -500.0),
            method_type="gravity",
            regularization_type="sparse",
            norms=(0.0, 2.0, 2.0, 1.0),
            irls_cooling_factor=1.1,
            max_irls_iterations=12,
            use_preconditioner=True,
            bounds_lower=-1.0,
            bounds_upper=2.0,
            active_cells=active,
            initial_model=np.zeros(int(active.sum())),
            observed_data=np.random.randn(50),
            data_std=np.ones(50),
            station_locations=np.random.randn(50, 3),
        )

        path = str(tmp_path / "sparse.zip")
        pack_task(task, path)
        task2 = unpack_task(path)

        assert task2.regularization_type == "sparse"
        assert tuple(task2.norms) == (0.0, 2.0, 2.0, 1.0)
        assert task2.irls_cooling_factor == 1.1
        assert task2.max_irls_iterations == 12
        assert task2.use_preconditioner is True
        assert task2.bounds_lower == -1.0
        assert task2.bounds_upper == 2.0
        np.testing.assert_array_almost_equal(task2.hx, hx)
        np.testing.assert_array_almost_equal(task2.hy, hy)
        np.testing.assert_array_almost_equal(task2.hz, hz)
        np.testing.assert_array_equal(task2.active_cells, active)
        assert task2.origin == (1000.0, 2000.0, -500.0)

    def test_backward_compat_defaults(self, tmp_path):
        """Old tasks without sparse fields should get sensible defaults."""
        task = InversionTask(
            task_id="old", nx=5, ny=5, nz=3,
            dx=10.0, dy=10.0, dz=5.0,
        )
        path = str(tmp_path / "old.zip")
        pack_task(task, path)
        task2 = unpack_task(path)
        assert task2.regularization_type == "sparse"
        assert tuple(task2.norms) == (0.0, 2.0, 2.0, 1.0)
        assert task2.use_preconditioner is True
        assert task2.active_cells is None
        assert task2.hx is None


class TestPackResult:
    def test_single_result(self, tmp_path):
        result = {
            "task_id": "r1",
            "method": "gravity",
            "converged": True,
            "n_iterations": 5,
            "iterations": [{"iteration": 1, "phi_d": 100.0}],
            "recovered_model": np.random.randn(75),
        }

        path = str(tmp_path / "result.zip")
        pack_result(result, path)
        assert os.path.exists(path)

        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            assert "result.json" in names
            assert "recovered_model.npy" in names

            meta = json.loads(zf.read("result.json"))
            assert meta["task_id"] == "r1"
            assert meta["converged"] is True
            assert "recovered_model" not in meta

    def test_joint_result(self, tmp_path):
        result = {
            "task_id": "jr1",
            "methods": ["gravity", "magnetics"],
            "converged": True,
            "n_iterations": 10,
            "iterations": [],
            "recovered_models": {
                "gravity": np.random.randn(75),
                "magnetics": np.random.randn(75),
            },
        }

        path = str(tmp_path / "joint_result.zip")
        pack_result(result, path)

        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            assert "recovered_gravity.npy" in names
            assert "recovered_magnetics.npy" in names
