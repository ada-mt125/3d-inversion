"""Every test keeps the API server's state in its own temporary folder, never in ~/.geoinv3d."""

import pytest


@pytest.fixture(autouse=True)
def _private_state(tmp_path, monkeypatch):
    for name, sub in (("GEOINV3D_INPUTS_DIR", "inputs"), ("GEOINV3D_LOCAL_DIR", "local"),
                      ("GEOINV3D_DEM_DIR", "dem"), ("GEOINV3D_WORKSPACES_DIR", "workspaces"),
                      ("GEOINV3D_COMPARISONS_DIR", "comparisons"),
                      ("GEOINV3D_JOBS_FILE", "jobs.json")):
        monkeypatch.setenv(name, str(tmp_path / "geoinv3d_state" / sub))
    try:   # a backend made by an earlier test would keep that test's folder
        from geoinv3d.api import server
        monkeypatch.setattr(server, "_local_backend", None)
    except ImportError:    # the server needs the cloud extras
        pass
