"""API server: local-only access, the vCPU limit, job groups, saved comparisons and workspaces."""

import json
import time

import numpy as np

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient   # noqa: E402

from geoinv3d.api import server   # noqa: E402
from tests.test_result_workflow import _run   # noqa: E402


class FakeBackend:
    """Jobs keep whatever status the test stores; results come from one zip."""

    def __init__(self, result_zip=None):
        self.result_zip, self.started, self.finished = result_zip, [], []

    def request_finish(self, record):
        self.finished.append(record["job_id"])

    def start_job(self, task_id, files, params, instance_type):
        self.started.append((task_id, params, instance_type))
        return f"i-{len(self.started)}"

    def initial_fields(self):
        return {"status": "STARTING", "backend": "ec2"}

    def refresh(self, record):
        return {"status": record.get("status", "RUNNING")}

    def result_summary(self, record):
        return {}

    def fetch_result(self, record, output_dir):
        return str(self.result_zip)


@pytest.fixture
def api(tmp_path, monkeypatch):
    backend = FakeBackend()
    store = server.JobStore(tmp_path / "jobs.json")
    monkeypatch.setattr(server, "_backend", backend)
    monkeypatch.setattr(server, "_store", store)
    monkeypatch.setenv("GEOINV3D_COMPARISONS_DIR", str(tmp_path / "comparisons"))
    monkeypatch.setenv("GEOINV3D_INPUTS_DIR", str(tmp_path / "inputs"))
    monkeypatch.setattr(server, "_local_backend", None)
    monkeypatch.setenv("GEOINV3D_LOCAL_DIR", str(tmp_path / "local"))
    monkeypatch.setenv("GEOINV3D_VCPU_LIMIT", "36")
    monkeypatch.delenv("GEOINV3D_ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("GEOINV3D_ALLOWED_HOSTS", raising=False)
    return TestClient(server.app), backend, store


def _submit(client, instance="c5.2xlarge", params=None, workspace_id="", backend="aws", **headers):
    return client.post("/api/inversion/submit", headers=headers,
                       files=[("files", ("g.csv", b"x,y,v\n0,0,1\n", "text/csv"))],
                       data={"method": "gravity", "instance_type": instance, "backend": backend,
                             "params_json": json.dumps(params or {}), "workspace_id": workspace_id})


class TestLocalOnly:
    @pytest.mark.parametrize("origin", ["null", "http://localhost:8765", "http://127.0.0.1",
                                        "http://[::1]:3000"])
    def test_local_pages_are_allowed(self, api, origin):
        client, _, _ = api
        r = client.get("/api/health", headers={"Origin": origin})
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == origin

    @pytest.mark.parametrize("origin", ["https://evil.example", "http://localhost.evil.example",
                                        "http://127.0.0.1.evil.example:8000"])
    def test_other_web_pages_are_refused(self, api, origin):
        client, backend, _ = api
        assert client.get("/api/health", headers={"Origin": origin}).status_code == 403
        # a cross-site form POST (sent without a CORS preflight) launches nothing
        assert _submit(client, Origin=origin).status_code == 403
        assert backend.started == []

    def test_other_host_names_are_refused(self, api):
        client, _, _ = api
        ok = TestClient(server.app, base_url="http://127.0.0.1:8000")
        assert ok.get("/api/health").status_code == 200
        rebound = TestClient(server.app, base_url="http://attacker.example:8000")
        assert rebound.get("/api/health").status_code == 403

    def test_extra_origin_from_the_environment(self, api, monkeypatch):
        client, _, _ = api
        monkeypatch.setenv("GEOINV3D_ALLOWED_ORIGINS", "https://lab.example")
        assert client.get("/api/health", headers={"Origin": "https://lab.example"}).status_code == 200

    def test_upload_page_is_served_and_points_at_this_server(self, api):
        client, _, _ = api
        r = TestClient(server.app, base_url="http://localhost:8123").get("/")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        head = r.text.split("</head>", 1)[0]
        assert "window.GEOINV3D_API = location.origin" in head
        assert "apiBase" in r.text
        rebound = TestClient(server.app, base_url="http://attacker.example:8000")
        assert rebound.get("/").status_code == 403

    def test_preflight_for_a_local_page(self, api):
        client, _, _ = api
        r = client.options("/api/inversion/submit", headers={
            "Origin": "http://localhost:8765", "Access-Control-Request-Method": "POST"})
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == "http://localhost:8765"


class TestSubmit:
    def test_group_is_recorded_and_not_sent_to_the_worker(self, api):
        client, backend, store = api
        group = {"id": "g1", "label": "Depth weighting β", "variant": "β = 0.5"}
        r = _submit(client, params={"max_iter": 30, "group": group})
        assert r.status_code == 200
        record = store.get(r.json()["job_id"])
        assert (record["group_id"], record["group_label"], record["variant"]) == \
            ("g1", "Depth weighting β", "β = 0.5")
        assert "group" not in backend.started[0][1]

    def test_vcpu_limit(self, api):
        client, backend, store = api
        for i in range(4):   # 4 x 8 vCPUs running
            store.put({"job_id": f"run-{i}", "task_id": f"t{i}", "status": "RUNNING",
                       "instance_type": "c5.2xlarge"})
        store.put({"job_id": "done", "task_id": "t9", "status": "SUCCEEDED", "summary": {},
                   "instance_type": "c5.9xlarge"})
        assert _submit(client, "c5.xlarge").status_code == 200        # 32 + 4 = 36
        r = _submit(client, "c5.2xlarge")                             # 36 + 8 > 36
        assert r.status_code == 409 and "vCPUs" in r.json()["detail"]
        assert len(backend.started) == 1


class TestStopAndKeep:
    def test_a_running_job_is_asked_to_finish(self, api):
        client, backend, store = api
        store.put({"job_id": "i-1", "task_id": "t1", "status": "RUNNING"})
        r = client.post("/api/inversion/i-1/finish")
        assert r.status_code == 200 and backend.finished == ["i-1"]
        assert store.get("i-1")["finish_requested"]
        assert not store.get("i-1").get("cancel_requested")   # nothing terminated

    def test_only_a_running_job(self, api):
        client, backend, store = api
        store.put({"job_id": "i-2", "task_id": "t2", "status": "SUCCEEDED", "summary": {}})
        assert client.post("/api/inversion/i-2/finish").status_code == 409
        assert client.post("/api/inversion/nope/finish").status_code == 404
        assert backend.finished == []


class TestComparisons:
    @pytest.fixture
    def finished(self, api, tmp_path):
        client, backend, store = api
        _, zpath, _ = _run(tmp_path)
        backend.result_zip = zpath
        for job_id, variant in [("i-a", "β = 0.5"), ("i-b", "β = 1")]:
            store.put({"job_id": job_id, "task_id": job_id, "status": "SUCCEEDED", "summary": {},
                       "group_id": "g1", "variant": variant, "regularization_type": "sparse"})
        return client

    def test_compare_jobs_in_one_workflow(self, finished):
        wf = finished.get("/api/workflow", params={"ids": "i-a,i-b"}).json()
        inv = [n for n in wf["nodes"] if n["type"] == "RegularizedInversionNode"]
        assert [n["params"]["task"] for n in inv] == ["β = 0.5", "β = 1"]
        assert sum(n["type"] == "MeshCreateNode" for n in wf["nodes"]) == 1

    @pytest.mark.parametrize("ids", ["", "i-a,i-a", "i-a,nope"])
    def test_bad_lists(self, finished, ids):
        assert finished.get("/api/workflow", params={"ids": ids}).status_code in (400, 404)

    def test_save_list_open_export_delete(self, finished):
        r = finished.post("/api/comparisons", json={"name": "Depth weighting β", "job_ids": ["i-a", "i-b"]})
        assert r.status_code == 200
        cid = r.json()["id"]
        listed = finished.get("/api/comparisons").json()["comparisons"]
        assert [c["name"] for c in listed] == ["Depth weighting β"]
        wf = finished.get(f"/api/comparisons/{cid}").json()
        assert wf["comparison"]["job_ids"] == ["i-a", "i-b"]
        page = finished.get(f"/api/comparisons/{cid}/page")
        assert page.status_code == 200 and 'id="embedded-data"' in page.text
        assert "Depth_weighting" in page.headers["content-disposition"]
        assert finished.delete(f"/api/comparisons/{cid}").status_code == 200
        assert finished.get(f"/api/comparisons/{cid}").status_code == 404

    def test_ids_cannot_reach_other_files(self, finished):
        assert finished.get("/api/comparisons/..%2Fjobs").status_code == 404
        assert finished.get("/api/comparisons/0123456789ab").status_code == 404


class TestWorkspaces:
    @pytest.fixture
    def ws_api(self, api, tmp_path, monkeypatch):
        monkeypatch.setenv("GEOINV3D_WORKSPACES_DIR", str(tmp_path / "workspaces"))
        return api

    def test_create_rename_list_delete(self, ws_api):
        client, _, _ = ws_api
        a = client.post("/api/workspaces", json={}).json()
        assert a["name"] == "Untitled workspace" and a["job_ids"] == []
        time.sleep(0.01)
        b = client.post("/api/workspaces", json={"name": "  Karnataka   gravity "}).json()
        assert b["name"] == "Karnataka gravity"
        r = client.patch(f"/api/workspaces/{a['id']}", json={"name": "β sweep"})
        assert r.status_code == 200 and r.json()["name"] == "β sweep"
        time.sleep(0.01)
        client.get(f"/api/workspaces/{a['id']}")   # opening it makes it the last opened
        listed = client.get("/api/workspaces").json()["workspaces"]
        assert [w["name"] for w in listed] == ["β sweep", "Karnataka gravity"]
        assert client.patch(f"/api/workspaces/{b['id']}", json={"name": "x" * 121}).status_code == 400
        assert client.delete(f"/api/workspaces/{a['id']}").status_code == 200
        assert client.get(f"/api/workspaces/{a['id']}").status_code == 404

    def test_ids_cannot_reach_other_files(self, ws_api):
        client, _, _ = ws_api
        assert client.get("/api/workspaces/..%2Fjobs").status_code == 404
        assert client.get("/api/workspaces/0123456789ab").status_code == 404

    def test_jobs_submitted_from_a_workspace_join_it(self, ws_api):
        client, backend, store = ws_api
        ws = client.post("/api/workspaces", json={"name": "A"}).json()
        job_id = _submit(client, workspace_id=ws["id"]).json()["job_id"]
        assert store.get(job_id)["workspace_id"] == ws["id"]
        assert client.get(f"/api/workspaces/{ws['id']}").json()["job_ids"] == [job_id]
        _submit(client)   # a job outside any workspace
        mine = client.get("/api/jobs", params={"workspace": ws["id"]}).json()["jobs"]
        assert [j["job_id"] for j in mine] == [job_id]
        assert len(client.get("/api/jobs").json()["jobs"]) == 2
        # an unknown workspace launches nothing
        started = len(backend.started)
        assert _submit(client, workspace_id="0123456789ab").status_code == 404
        assert len(backend.started) == started

    def test_workflow_follows_the_finished_runs(self, ws_api, tmp_path):
        client, backend, store = ws_api
        _, zpath, _ = _run(tmp_path)
        backend.result_zip = zpath
        ws = client.post("/api/workspaces", json={}).json()
        url = f"/api/workspaces/{ws['id']}/workflow"
        ids = [_submit(client, params={"group": {"variant": v}}, workspace_id=ws["id"]).json()["job_id"]
               for v in ("β = 0.5", "β = 1")]
        d = client.get(url).json()
        assert d["workflow"] is None and (d["finished"], d["running"]) == (0, 2)

        store.update(ids[0], status="SUCCEEDED", summary={})
        d = client.get(url).json()
        assert (d["finished"], d["running"]) == (1, 1)
        runs = [n for n in d["workflow"]["nodes"] if n["type"] == "RegularizedInversionNode"]
        assert [n["params"]["task"] for n in runs] == ["β = 0.5"]
        # nothing new has finished: only the counts come back
        same = client.get(url, params={"known": d["key"]}).json()
        assert same["unchanged"] and "workflow" not in same

        store.update(ids[1], status="SUCCEEDED", summary={})
        d2 = client.get(url, params={"known": d["key"]}).json()
        assert d2["key"] != d["key"] and d2["finished"] == 2
        nodes = d2["workflow"]["nodes"]
        runs = [n for n in nodes if n["type"] == "RegularizedInversionNode"]
        assert sorted(n["params"]["task"] for n in runs) == ["β = 0.5", "β = 1"]
        assert sum(n["type"] == "SurveyCreateNode" for n in nodes) == 1   # the same data: one data node
        assert client.get(url).json()["workflow"] == d2["workflow"]      # served from the cache


class FakeLocal(FakeBackend):
    """The local backend: jobs named local-<task>, queued first."""

    def start_job(self, task_id, files, params, instance_type):
        self.started.append((task_id, params, instance_type))
        return f"local-{task_id}"

    def initial_fields(self):
        return {"status": "RUNNABLE", "phase": "queued", "backend": "local"}

    def resources(self):
        return {"cpus": 8, "memory_gb": 16.0, "max_jobs": 1}


class TestLocalJobsAndReruns:
    @pytest.fixture
    def both(self, api, monkeypatch):
        client, aws, store = api
        local = FakeLocal()
        monkeypatch.setattr(server, "_local_backend", local)
        return client, aws, local, store

    def test_a_local_job_needs_no_vcpus_and_keeps_its_inputs(self, both):
        client, aws, local, store = both
        for i in range(2):   # the account's vCPUs all in use
            store.put({"job_id": f"run-{i}", "task_id": f"t{i}", "status": "RUNNING",
                       "instance_type": "c5.9xlarge"})
        assert _submit(client).status_code == 409
        r = _submit(client, backend="local", params={"max_iter": 30})
        assert r.status_code == 200 and len(local.started) == 1 and not aws.started
        assert local.started[0][2] is None          # no instance type for this machine
        record = store.get(r.json()["job_id"])
        assert (record["backend"], record["instance_type"]) == ("local", "local")
        p = client.get(f"/api/inversion/{record['job_id']}/params").json()
        assert p["files"] == ["g.csv"] and p["params"]["max_iter"] == 30
        assert client.get("/api/jobs").json()["jobs"][0]["rerun"] is True

    def test_unknown_backend(self, both):
        client, *_ = both
        assert _submit(client, backend="mars").status_code == 400

    def test_rerun_with_changes(self, both):
        client, aws, local, store = both
        base = {"param_mode": "auto", "max_iter": 30, "datasets": [{"method": "gravity", "noise_floor": 0.05}]}
        job = _submit(client, backend="local", params=base).json()["job_id"]
        r = client.post(f"/api/inversion/{job}/rerun",
                        json={"changes": {"max_iter": 60, "datasets": [{"noise_floor": 0.5}]}})
        assert r.status_code == 200
        params = local.started[-1][1]
        assert params["max_iter"] == 60 and params["datasets"] == [{"method": "gravity", "noise_floor": 0.5}]
        assert params["param_mode"] == "auto"       # neither change is a manual-mode setting
        record = store.get(r.json()["job_id"])
        assert record["parent_job"] == job and record["changes"]["max_iter"] == 60
        assert record["note"].startswith("re-run of")
        # a regularization setting makes an auto job a manual one
        client.post(f"/api/inversion/{job}/rerun", json={"changes": {"norms": [1, 2, 2, 2]}})
        assert local.started[-1][1]["param_mode"] == "manual"

    def test_rerun_with_the_whole_parameters_on_aws(self, both):
        client, aws, local, store = both
        job = _submit(client, backend="local", params={"max_iter": 30, "alpha_s": 1}).json()["job_id"]
        r = client.post(f"/api/inversion/{job}/rerun", json={
            "params": {"max_iter": 30, "alpha_s": 0.1}, "backend": "aws", "instance_type": "c5.4xlarge"})
        assert r.status_code == 200
        task, params, instance = aws.started[-1]
        assert params["alpha_s"] == 0.1 and instance == "c5.4xlarge" and params["task_id"] == task
        assert store.get(r.json()["job_id"])["changes"] == {"alpha_s": 0.1}

    def test_a_preview_runs_its_full_settings_where_they_were_meant_to_go(self, both):
        client, aws, local, store = both
        full = {"core_cell_m": 1000, "max_iter": 30}
        r = _submit(client, backend="local", params={"core_cell_m": 2000, "max_iter": 30, "preview_of": {
            "params": full, "backend": "aws", "instance_type": "c5.4xlarge"}})
        job = r.json()["job_id"]
        assert store.get(job)["preview"] is True
        assert "preview_of" not in local.started[-1][1]     # not sent to the worker
        r = client.post(f"/api/inversion/{job}/rerun", json={"full": True})
        assert r.status_code == 200
        task, params, instance = aws.started[-1]
        assert params["core_cell_m"] == 1000 and instance == "c5.4xlarge"
        assert store.get(r.json()["job_id"])["note"].startswith("full resolution of")
        # a job that previews nothing
        plain = _submit(client, backend="local").json()["job_id"]
        assert client.post(f"/api/inversion/{plain}/rerun", json={"full": True}).status_code == 400

    def test_jobs_whose_inputs_were_not_kept(self, both):
        client, _, _, store = both
        store.put({"job_id": "i-old", "task_id": "old", "status": "SUCCEEDED", "summary": {}})
        assert client.post("/api/inversion/i-old/rerun", json={}).status_code == 404
        assert client.get("/api/inversion/nope/params").status_code == 404

    def test_the_server_reports_this_machine(self, both):
        client, *_ = both
        health = client.get("/api/health").json()
        assert "local" in health and "comparisons_dir" in health


class TestMergeParams:
    def test_merge(self):
        base = {"a": 1, "d": {"x": 1, "y": 2}, "ds": [{"f": 1, "g": 2}], "l": [1, 2]}
        out = server.merge_params(base, {"a": None, "d": {"y": 3}, "ds": [{"f": 5}], "l": [3]})
        assert out == {"d": {"x": 1, "y": 3}, "ds": [{"f": 5, "g": 2}], "l": [3]}
        assert base["d"] == {"x": 1, "y": 2}         # the base is not changed


class TestEstimates:
    def test_rates_from_finished_jobs(self, api):
        client, _, store = api
        summary = {"n_data": 1000, "n_active_cells": 10000, "n_iterations": 10}
        store.put({"job_id": "local-a", "task_id": "a", "status": "SUCCEEDED", "backend": "local",
                   "method": "gravity", "started": 0, "stopped": 25_000, "summary": summary})
        store.put({"job_id": "i-b", "task_id": "b", "status": "SUCCEEDED", "backend": "ec2",
                   "instance_type": "c5.4xlarge", "method": "joint", "inversion_mode": "joint",
                   "created": 0, "started": 60_000, "stopped": 260_000, "summary": summary})
        store.put({"job_id": "i-c", "task_id": "c", "status": "FAILED", "backend": "ec2",
                   "method": "gravity", "started": 0, "stopped": 1, "summary": summary})
        d = client.get("/api/estimates").json()
        assert d["rates"]["local"] == {"rate": pytest.approx(20 / 1e8), "n": 1}   # 25 s less 5 s set-up
        assert d["rates"]["c5.4xlarge"]["rate"] == pytest.approx(200 / (2 * 1e8))
        assert d["setup"]["ec2"] == {"seconds": 60.0, "n": 1}
        assert d["rates"]["c5.xlarge"]["n"] == 0           # a default


class TestMapLayers:
    def test_put_get_and_check(self, api, tmp_path, monkeypatch):
        client, _, _ = api
        monkeypatch.setenv("GEOINV3D_WORKSPACES_DIR", str(tmp_path / "ws"))
        ws = client.post("/api/workspaces", json={"name": "Sandur"}).json()["id"]
        assert client.get(f"/api/workspaces/{ws}/layers").json()["layers"] == []
        layers = [{"name": "Mines", "kind": "points", "items": [{"x": 671000, "y": 1660100, "label": "K"}]},
                  {"name": "Belt", "kind": "lines", "items": [{"xy": [[0, 0], [1, 1]], "closed": True}]}]
        assert client.put(f"/api/workspaces/{ws}/layers", json={"layers": layers}).status_code == 200
        assert client.get(f"/api/workspaces/{ws}/layers").json()["layers"] == layers
        assert client.get(f"/api/workspaces/{ws}").json()["layers"] == layers   # the page reads them here
        bad = [{"name": "x", "kind": "raster", "items": []}]
        assert client.put(f"/api/workspaces/{ws}/layers", json={"layers": bad}).status_code == 400


class TestInputsAndBundle:
    def test_input_files_and_bundle(self, api):
        import io
        import zipfile
        client, _, store = api
        job = _submit(client, params={"max_iter": 30}).json()["job_id"]
        r = client.get(f"/api/inversion/{job}/inputs/g.csv")
        assert r.status_code == 200 and r.content == b"x,y,v\n0,0,1\n"
        assert client.get(f"/api/inversion/{job}/inputs/..%2Fparams.json").status_code == 404
        assert client.get(f"/api/inversion/{job}/inputs/nope.csv").status_code == 404
        r = client.get(f"/api/inversion/{job}/bundle")
        assert r.status_code == 200
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            assert sorted(zf.namelist()) == ["README.txt", "data/g.csv", "params.json"]
            assert json.loads(zf.read("params.json"))["max_iter"] == 30
            assert b"geoinv3d.cloud.worker --local params.json data out" in zf.read("README.txt")


class TestGroundAndField:
    def test_crs(self, api):
        pytest.importorskip("rasterio")
        client, _, _ = api
        d = client.get("/api/crs", params={"crs": "32643"}).json()
        assert d["crs"] == "EPSG:32643" and d["geographic"] is False and "43N" in d["description"]
        assert client.get("/api/crs", params={"crs": "EPSG:4326"}).json()["geographic"] is True
        assert client.get("/api/crs", params={"crs": "nonsense"}).status_code == 400

    def test_igrf(self, api):
        pytest.importorskip("rasterio")
        client, _, _ = api
        d = client.get("/api/igrf", params={"x": 676000, "y": 1669000, "crs": "EPSG:32643",
                                            "alt_m": 500, "date": "2020-01-01"}).json()
        assert d["inducing_field"] == [42094.0, 19.27, -1.35] and d["model"] == "IGRF-14"
        assert client.get("/api/igrf", params={"x": 76, "y": 15, "date": "1850-01-01"}).status_code == 400

    def test_dem(self, api, tmp_path, monkeypatch):
        client, _, _ = api
        import geoinv3d.io.dem as dem
        tif = tmp_path / "dem_abc123.tif"
        tif.write_bytes(b"II*\x00")
        meta = {"id": "abc123", "source": "srtm", "tiles": ["N15E076"], "sea_tiles": [], "pixel_m": 30.0,
                "shape": [2, 2], "elev_min": 1.0, "elev_max": 2.0, "bounds": [0, 1, 0, 1], "crs": "EPSG:32643",
                "file_name": "srtm30_dem_abc123.tif"}
        asked = {}

        def fake_build(bounds, crs, source="auto", margin_m=0.0, max_pixels=3000, **kw):
            asked.update(bounds=bounds, crs=crs, source=source, margin_m=margin_m)
            return {**meta, "path": str(tif), "cached": False}

        monkeypatch.setattr(dem, "build_dem", fake_build)
        monkeypatch.setattr(dem, "cached_dem", lambda i: {**meta, "path": str(tif)} if i == "abc123" else None)
        r = client.post("/api/dem", json={"bounds": [641000, 711000, 1634000, 1704000], "crs": "32643",
                                          "margin_m": 44000})
        assert r.status_code == 200 and r.json()["id"] == "abc123" and "path" not in r.json()
        assert asked == {"bounds": [641000.0, 711000.0, 1634000.0, 1704000.0], "crs": "EPSG:32643",
                         "source": "auto", "margin_m": 44000.0}
        assert client.get("/api/dem/abc123").content == b"II*\x00"
        assert client.get("/api/dem/zzz").status_code == 404
        assert client.post("/api/dem", json={"bounds": [1, 2], "crs": "EPSG:32643"}).status_code == 400


class TestDemRelief:
    def test_the_ground_within_bounds(self, api, tmp_path, monkeypatch):
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_origin
        import geoinv3d.io.dem as dem
        monkeypatch.setenv("GEOINV3D_DEM_DIR", str(tmp_path))
        z = np.arange(100, dtype="float32").reshape(10, 10) * 10.0     # rises east and south
        z[0, 0] = -32768.0                                                # a void
        with rasterio.open(tmp_path / "dem_abc123.tif", "w", driver="GTiff", width=10, height=10,
                           count=1, dtype="float32", crs="EPSG:32643", nodata=-32768.0,
                           transform=from_origin(0, 1000, 100, 100)) as out:
            out.write(z, 1)
        (tmp_path / "dem_abc123.json").write_text('{"id": "abc123", "file_name": "x.tif"}')
        assert dem.cached_dem("abc123")
        client, _, _ = api
        whole = client.get("/api/dem/abc123/relief", params={"bounds": "0,1000,0,1000"}).json()
        assert whole == {"zmin": 10.0, "zmax": 990.0, "n": 99}           # the void left out
        part = client.get("/api/dem/abc123/relief", params={"bounds": "200,400,600,800"}).json()
        assert (part["zmin"], part["zmax"], part["n"]) == (220.0, 330.0, 4)
        assert client.get("/api/dem/abc123/relief", params={"bounds": "5000,6000,0,10"}).json()["n"] == 0
        assert client.get("/api/dem/abc123/relief", params={"bounds": "1,0,0,1"}).status_code == 400
        assert client.get("/api/dem/nope/relief", params={"bounds": "0,1,0,1"}).status_code == 404


class TestMeshCells:
    """The page's memory estimate counts the octree a job would build, not a tensor mesh."""

    def test_an_octree_of_a_5_km_window(self, api):
        pytest.importorskip("discretize")
        client, _, _ = api
        mesh = {"extent": [0, 5000, 0, 5000], "core_cell_m": 50, "core_cell_z_m": 25,
                "depth_core_m": 4000, "pad_distance_m": 2000}
        d = client.post("/api/mesh/cells", json=mesh).json()
        assert d["ok"] and not d["too_large"]
        # finest cells only near the surface: about 11 per column of the 100 x 100 core,
        # where a tensor mesh of these cells would have over 2 million
        assert 80_000 < d["n_cells"] < 150_000 and d["n_total"] > d["n_cells"]
        tensor = client.post("/api/mesh/cells", json={**mesh, "mesh_type": "tensor"}).json()
        assert tensor["n_cells"] > 10 * d["n_cells"]

    def test_too_large_or_bad_meshes(self, api):
        client, _, _ = api
        whole = {"extent": [607200, 796725, 1520325, 1743187.5], "core_cell_m": 50,
                 "core_cell_z_m": 25, "depth_core_m": 4000, "pad_distance_m": 2000}
        assert client.post("/api/mesh/cells", json=whole).json() == {"ok": True, "too_large": True, "n_cells": None}
        assert client.post("/api/mesh/cells", json={**whole, "core_cell_m": 0}).status_code == 400
        assert client.post("/api/mesh/cells", json={"extent": [0, 1]}).status_code == 400


class TestEnhance:
    def test_maps_and_reuse(self, api):
        pytest.importorskip("scipy")
        client, _, _ = api
        x1 = np.arange(0, 20000, 500.0)
        X, Y = np.meshgrid(x1, x1)
        v = 100 * np.exp(-((X - 10000) ** 2 + (Y - 9000) ** 2) / (2 * 1500 ** 2)) + 0.001 * X
        csv = "x,y,tmi\n" + "\n".join(f"{a},{b},{c}" for a, b, c in zip(X.ravel(), Y.ravel(), v.ravel()))
        params = {"products": ["data", "rtp", "tilt", "residual"], "inc": 60, "dec": 0,
                  "regional": {"method": "butterworth", "cutoff_m": 8000}, "max_px": 20}
        r = client.post("/api/enhance", files=[("files", ("m.csv", csv.encode(), "text/csv"))],
                        data={"method": "magnetic", "component": "tmi", "params_json": json.dumps(params)})
        assert r.status_code == 200, r.text
        d = r.json()
        assert set(d["maps"]) == {"data", "rtp", "tilt", "residual"} and d["shape"] == [40, 40]
        assert d["display_step"] == 2 and len(d["maps"]["tilt"]["values"]) == 20
        assert d["maps"]["tilt"]["unit"] == "°" and d["maps"]["residual"]["symmetric"]
        r2 = client.post("/api/enhance", data={"data_id": d["data_id"], "method": "magnetic",
                                               "params_json": json.dumps({"products": ["nstd"]})})
        assert r2.status_code == 200 and list(r2.json()["maps"]) == ["nstd"]
        assert client.post("/api/enhance", data={"data_id": "nope", "params_json": "{}"}).status_code == 404
        # the residual of the data reduced to the pole, when the job inverts those
        both = {}
        for flag in (False, True):
            q = {**params, "products": ["residual"], "invert_rtp": flag}
            r3 = client.post("/api/enhance", data={"data_id": d["data_id"], "method": "magnetic",
                                                   "params_json": json.dumps(q)}).json()
            both[flag] = np.array(r3["maps"]["residual"]["values"], dtype=float)
            assert ("reduced to the pole" in " ".join(r3["notes"])) == flag
        assert not np.allclose(both[False], both[True], equal_nan=True)
