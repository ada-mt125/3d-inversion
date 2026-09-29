"""API server: local-only access, the vCPU limit, job groups, saved comparisons and workspaces."""

import json
import time

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
    monkeypatch.setenv("GEOINV3D_VCPU_LIMIT", "36")
    monkeypatch.delenv("GEOINV3D_ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("GEOINV3D_ALLOWED_HOSTS", raising=False)
    return TestClient(server.app), backend, store


def _submit(client, instance="c5.2xlarge", params=None, workspace_id="", **headers):
    return client.post("/api/inversion/submit", headers=headers,
                       files=[("files", ("g.csv", b"x,y,v\n0,0,1\n", "text/csv"))],
                       data={"method": "gravity", "instance_type": instance,
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
