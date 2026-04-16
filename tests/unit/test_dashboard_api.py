from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def test_dashboard_api_health(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from llama_vllm.dashboard.app import create_dashboard_app

    db_path = tmp_path / "history.db"
    client = TestClient(create_dashboard_app(db_path=str(db_path)))
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_dashboard_index_contains_history_ui_hooks(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from llama_vllm.dashboard.app import create_dashboard_app

    db_path = tmp_path / "history.db"
    client = TestClient(create_dashboard_app(db_path=str(db_path)))
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert 'id="historyTaskFilter"' in html
    assert 'id="historyActionFilter"' in html
    assert 'id="historyOkFilter"' in html
    assert "function renderHistoryTable(items)" in html
    assert "function backfillFromHistory(encoded)" in html


def test_dashboard_api_command_preview(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from llama_vllm.dashboard.app import create_dashboard_app

    db_path = tmp_path / "history.db"
    client = TestClient(create_dashboard_app(db_path=str(db_path)))
    payload = {
        "task_type": "distill",
        "config_path": "configs/distillation/feature_distill.yaml",
        "overrides": ["training.bf16=false"],
        "shell_style": "powershell",
    }
    resp = client.post("/api/command-preview", json=payload)
    assert resp.status_code == 200
    assert "llama-vllm distill run" in resp.json()["command"]

    history_resp = client.get("/api/history?limit=10")
    assert history_resp.status_code == 200
    data = history_resp.json()
    assert len(data["items"]) >= 1
    assert data["items"][0]["action"] in {"command-preview", "preflight"}


def test_dashboard_api_history_filters_and_item_lookup(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from llama_vllm.dashboard.app import create_dashboard_app

    db_path = tmp_path / "history.db"
    client = TestClient(create_dashboard_app(db_path=str(db_path)))

    payload_preview = {
        "task_type": "distill",
        "config_path": "configs/distillation/feature_distill.yaml",
        "overrides": [],
        "shell_style": "powershell",
    }
    client.post("/api/command-preview", json=payload_preview)

    payload_preflight = {
        "task_type": "infer",
        "config_path": "configs/inference/server.yaml",
        "overrides": [],
        "shell_style": "auto",
    }
    client.post("/api/preflight", json=payload_preflight)

    filtered = client.get("/api/history?action=preflight")
    assert filtered.status_code == 200
    items = filtered.json()["items"]
    assert len(items) >= 1
    assert all(item["action"] == "preflight" for item in items)

    event_id = items[0]["id"]
    item_resp = client.get(f"/api/history/{event_id}")
    assert item_resp.status_code == 200
    assert item_resp.json()["id"] == event_id


def test_dashboard_api_clear_history(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from llama_vllm.dashboard.app import create_dashboard_app

    db_path = tmp_path / "history.db"
    client = TestClient(create_dashboard_app(db_path=str(db_path)))

    payload = {
        "task_type": "distill",
        "config_path": "configs/distillation/feature_distill.yaml",
        "overrides": [],
        "shell_style": "auto",
    }
    client.post("/api/command-preview", json=payload)

    clear_resp = client.delete("/api/history")
    assert clear_resp.status_code == 200
    assert clear_resp.json()["deleted"] >= 1

    history_resp = client.get("/api/history")
    assert history_resp.status_code == 200
    assert history_resp.json()["items"] == []


def test_dashboard_api_clear_history_by_filter(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from llama_vllm.dashboard.app import create_dashboard_app

    db_path = tmp_path / "history.db"
    client = TestClient(create_dashboard_app(db_path=str(db_path)))

    payload_preview = {
        "task_type": "distill",
        "config_path": "configs/distillation/feature_distill.yaml",
        "overrides": [],
        "shell_style": "auto",
    }
    payload_preflight = {
        "task_type": "infer",
        "config_path": "configs/inference/server.yaml",
        "overrides": [],
        "shell_style": "auto",
    }
    client.post("/api/command-preview", json=payload_preview)
    client.post("/api/preflight", json=payload_preflight)

    clear_resp = client.delete("/api/history?action=command-preview")
    assert clear_resp.status_code == 200
    assert clear_resp.json()["deleted"] >= 1

    remaining = client.get("/api/history")
    assert remaining.status_code == 200
    items = remaining.json()["items"]
    assert all(item["action"] != "command-preview" for item in items)


