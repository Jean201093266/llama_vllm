from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llama_vllm.dashboard.history import (
    clear_events_filtered,
    clear_events,
    get_event_by_id,
    init_db,
    list_recent_events,
    list_recent_events_filtered,
    record_event,
)


def test_dashboard_history_sqlite_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "history.db"
    init_db(str(db_path))

    row_id = record_event(
        str(db_path),
        action="preflight",
        task_type="distill",
        config_path="configs/distillation/feature_distill.yaml",
        overrides=["training.bf16=false"],
        shell_style="powershell",
        ok=False,
        result={"ok": False, "errors": ["x"]},
    )
    assert row_id >= 1

    rows = list_recent_events(str(db_path), limit=10)
    assert len(rows) == 1
    assert rows[0]["action"] == "preflight"
    assert rows[0]["task_type"] == "distill"
    assert rows[0]["overrides"] == ["training.bf16=false"]
    assert rows[0]["ok"] is False


def test_dashboard_history_filters_get_and_clear(tmp_path: Path) -> None:
    db_path = tmp_path / "history.db"
    init_db(str(db_path))

    id1 = record_event(
        str(db_path),
        action="preflight",
        task_type="distill",
        config_path="a.yaml",
        overrides=[],
        shell_style="auto",
        ok=False,
        result={"ok": False},
    )
    id2 = record_event(
        str(db_path),
        action="command-preview",
        task_type="infer",
        config_path="b.yaml",
        overrides=[],
        shell_style="auto",
        ok=True,
        result={"command": "x"},
    )
    assert id1 != id2

    filtered = list_recent_events_filtered(str(db_path), action="preflight", limit=10)
    assert len(filtered) == 1
    assert filtered[0]["action"] == "preflight"

    item = get_event_by_id(str(db_path), id2)
    assert item is not None
    assert item["action"] == "command-preview"

    deleted = clear_events(str(db_path))
    assert deleted == 2
    assert list_recent_events(str(db_path), limit=10) == []


def test_dashboard_history_clear_filtered(tmp_path: Path) -> None:
    db_path = tmp_path / "history.db"
    init_db(str(db_path))

    record_event(
        str(db_path),
        action="preflight",
        task_type="distill",
        config_path="a.yaml",
        overrides=[],
        shell_style="auto",
        ok=False,
        result={"ok": False},
    )
    record_event(
        str(db_path),
        action="preflight",
        task_type="infer",
        config_path="b.yaml",
        overrides=[],
        shell_style="auto",
        ok=True,
        result={"ok": True},
    )

    deleted = clear_events_filtered(str(db_path), action="preflight", task_type="distill")
    assert deleted == 1

    rows = list_recent_events(str(db_path), limit=10)
    assert len(rows) == 1
    assert rows[0]["task_type"] == "infer"


