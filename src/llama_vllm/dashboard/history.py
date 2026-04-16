"""SQLite persistence for dashboard dry-run history."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any


def _connect(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def init_db(db_path: str) -> None:
    """Initialize history database and schema if missing."""
    parent = os.path.dirname(os.path.abspath(db_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dashboard_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_utc TEXT NOT NULL,
                action TEXT NOT NULL,
                task_type TEXT,
                config_path TEXT,
                overrides_json TEXT NOT NULL,
                shell_style TEXT,
                ok INTEGER,
                result_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_dashboard_history_id
            ON dashboard_history(id DESC)
            """
        )


def record_event(
    db_path: str,
    *,
    action: str,
    task_type: str | None,
    config_path: str | None,
    overrides: list[str],
    shell_style: str | None,
    ok: bool | None,
    result: dict[str, Any],
) -> int:
    """Insert one dashboard event row and return row id."""
    payload = (
        datetime.now(timezone.utc).isoformat(),
        action,
        task_type,
        config_path,
        json.dumps(overrides, ensure_ascii=False),
        shell_style,
        int(ok) if ok is not None else None,
        json.dumps(result, ensure_ascii=False),
    )

    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO dashboard_history (
                created_at_utc, action, task_type, config_path,
                overrides_json, shell_style, ok, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        return int(cursor.lastrowid)


def list_recent_events(db_path: str, limit: int = 50) -> list[dict[str, Any]]:
    """Return recent dashboard history rows in descending id order."""
    safe_limit = max(1, min(limit, 500))

    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT id, created_at_utc, action, task_type, config_path,
                   overrides_json, shell_style, ok, result_json
            FROM dashboard_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        )
        rows = cursor.fetchall()

    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "id": row[0],
                "created_at_utc": row[1],
                "action": row[2],
                "task_type": row[3],
                "config_path": row[4],
                "overrides": json.loads(row[5]),
                "shell_style": row[6],
                "ok": None if row[7] is None else bool(row[7]),
                "result": json.loads(row[8]),
            }
        )
    return result


def list_recent_events_filtered(
    db_path: str,
    *,
    limit: int = 50,
    action: str | None = None,
    task_type: str | None = None,
    ok: bool | None = None,
) -> list[dict[str, Any]]:
    """Return filtered recent events in descending id order."""
    safe_limit = max(1, min(limit, 500))
    clauses: list[str] = []
    values: list[Any] = []

    if action is not None:
        clauses.append("action = ?")
        values.append(action)
    if task_type is not None:
        clauses.append("task_type = ?")
        values.append(task_type)
    if ok is not None:
        clauses.append("ok = ?")
        values.append(int(ok))

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = (
        "SELECT id, created_at_utc, action, task_type, config_path, "
        "overrides_json, shell_style, ok, result_json "
        "FROM dashboard_history "
        f"{where_sql} "
        "ORDER BY id DESC LIMIT ?"
    )
    values.append(safe_limit)

    with _connect(db_path) as conn:
        cursor = conn.execute(query, tuple(values))
        rows = cursor.fetchall()

    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "id": row[0],
                "created_at_utc": row[1],
                "action": row[2],
                "task_type": row[3],
                "config_path": row[4],
                "overrides": json.loads(row[5]),
                "shell_style": row[6],
                "ok": None if row[7] is None else bool(row[7]),
                "result": json.loads(row[8]),
            }
        )
    return result


def get_event_by_id(db_path: str, event_id: int) -> dict[str, Any] | None:
    """Get one event by id."""
    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT id, created_at_utc, action, task_type, config_path,
                   overrides_json, shell_style, ok, result_json
            FROM dashboard_history
            WHERE id = ?
            """,
            (event_id,),
        )
        row = cursor.fetchone()

    if row is None:
        return None
    return {
        "id": row[0],
        "created_at_utc": row[1],
        "action": row[2],
        "task_type": row[3],
        "config_path": row[4],
        "overrides": json.loads(row[5]),
        "shell_style": row[6],
        "ok": None if row[7] is None else bool(row[7]),
        "result": json.loads(row[8]),
    }


def clear_events(db_path: str) -> int:
    """Delete all history rows and return deleted count."""
    with _connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM dashboard_history")
        count = int(cursor.fetchone()[0])
        conn.execute("DELETE FROM dashboard_history")
    return count


def clear_events_filtered(
    db_path: str,
    *,
    action: str | None = None,
    task_type: str | None = None,
    ok: bool | None = None,
) -> int:
    """Delete history rows by optional filters and return deleted count."""
    clauses: list[str] = []
    values: list[Any] = []

    if action is not None:
        clauses.append("action = ?")
        values.append(action)
    if task_type is not None:
        clauses.append("task_type = ?")
        values.append(task_type)
    if ok is not None:
        clauses.append("ok = ?")
        values.append(int(ok))

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    with _connect(db_path) as conn:
        count_query = f"SELECT COUNT(*) FROM dashboard_history {where_sql}"
        cursor = conn.execute(count_query, tuple(values))
        count = int(cursor.fetchone()[0])
        delete_query = f"DELETE FROM dashboard_history {where_sql}"
        conn.execute(delete_query, tuple(values))

    return count


