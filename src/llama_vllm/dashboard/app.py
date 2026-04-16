"""FastAPI dashboard for dry-run preflight and command preview."""

from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llama_vllm.dashboard.history import (
    clear_events,
    clear_events_filtered,
    get_event_by_id,
    init_db,
    list_recent_events_filtered,
    record_event,
)
from llama_vllm.dashboard.service import build_command_preview, run_preflight


class DashboardRequest(BaseModel):
    task_type: Literal["distill", "finetune", "infer"]
    config_path: str
    overrides: list[str] = Field(default_factory=list)
    shell_style: Literal["auto", "powershell", "posix"] = "auto"


class PreflightResponse(BaseModel):
    ok: bool
    task_type: str
    message: str
    errors: list[str]
    suggestions: list[str]
    formatted_suggestions: list[str]


class CommandPreviewResponse(BaseModel):
    command: str


class HistoryEvent(BaseModel):
    id: int
    created_at_utc: str
    action: str
    task_type: str | None = None
    config_path: str | None = None
    overrides: list[str] = Field(default_factory=list)
    shell_style: str | None = None
    ok: bool | None = None
    result: dict = Field(default_factory=dict)


class HistoryResponse(BaseModel):
    items: list[HistoryEvent]


class ClearHistoryResponse(BaseModel):
    deleted: int


def create_dashboard_app(db_path: str | None = None) -> FastAPI:
    app = FastAPI(title="llama-vllm dashboard", version="0.1.0")
    resolved_db_path = db_path or os.path.join(".dashboard", "history.db")
    init_db(resolved_db_path)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}



    @app.post("/api/preflight", response_model=PreflightResponse)
    def api_preflight(request: DashboardRequest) -> PreflightResponse:
        result = run_preflight(
            request.task_type,
            request.config_path,
            request.overrides,
            shell_style=request.shell_style,
        )
        record_event(
            resolved_db_path,
            action="preflight",
            task_type=request.task_type,
            config_path=request.config_path,
            overrides=request.overrides,
            shell_style=request.shell_style,
            ok=result.get("ok"),
            result=result,
        )
        return PreflightResponse(**result)

    @app.post("/api/command-preview", response_model=CommandPreviewResponse)
    def api_command_preview(request: DashboardRequest) -> CommandPreviewResponse:
        cmd = build_command_preview(
            request.task_type,
            request.config_path,
            request.overrides,
            shell_style=request.shell_style,
        )
        record_event(
            resolved_db_path,
            action="command-preview",
            task_type=request.task_type,
            config_path=request.config_path,
            overrides=request.overrides,
            shell_style=request.shell_style,
            ok=True,
            result={"command": cmd},
        )
        return CommandPreviewResponse(command=cmd)

    @app.get("/api/history", response_model=HistoryResponse)
    def api_history(
        limit: int = 50,
        action: str | None = None,
        task_type: str | None = None,
        ok: bool | None = None,
    ) -> HistoryResponse:
        items = [
            HistoryEvent(**item)
            for item in list_recent_events_filtered(
                resolved_db_path,
                limit=limit,
                action=action,
                task_type=task_type,
                ok=ok,
            )
        ]
        return HistoryResponse(items=items)

    @app.get("/api/history/{event_id}", response_model=HistoryEvent)
    def api_history_item(event_id: int) -> HistoryEvent:
        item = get_event_by_id(resolved_db_path, event_id)
        if item is None:
            raise HTTPException(status_code=404, detail="History event not found")
        return HistoryEvent(**item)

    @app.delete("/api/history", response_model=ClearHistoryResponse)
    def api_history_clear(
        action: str | None = None,
        task_type: str | None = None,
        ok: bool | None = None,
    ) -> ClearHistoryResponse:
        if action is None and task_type is None and ok is None:
            deleted = clear_events(resolved_db_path)
        else:
            deleted = clear_events_filtered(
                resolved_db_path,
                action=action,
                task_type=task_type,
                ok=ok,
            )
        return ClearHistoryResponse(deleted=deleted)

    return app

