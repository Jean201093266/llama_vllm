"""Service layer for dashboard preflight and command preview."""

from __future__ import annotations

import os
from typing import Any, Literal

from llama_vllm.config.preflight import PreflightValidationError, validate_training_preflight
from llama_vllm.config.schemas import load_config

TaskType = Literal["distill", "finetune", "infer"]


def _normalize_overrides(overrides: list[str] | None) -> list[str]:
    """Normalize dashboard override inputs.

    Accepts either:
      - key=value
      - --override key=value
    """
    normalized: list[str] = []
    for item in overrides or []:
        text = str(item).strip()
        if not text:
            continue
        if text.startswith("--override "):
            text = text[len("--override ") :].strip()
        normalized.append(text)
    return normalized


def _resolve_shell_style(shell_style: str) -> str:
    style = (shell_style or "auto").lower()
    if style == "auto":
        return "powershell" if os.name == "nt" else "posix"
    if style in {"powershell", "posix"}:
        return style
    return "powershell" if os.name == "nt" else "posix"


def _quote_arg(value: str, shell_style: str) -> str:
    text = str(value)
    if shell_style == "powershell":
        escaped = text.replace("`", "``").replace('"', '`"')
        return f'"{escaped}"'
    escaped = text.replace("'", "'\"'\"'")
    return f"'{escaped}'"


def _task_to_config_type(task_type: TaskType) -> str:
    if task_type == "distill":
        return "distillation"
    if task_type == "finetune":
        return "finetuning"
    return "inference"


def build_command_preview(
    task_type: TaskType,
    config_path: str,
    overrides: list[str] | None = None,
    shell_style: str = "auto",
) -> str:
    """Build runnable command preview for selected task."""
    overrides = _normalize_overrides(overrides)
    shell = _resolve_shell_style(shell_style)

    if task_type == "distill":
        base = "llama-vllm distill run"
    elif task_type == "finetune":
        base = "llama-vllm finetune run"
    else:
        cfg = load_config(config_path, config_type="inference", overrides=overrides)
        mode = getattr(cfg, "mode", "server")
        mode_map = {"batch": "infer batch", "streaming": "infer stream", "server": "infer serve"}
        base = f"llama-vllm {mode_map.get(mode, 'infer serve')}"

    parts = [base, f"--config {_quote_arg(config_path, shell)}"]
    for item in overrides:
        parts.append(f"--override {_quote_arg(item, shell)}")
    return " ".join(parts)


def run_preflight(
    task_type: TaskType,
    config_path: str,
    overrides: list[str] | None = None,
    shell_style: str = "auto",
) -> dict[str, Any]:
    """Run task preflight and return structured result for UI/API."""
    overrides = _normalize_overrides(overrides)
    config_type = _task_to_config_type(task_type)

    try:
        cfg = load_config(config_path, config_type=config_type, overrides=overrides)
    except Exception as exc:
        return {
            "ok": False,
            "task_type": task_type,
            "message": f"Config load failed: {exc}",
            "errors": [str(exc)],
            "suggestions": [],
            "formatted_suggestions": [],
        }

    if task_type == "infer":
        return {
            "ok": True,
            "task_type": task_type,
            "message": "Inference config validated. Training preflight is skipped for infer task.",
            "errors": [],
            "suggestions": [],
            "formatted_suggestions": [],
        }

    base_command = "llama-vllm distill run" if task_type == "distill" else "llama-vllm finetune run"
    try:
        validate_training_preflight(
            cfg,
            base_command=base_command,
            config_path=config_path,
            overrides=overrides,
            shell_style=shell_style,
        )
        return {
            "ok": True,
            "task_type": task_type,
            "message": "Preflight passed.",
            "errors": [],
            "suggestions": [],
            "formatted_suggestions": [],
        }
    except PreflightValidationError as exc:
        return {
            "ok": False,
            "task_type": task_type,
            "message": str(exc),
            "errors": exc.errors,
            "suggestions": exc.suggestions,
            "formatted_suggestions": exc.formatted_suggestions,
        }

