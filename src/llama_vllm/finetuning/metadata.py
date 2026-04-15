"""Run metadata persistence for training jobs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def build_run_metadata(
    *,
    stage: str,
    config: Any,
    resume_from_checkpoint: Optional[str] = None,
    status: str = "started",
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "stage": stage,
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": getattr(config, "output_dir", None),
        "resume_from_checkpoint": resume_from_checkpoint,
    }
    if hasattr(config, "model_dump"):
        payload["config"] = config.model_dump()
    else:
        payload["config"] = dict(config)
    if extras:
        payload.update(extras)
    return payload


def write_run_metadata(output_dir: str, filename: str, payload: Dict[str, Any]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

