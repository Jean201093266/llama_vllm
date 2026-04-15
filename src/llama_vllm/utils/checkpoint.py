"""Checkpoint save/resume/export utilities."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

from llama_vllm.utils.logging import get_logger

logger = get_logger(__name__)

LATEST_MANIFEST = "latest_checkpoint.json"
BEST_MANIFEST = "best_checkpoint.json"


def _extract_step(checkpoint_path: Optional[str]) -> Optional[int]:
    if not checkpoint_path:
        return None
    name = Path(checkpoint_path).name
    if not name.startswith("checkpoint-"):
        return None
    try:
        return int(name.split("-")[-1])
    except ValueError:
        return None


def write_checkpoint_manifest(
    output_dir: str,
    kind: str,
    checkpoint_path: str,
    *,
    metric: Optional[float] = None,
) -> str:
    """Persist a latest/best checkpoint manifest."""
    filename = LATEST_MANIFEST if kind == "latest" else BEST_MANIFEST
    payload = {
        "kind": kind,
        "path": checkpoint_path,
        "step": _extract_step(checkpoint_path),
        "metric": metric,
    }
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def read_checkpoint_manifest(output_dir: str, kind: str = "latest") -> Optional[dict]:
    """Load a latest/best checkpoint manifest if present and still valid."""
    filename = LATEST_MANIFEST if kind == "latest" else BEST_MANIFEST
    path = os.path.join(output_dir, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    checkpoint_path = data.get("path")
    if checkpoint_path and os.path.exists(checkpoint_path):
        return data
    return None


def refresh_checkpoint_manifests(
    output_dir: str,
    *,
    latest_checkpoint: Optional[str] = None,
    best_checkpoint: Optional[str] = None,
    best_metric: Optional[float] = None,
) -> None:
    """Refresh latest/best manifests based on trainer output."""
    if latest_checkpoint and os.path.exists(latest_checkpoint):
        write_checkpoint_manifest(output_dir, "latest", latest_checkpoint)
    if best_checkpoint and os.path.exists(best_checkpoint):
        write_checkpoint_manifest(output_dir, "best", best_checkpoint, metric=best_metric)


def save_checkpoint(
    model,
    tokenizer,
    output_dir: str,
    step: int,
    is_best: bool = False,
    save_total_limit: int = 3,
    is_lora: bool = False,
) -> str:
    """Save model checkpoint and optionally mark as best."""
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    if is_lora:
        model.save_pretrained(ckpt_dir)
    else:
        model.save_pretrained(ckpt_dir, safe_serialization=True)

    tokenizer.save_pretrained(ckpt_dir)

    # Write checkpoint metadata
    meta = {"step": step, "is_best": is_best}
    with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w") as f:
        json.dump(meta, f)

    write_checkpoint_manifest(output_dir, "latest", ckpt_dir)

    if is_best:
        best_dir = os.path.join(output_dir, "best_checkpoint")
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        shutil.copytree(ckpt_dir, best_dir)
        write_checkpoint_manifest(output_dir, "best", ckpt_dir)
        logger.info(f"✓ Best checkpoint saved → {best_dir}")

    # Enforce save_total_limit
    _cleanup_old_checkpoints(output_dir, save_total_limit)

    logger.info(f"Checkpoint saved → {ckpt_dir}")
    return ckpt_dir


def _cleanup_old_checkpoints(output_dir: str, limit: int) -> None:
    """Remove oldest checkpoints beyond save_total_limit."""
    checkpoints = sorted(
        [d for d in Path(output_dir).iterdir() if d.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    while len(checkpoints) > limit:
        oldest = checkpoints.pop(0)
        shutil.rmtree(oldest)
        logger.debug(f"Removed old checkpoint: {oldest}")


def get_last_checkpoint(output_dir: str) -> Optional[str]:
    """Return path to the latest checkpoint, or None."""
    if not os.path.exists(output_dir):
        return None
    manifest = read_checkpoint_manifest(output_dir, kind="latest")
    if manifest is not None:
        return manifest.get("path")
    checkpoints = sorted(
        [d for d in Path(output_dir).iterdir() if d.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def merge_lora_adapter(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    dtype: str = "bfloat16",
    safe_serialization: bool = True,
) -> str:
    """Merge LoRA adapter weights into the base model and save."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)

    logger.info(f"Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    logger.info(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    logger.info("Merging LoRA weights into base model...")
    merged = model.merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    merged.save_pretrained(output_path, safe_serialization=safe_serialization)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)

    logger.info(f"✓ Merged model saved → {output_path}")
    return output_path

