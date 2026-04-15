"""Runtime helpers for robust training execution."""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Type

from llama_vllm.utils.checkpoint import get_last_checkpoint


def resolve_resume_checkpoint(
    output_dir: str,
    requested_checkpoint: Optional[str] = None,
    auto_resume_from_last_checkpoint: bool = True,
) -> Optional[str]:
    """Resolve the checkpoint path to resume from."""
    if requested_checkpoint:
        import os

        if not os.path.exists(requested_checkpoint):
            raise FileNotFoundError(f"Requested checkpoint does not exist: {requested_checkpoint}")
        return requested_checkpoint

    if auto_resume_from_last_checkpoint:
        return get_last_checkpoint(output_dir)
    return None


def build_trainer_callbacks(training_config: Any, has_eval: bool = True) -> List[Any]:
    """Create trainer callbacks based on config and runtime availability."""
    callbacks: List[Any] = []
    patience = getattr(training_config, "early_stopping_patience", None)
    threshold = getattr(training_config, "early_stopping_threshold", 0.0)
    if has_eval and patience is not None:
        from transformers import EarlyStoppingCallback

        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience, early_stopping_threshold=threshold))
    return callbacks


def build_dpo_trainer_kwargs(
    trainer_cls: Type[Any],
    *,
    model: Any,
    ref_model: Any,
    args: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    beta: float,
    loss_type: str,
) -> Dict[str, Any]:
    """Adapt kwargs to different TRL DPOTrainer signatures across versions."""
    signature = inspect.signature(trainer_cls.__init__)
    params = signature.parameters

    kwargs: Dict[str, Any] = {
        "model": model,
        "ref_model": ref_model,
        "args": args,
        "train_dataset": train_dataset,
    }
    if eval_dataset is not None:
        kwargs["eval_dataset"] = eval_dataset
    if "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    if "beta" in params:
        kwargs["beta"] = beta
    if "loss_type" in params:
        kwargs["loss_type"] = loss_type
    if "max_length" in params:
        kwargs["max_length"] = None
    if "max_prompt_length" in params:
        kwargs["max_prompt_length"] = None
    return kwargs

