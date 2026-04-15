"""Lightweight preflight validation for training-oriented configs."""

from __future__ import annotations

import importlib.util
from typing import Union

from llama_vllm.config.schemas import DistillationConfig, FineTuningConfig


def validate_training_preflight(config: Union[DistillationConfig, FineTuningConfig]) -> None:
    """Run lightweight, dependency-free validation before importing heavy training stacks."""
    errors = []

    if isinstance(config, DistillationConfig):
        errors.extend(_validate_distillation_preflight(config))
    elif isinstance(config, FineTuningConfig):
        errors.extend(_validate_finetuning_preflight(config))

    if errors:
        details = "\n".join(f"- {item}" for item in errors)
        raise ValueError(f"Preflight validation failed:\n{details}")


def check_optional_dependency(package_name: str) -> bool:
    """Return True when an optional package can be imported in the current environment."""
    return importlib.util.find_spec(package_name) is not None


def _validate_distillation_preflight(config: DistillationConfig) -> list[str]:
    errors: list[str] = []

    if config.distill_type in {"feature", "combined"} and config.use_vllm_teacher:
        errors.append("Feature/combined distillation requires a HuggingFace teacher; set `use_vllm_teacher: false`.")

    if config.distill_type in {"feature", "combined"} and not config.feature_layers:
        errors.append("Feature/combined distillation requires non-empty `feature_layers`.")

    if config.feature_layers:
        if any(layer < 0 for layer in config.feature_layers):
            errors.append("`feature_layers` must contain non-negative indices.")
        if len(set(config.feature_layers)) != len(config.feature_layers):
            errors.append("`feature_layers` must not contain duplicates.")

    if config.use_lora_student and not check_optional_dependency("peft"):
        errors.append("`use_lora_student=true` requires optional dependency `peft`.")
    if config.use_lora_student and config.quantization.bits in {4, 8} and not check_optional_dependency("bitsandbytes"):
        errors.append("Quantized LoRA distillation requires optional dependency `bitsandbytes`.")

    if config.use_vllm_teacher and not check_optional_dependency("vllm"):
        errors.append("`use_vllm_teacher=true` requires optional dependency `vllm`.")

    return errors


def _validate_finetuning_preflight(config: FineTuningConfig) -> list[str]:
    errors: list[str] = []

    if config.method == "qlora" and config.quantization.bits != 4:
        errors.append("QLoRA requires `quantization.bits: 4`.")

    if config.method in {"lora", "qlora"} and not check_optional_dependency("peft"):
        errors.append(f"`method={config.method}` requires optional dependency `peft`.")

    if config.method == "qlora" and not check_optional_dependency("bitsandbytes"):
        errors.append("`method=qlora` requires optional dependency `bitsandbytes`.")

    if config.method in {"dpo", "rlhf"} and not check_optional_dependency("trl"):
        errors.append(f"`method={config.method}` requires optional dependency `trl`.")

    return errors

