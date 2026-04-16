"""Lightweight preflight validation for training-oriented configs."""

from __future__ import annotations

import importlib.util
import os
from typing import Optional, Union

from llama_vllm.config.schemas import DistillationConfig, FineTuningConfig
from llama_vllm.utils.hardware import probe_training_capabilities


class PreflightValidationError(ValueError):
    """Structured preflight validation error carrying remediation suggestions."""

    def __init__(
        self,
        message: str,
        *,
        errors: list[str],
        suggestions: list[str],
        formatted_suggestions: list[str],
    ) -> None:
        super().__init__(message)
        self.errors = errors
        self.suggestions = suggestions
        self.formatted_suggestions = formatted_suggestions


def validate_training_preflight(
    config: Union[DistillationConfig, FineTuningConfig],
    *,
    base_command: Optional[str] = None,
    config_path: Optional[str] = None,
    overrides: Optional[list[str]] = None,
    shell_style: str = "auto",
) -> None:
    """Run lightweight, dependency-free validation before importing heavy training stacks."""
    errors = []

    if isinstance(config, DistillationConfig):
        errors.extend(_validate_distillation_preflight(config))
    elif isinstance(config, FineTuningConfig):
        errors.extend(_validate_finetuning_preflight(config))

    capabilities = probe_training_capabilities()
    errors.extend(_validate_precision_requirements(config, capabilities))
    suggestions = _build_override_suggestions(config, capabilities)
    formatted_suggestions = _format_suggestions(
        suggestions,
        base_command=base_command,
        config_path=config_path,
        overrides=overrides or [],
        shell_style=shell_style,
    )

    if errors:
        diagnostics = capabilities.get("diagnostics")
        if diagnostics:
            errors.append(f"Detected runtime: {diagnostics}.")
        details = "\n".join(f"- {item}" for item in errors)
        if formatted_suggestions:
            suggestion_lines = "\n".join(f"  {item}" for item in formatted_suggestions)
            message = f"Preflight validation failed:\n{details}\nQuick override suggestions:\n{suggestion_lines}"
            raise PreflightValidationError(
                message,
                errors=errors,
                suggestions=suggestions,
                formatted_suggestions=formatted_suggestions,
            )

        message = f"Preflight validation failed:\n{details}"
        raise PreflightValidationError(
            message,
            errors=errors,
            suggestions=suggestions,
            formatted_suggestions=formatted_suggestions,
        )


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
        errors.append("`use_lora_student=true` requires optional dependency `peft`. Suggested fix: install `peft` or disable LoRA student.")
    if config.use_lora_student and config.quantization.bits in {4, 8} and not check_optional_dependency("bitsandbytes"):
        errors.append("Quantized LoRA distillation requires optional dependency `bitsandbytes`. Suggested fix: install `bitsandbytes` or disable quantization.")

    if config.use_vllm_teacher and not check_optional_dependency("vllm"):
        errors.append("`use_vllm_teacher=true` requires optional dependency `vllm`. Suggested fix: install `vllm` or set `use_vllm_teacher=false`.")

    return errors


def _validate_finetuning_preflight(config: FineTuningConfig) -> list[str]:
    errors: list[str] = []

    if config.method == "qlora" and config.quantization.bits != 4:
        errors.append("QLoRA requires `quantization.bits: 4`.")

    if config.method in {"lora", "qlora"} and not check_optional_dependency("peft"):
        errors.append(f"`method={config.method}` requires optional dependency `peft`. Suggested fix: install `peft`.")

    if config.method == "qlora" and not check_optional_dependency("bitsandbytes"):
        errors.append("`method=qlora` requires optional dependency `bitsandbytes`. Suggested fix: install `bitsandbytes` or switch to `method=lora`.")

    if config.method in {"dpo", "rlhf"} and not check_optional_dependency("trl"):
        errors.append(f"`method={config.method}` requires optional dependency `trl`. Suggested fix: install `trl` or switch to SFT/LoRA methods.")

    return errors


def _validate_precision_requirements(
    config: Union[DistillationConfig, FineTuningConfig],
    capabilities: dict,
) -> list[str]:
    """Validate bf16/fp16 settings against runtime hardware capabilities."""
    errors: list[str] = []
    training = config.training

    if training.bf16:
        if not capabilities.get("torch_installed", False):
            errors.append("`training.bf16=true` requires `torch` to be installed. Suggested fix: install CUDA-enabled `torch` or set `training.bf16=false`.")
        elif not capabilities.get("cuda_available", False):
            errors.append("`training.bf16=true` requires CUDA-capable hardware. Suggested fix: set `training.bf16=false` or run on a CUDA GPU.")
        elif not capabilities.get("bf16_supported", False):
            errors.append(
                "`training.bf16=true` but bf16 support is not detected on this runtime. "
                "Suggested fix: set `training.bf16=false` and optionally `training.fp16=true`, or use bf16-capable GPUs."
            )

    if training.fp16:
        if not capabilities.get("torch_installed", False):
            errors.append("`training.fp16=true` requires `torch` to be installed. Suggested fix: install CUDA-enabled `torch` or set `training.fp16=false`.")
        elif not capabilities.get("cuda_available", False):
            errors.append("`training.fp16=true` requires CUDA-capable hardware. Suggested fix: set `training.fp16=false` or run on a CUDA GPU.")
        elif not capabilities.get("fp16_supported", False):
            errors.append("`training.fp16=true` but fp16 support is not detected in this runtime. Suggested fix: disable fp16 or verify CUDA stack.")

    return errors


def _build_override_suggestions(
    config: Union[DistillationConfig, FineTuningConfig],
    capabilities: dict,
) -> list[str]:
    """Build CLI --override suggestions for common precision mismatch failures."""
    suggestions: list[str] = []
    training = config.training

    if training.bf16 and (
        not capabilities.get("torch_installed", False)
        or not capabilities.get("cuda_available", False)
        or not capabilities.get("bf16_supported", False)
    ):
        suggestions.append("--override training.bf16=false")
        if capabilities.get("cuda_available", False):
            suggestions.append("--override training.fp16=true")

    if training.fp16 and (
        not capabilities.get("torch_installed", False)
        or not capabilities.get("cuda_available", False)
        or not capabilities.get("fp16_supported", False)
    ):
        suggestions.append("--override training.fp16=false")

    # Keep order stable while removing duplicates
    return list(dict.fromkeys(suggestions))


def _format_suggestions(
    override_suggestions: list[str],
    *,
    base_command: Optional[str],
    config_path: Optional[str],
    overrides: list[str],
    shell_style: str,
) -> list[str]:
    """Format suggestions as full rerun commands when CLI context is available."""
    if not override_suggestions:
        return []

    if not base_command or not config_path:
        return override_suggestions

    results: list[str] = []
    active_shell = _resolve_shell_style(shell_style)
    for suggestion in override_suggestions:
        merged_overrides = _merge_overrides(overrides, suggestion)
        parts = [base_command, f"--config {_quote_arg(config_path, active_shell)}"]
        for item in merged_overrides:
            parts.append(f"--override {_quote_arg(item, active_shell)}")
        results.append(" ".join(parts))
    return results


def _merge_overrides(existing: list[str], suggestion_flag: str) -> list[str]:
    """Merge overrides keeping order, with suggested key overriding existing value."""
    suggestion = suggestion_flag.strip()
    if suggestion.startswith("--override "):
        suggestion = suggestion[len("--override ") :]

    def _key(item: str) -> str:
        if "=" not in item:
            return item.strip()
        return item.split("=", 1)[0].strip()

    suggestion_key = _key(suggestion)
    merged = [item for item in existing if _key(item) != suggestion_key]
    merged.append(suggestion)
    return merged


def _resolve_shell_style(shell_style: str) -> str:
    """Resolve shell style to 'powershell' or 'posix'."""
    style = (shell_style or "auto").lower()
    if style == "auto":
        return "powershell" if os.name == "nt" else "posix"
    if style in {"powershell", "posix"}:
        return style
    return "powershell" if os.name == "nt" else "posix"


def _quote_arg(value: str, shell_style: str) -> str:
    """Quote command argument for selected shell style."""
    text = str(value)
    if shell_style == "powershell":
        escaped = text.replace("`", "``").replace('"', '`"')
        return f'"{escaped}"'

    # POSIX style
    escaped = text.replace("'", "'\"'\"'")
    return f"'{escaped}'"


