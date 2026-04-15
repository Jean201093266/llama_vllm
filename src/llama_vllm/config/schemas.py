"""Configuration schemas using Pydantic v2."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────
# Shared sub-schemas
# ─────────────────────────────────────────────

class DataArgs(BaseModel):
    """Dataset configuration shared across training modes."""
    dataset_name_or_path: str = Field(..., description="HF dataset name or local path")
    dataset_format: Literal["alpaca", "sharegpt", "openai", "dpo_pairs", "raw"] = "alpaca"
    train_split: str = "train"
    eval_split: Optional[str] = "validation"
    max_samples: Optional[int] = None
    max_seq_length: int = 2048
    preprocessing_num_workers: int = 4
    cache_dir: Optional[str] = None
    system_prompt: Optional[str] = None
    input_key: str = "input"
    output_key: str = "output"
    # DPO specific
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"


class TrainingArgs(BaseModel):
    """Training hyperparameters mapped to HF TrainingArguments."""
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    report_to: List[str] = Field(default_factory=lambda: ["tensorboard"])
    dataloader_num_workers: int = 4
    group_by_length: bool = True
    seed: int = 42
    deepspeed: Optional[str] = None
    fsdp: Optional[str] = None
    fsdp_config: Optional[Dict[str, Any]] = None


class LoRAArgs(BaseModel):
    """LoRA/QLoRA specific parameters."""
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Union[List[str], Literal["auto"]] = "auto"
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"


class QuantizationArgs(BaseModel):
    """Quantization configuration."""
    bits: Optional[Literal[4, 8]] = None
    quant_type: Literal["nf4", "fp4", "int8"] = "nf4"
    double_quant: bool = True
    compute_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"


# ─────────────────────────────────────────────
# Distillation Config
# ─────────────────────────────────────────────

class DistillationConfig(BaseModel):
    """Configuration for knowledge distillation training."""
    # Models
    teacher_model: str = Field(..., description="Teacher model path or HF hub ID")
    student_model: str = Field(..., description="Student model path or HF hub ID")

    # Distillation type
    distill_type: Literal["logit", "feature", "combined"] = "logit"
    temperature: float = Field(default=4.0, ge=0.1, le=100.0, description="Softening temperature")
    alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for distillation loss")

    # Feature distillation
    feature_layers: List[int] = Field(default_factory=list, description="Layer indices for feature distillation")
    feature_loss_type: Literal["mse", "cosine"] = "mse"
    project_hidden: bool = False  # Use linear projection if hidden dims differ

    # Teacher backend
    use_vllm_teacher: bool = True
    teacher_tensor_parallel_size: int = 1
    teacher_dtype: Literal["auto", "float16", "bfloat16"] = "bfloat16"

    # Student LoRA (optional — distill into LoRA student)
    use_lora_student: bool = False
    lora: LoRAArgs = Field(default_factory=LoRAArgs)
    quantization: QuantizationArgs = Field(default_factory=QuantizationArgs)

    # Data & Training
    data: DataArgs = Field(...)
    training: TrainingArgs = Field(default_factory=TrainingArgs)
    output_dir: str = "./outputs/distillation"
    resume_from_checkpoint: Optional[str] = None


# ─────────────────────────────────────────────
# Fine-tuning Config
# ─────────────────────────────────────────────

class FineTuningConfig(BaseModel):
    """Configuration for supervised fine-tuning and alignment."""
    model_name_or_path: str = Field(..., description="Base model path or HF hub ID")
    method: Literal["sft", "lora", "qlora", "dpo", "rlhf"] = "lora"

    # LoRA/QLoRA params
    lora: LoRAArgs = Field(default_factory=LoRAArgs)
    quantization: QuantizationArgs = Field(default_factory=QuantizationArgs)

    # DPO-specific
    dpo_beta: float = Field(default=0.1, ge=0.0, le=1.0)
    ref_model_path: Optional[str] = None
    dpo_loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair"] = "sigmoid"

    # RLHF-specific
    reward_model_path: Optional[str] = None
    ppo_epochs: int = 1
    kl_penalty: str = "kl"

    # Data & Training
    data: DataArgs = Field(...)
    training: TrainingArgs = Field(default_factory=TrainingArgs)
    output_dir: str = "./outputs/finetuning"
    resume_from_checkpoint: Optional[str] = None

    # LLaMA Factory compatibility
    use_llamafactory: bool = False
    llamafactory_args: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Inference Config
# ─────────────────────────────────────────────

class SamplingConfig(BaseModel):
    """Default sampling parameters for vLLM."""
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = -1
    max_tokens: int = 512
    stop: List[str] = Field(default_factory=list)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    best_of: int = 1
    use_beam_search: bool = False


class LoRAModuleConfig(BaseModel):
    """Hot-loadable LoRA adapter for vLLM."""
    name: str
    path: str
    base_model_name: Optional[str] = None


class ServerConfig(BaseModel):
    """FastAPI server settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    api_key: Optional[str] = None
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    max_concurrent_requests: int = 256
    timeout: int = 300


class InferenceConfig(BaseModel):
    """Configuration for vLLM-based inference."""
    model_name_or_path: str = Field(..., description="Model path or HF hub ID")
    mode: Literal["batch", "streaming", "server"] = "server"

    # vLLM engine parameters
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = Field(default=0.90, ge=0.1, le=1.0)
    max_model_len: Optional[int] = None
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    quantization: Optional[Literal["gptq", "awq", "squeezellm", "fp8"]] = None
    enforce_eager: bool = False
    trust_remote_code: bool = False
    swap_space: int = 4

    # LoRA adapters
    enable_lora: bool = False
    lora_modules: List[LoRAModuleConfig] = Field(default_factory=list)
    max_lora_rank: int = 64

    # Sampling defaults
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)

    # Server config (mode=server)
    server: ServerConfig = Field(default_factory=ServerConfig)

    # Batch inference (mode=batch)
    input_file: Optional[str] = None
    output_file: Optional[str] = "./outputs/inference/results.jsonl"
    batch_size: int = 32
    prompt_key: str = "prompt"


# ─────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────

_CONFIG_MAP = {
    "distillation": DistillationConfig,
    "finetuning": FineTuningConfig,
    "inference": InferenceConfig,
}


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _parse_override(override: str) -> tuple[str, Any]:
    """Parse a 'key=value' or 'nested.key=value' override string."""
    if "=" not in override:
        raise ValueError(f"Invalid override format '{override}'. Expected 'key=value'.")
    key, _, raw_value = override.partition("=")
    # Try to parse as int, float, bool, then fall back to string
    for converter in (int, float):
        try:
            return key, converter(raw_value)
        except ValueError:
            pass
    if raw_value.lower() in ("true", "yes"):
        return key, True
    if raw_value.lower() in ("false", "no"):
        return key, False
    return key, raw_value


def _apply_overrides(data: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply CLI override strings to a config dict."""
    for override in overrides or []:
        key, value = _parse_override(override)
        # Support nested keys with dot notation
        keys = key.split(".")
        target = data
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        target[keys[-1]] = value
    return data


def load_config(
    path: Union[str, Path],
    config_type: Optional[Literal["distillation", "finetuning", "inference"]] = None,
    overrides: Optional[List[str]] = None,
) -> Union[DistillationConfig, FineTuningConfig, InferenceConfig]:
    """
    Load and validate a YAML config file.

    Args:
        path: Path to YAML configuration file.
        config_type: Explicit config type; auto-detected from 'type' key if None.
        overrides: List of 'key=value' strings to override config values.

    Returns:
        Validated config object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    # Auto-detect type from YAML 'type' key or from file path
    if config_type is None:
        config_type = data.pop("type", None)
        if config_type is None:
            for key in _CONFIG_MAP:
                if key in str(path):
                    config_type = key
                    break
        if config_type is None:
            raise ValueError(
                "Cannot determine config type. Add 'type: distillation|finetuning|inference' "
                "to your YAML file or pass config_type explicitly."
            )

    if config_type not in _CONFIG_MAP:
        raise ValueError(f"Unknown config type '{config_type}'. Choose from {list(_CONFIG_MAP.keys())}")

    # Apply CLI overrides
    if overrides:
        data = _apply_overrides(data, overrides)

    return _CONFIG_MAP[config_type](**data)

