from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llama_vllm.config.schemas import load_config


def test_training_args_reject_bf16_and_fp16(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError):
        load_config(
            root / "configs" / "finetuning" / "lora.yaml",
            config_type="finetuning",
            overrides=["training.bf16=true", "training.fp16=true"],
        )


def test_rlhf_requires_reward_model_path() -> None:
    root = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError):
        load_config(
            root / "configs" / "finetuning" / "lora.yaml",
            config_type="finetuning",
            overrides=["method=rlhf"],
        )


def test_dpo_requires_supported_dataset_format() -> None:
    root = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError):
        load_config(
            root / "configs" / "finetuning" / "dpo.yaml",
            config_type="finetuning",
            overrides=["data.dataset_format=alpaca"],
        )


def test_qlora_requires_4bit_quantization() -> None:
    root = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError):
        load_config(
            root / "configs" / "finetuning" / "qlora.yaml",
            config_type="finetuning",
            overrides=["quantization.bits=8"],
        )


def test_feature_distillation_requires_hf_teacher() -> None:
    root = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError):
        load_config(
            root / "configs" / "distillation" / "feature_distill.yaml",
            config_type="distillation",
            overrides=["use_vllm_teacher=true"],
        )


