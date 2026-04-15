from pathlib import Path
import sys
import importlib.util

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llama_vllm.config.preflight import validate_training_preflight
from llama_vllm.config.schemas import load_config


def _mock_find_spec_factory(missing: set[str]):
    real_find_spec = importlib.util.find_spec

    def _mock(package_name: str):
        if package_name in missing:
            return None
        return object() if real_find_spec(package_name) is None else real_find_spec(package_name)

    return _mock


def test_preflight_rejects_vllm_feature_distillation(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(
        root / "configs" / "distillation" / "feature_distill.yaml",
        config_type="distillation",
    )
    cfg.use_vllm_teacher = True
    monkeypatch.setattr(importlib.util, "find_spec", _mock_find_spec_factory({"vllm"}))
    with pytest.raises(ValueError):
        validate_training_preflight(cfg)


def test_preflight_rejects_invalid_qlora_bits(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(
        root / "configs" / "finetuning" / "qlora.yaml",
        config_type="finetuning",
    )
    cfg.quantization.bits = 8
    monkeypatch.setattr(importlib.util, "find_spec", _mock_find_spec_factory(set()))
    with pytest.raises(ValueError):
        validate_training_preflight(cfg)


def test_preflight_rejects_missing_peft_for_lora(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(
        root / "configs" / "finetuning" / "lora.yaml",
        config_type="finetuning",
    )
    monkeypatch.setattr(importlib.util, "find_spec", _mock_find_spec_factory({"peft"}))
    with pytest.raises(ValueError):
        validate_training_preflight(cfg)


def test_preflight_rejects_duplicate_feature_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(
        root / "configs" / "distillation" / "feature_distill.yaml",
        config_type="distillation",
    )
    cfg.feature_layers = [8, 8, 16]
    monkeypatch.setattr(importlib.util, "find_spec", _mock_find_spec_factory(set()))
    with pytest.raises(ValueError):
        validate_training_preflight(cfg)

