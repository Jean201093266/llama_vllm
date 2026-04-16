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
    with pytest.raises(ValueError, match="Suggested fix"):
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


def test_preflight_passes_when_precision_disabled_and_deps_present(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(
        root / "configs" / "finetuning" / "lora.yaml",
        config_type="finetuning",
    )
    cfg.method = "sft"
    cfg.training.bf16 = False
    cfg.training.fp16 = False
    monkeypatch.setattr(importlib.util, "find_spec", _mock_find_spec_factory(set()))
    validate_training_preflight(cfg)


def test_preflight_precision_error_includes_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(
        root / "configs" / "finetuning" / "lora.yaml",
        config_type="finetuning",
    )
    cfg.training.bf16 = True
    cfg.training.fp16 = False

    monkeypatch.setattr(importlib.util, "find_spec", _mock_find_spec_factory(set()))
    monkeypatch.setattr(
        "llama_vllm.config.preflight.probe_training_capabilities",
        lambda: {
            "torch_installed": False,
            "cuda_available": False,
            "gpu_count": 0,
            "bf16_supported": False,
            "fp16_supported": False,
            "diagnostics": "torch is not installed",
            "notes": ["torch is not installed."],
        },
    )

    with pytest.raises(ValueError) as exc_info:
        validate_training_preflight(
            cfg,
            base_command="llama-vllm finetune run",
            config_path="configs/finetuning/lora.yaml",
        )

    message = str(exc_info.value)
    assert "Detected runtime: torch is not installed" in message
    assert "Quick override suggestions:" in message
    assert (
        "llama-vllm finetune run --config \"configs/finetuning/lora.yaml\" --override \"training.bf16=false\""
        in message
    )


def test_preflight_suggestions_fallback_to_override_when_no_cli_context(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(
        root / "configs" / "finetuning" / "lora.yaml",
        config_type="finetuning",
    )
    cfg.training.bf16 = True
    cfg.training.fp16 = False

    monkeypatch.setattr(importlib.util, "find_spec", _mock_find_spec_factory(set()))
    monkeypatch.setattr(
        "llama_vllm.config.preflight.probe_training_capabilities",
        lambda: {
            "torch_installed": False,
            "cuda_available": False,
            "gpu_count": 0,
            "bf16_supported": False,
            "fp16_supported": False,
            "diagnostics": "torch is not installed",
            "notes": ["torch is not installed."],
        },
    )

    with pytest.raises(ValueError) as exc_info:
        validate_training_preflight(cfg)

    message = str(exc_info.value)
    assert "--override training.bf16=false" in message


def test_preflight_suggestion_overrides_duplicate_existing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(
        root / "configs" / "finetuning" / "lora.yaml",
        config_type="finetuning",
    )
    cfg.training.bf16 = True
    cfg.training.fp16 = False

    monkeypatch.setattr(importlib.util, "find_spec", _mock_find_spec_factory(set()))
    monkeypatch.setattr(
        "llama_vllm.config.preflight.probe_training_capabilities",
        lambda: {
            "torch_installed": False,
            "cuda_available": False,
            "gpu_count": 0,
            "bf16_supported": False,
            "fp16_supported": False,
            "diagnostics": "torch is not installed",
            "notes": ["torch is not installed."],
        },
    )

    with pytest.raises(ValueError) as exc_info:
        validate_training_preflight(
            cfg,
            base_command="llama-vllm finetune run",
            config_path="configs/finetuning/lora.yaml",
            overrides=["training.bf16=true", "training.learning_rate=1e-4"],
        )

    message = str(exc_info.value)
    assert "--override \"training.bf16=true\"" not in message
    assert "--override \"training.bf16=false\"" in message
    assert "--override \"training.learning_rate=1e-4\"" in message


def test_preflight_quotes_path_and_override_values_with_spaces(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(
        root / "configs" / "finetuning" / "lora.yaml",
        config_type="finetuning",
    )
    cfg.training.bf16 = True
    cfg.training.fp16 = False

    monkeypatch.setattr(importlib.util, "find_spec", _mock_find_spec_factory(set()))
    monkeypatch.setattr(
        "llama_vllm.config.preflight.probe_training_capabilities",
        lambda: {
            "torch_installed": False,
            "cuda_available": False,
            "gpu_count": 0,
            "bf16_supported": False,
            "fp16_supported": False,
            "diagnostics": "torch is not installed",
            "notes": ["torch is not installed."],
        },
    )

    with pytest.raises(ValueError) as exc_info:
        validate_training_preflight(
            cfg,
            base_command="llama-vllm finetune run",
            config_path="configs/finetuning/lora with space.yaml",
            overrides=["output_dir=./outputs/my run"],
            shell_style="powershell",
        )

    message = str(exc_info.value)
    assert '--config "configs/finetuning/lora with space.yaml"' in message
    assert '--override "output_dir=./outputs/my run"' in message


