from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llama_vllm.dashboard.service import build_command_preview, run_preflight


def test_command_preview_distill_contains_config_and_overrides() -> None:
    command = build_command_preview(
        "distill",
        "configs/distillation/feature_distill.yaml",
        ["training.bf16=false", "output_dir=./outputs/my run"],
        shell_style="powershell",
    )
    assert command.startswith("llama-vllm distill run")
    assert '--config "configs/distillation/feature_distill.yaml"' in command
    assert '--override "training.bf16=false"' in command
    assert '--override "output_dir=./outputs/my run"' in command


def test_command_preview_accepts_override_prefix_form() -> None:
    command = build_command_preview(
        "distill",
        "configs/distillation/feature_distill.yaml",
        ["--override training.bf16=false"],
        shell_style="powershell",
    )
    assert '--override "training.bf16=false"' in command


def test_preflight_infer_skips_training_checks() -> None:
    result = run_preflight("infer", "configs/inference/server.yaml", [], shell_style="auto")
    assert result["ok"] is True
    assert "skipped" in result["message"].lower() or "validated" in result["message"].lower()


def test_preflight_distill_failure_has_suggestions() -> None:
    result = run_preflight("distill", "configs/distillation/feature_distill.yaml", [], shell_style="auto")
    assert result["ok"] is False
    assert len(result["errors"]) >= 1
    assert isinstance(result["formatted_suggestions"], list)

