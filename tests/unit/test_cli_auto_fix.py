from pathlib import Path
import sys

from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llama_vllm.cli.distill import app as distill_app
from llama_vllm.cli.finetune import app as finetune_app
from llama_vllm.config.preflight import PreflightValidationError


def _fake_preflight_error() -> PreflightValidationError:
    return PreflightValidationError(
        "Preflight validation failed:\n- mock error",
        errors=["mock error"],
        suggestions=["--override training.bf16=false"],
        formatted_suggestions=["llama-vllm ..."],
    )


def test_distill_auto_fix_exits_cleanly(monkeypatch) -> None:
    monkeypatch.setattr("llama_vllm.cli.distill.load_config", lambda *args, **kwargs: object())

    def _raise(*args, **kwargs):
        raise _fake_preflight_error()

    monkeypatch.setattr("llama_vllm.cli.distill.validate_training_preflight", _raise)

    runner = CliRunner()
    result = runner.invoke(distill_app, ["--config", "dummy.yaml", "--auto-fix"])
    assert result.exit_code == 2
    assert "Recommended fix command (1):" in result.output
    assert "1) llama-vllm ..." in result.output


def test_finetune_auto_fix_exits_cleanly(monkeypatch) -> None:
    monkeypatch.setattr("llama_vllm.cli.finetune.load_config", lambda *args, **kwargs: object())

    def _raise(*args, **kwargs):
        raise _fake_preflight_error()

    monkeypatch.setattr("llama_vllm.cli.finetune.validate_training_preflight", _raise)

    runner = CliRunner()
    result = runner.invoke(finetune_app, ["run", "--config", "dummy.yaml", "--auto-fix"])
    assert result.exit_code == 2
    assert "Recommended fix command (1):" in result.output
    assert "1) llama-vllm ..." in result.output


def test_distill_auto_fix_show_raw(monkeypatch) -> None:
    monkeypatch.setattr("llama_vllm.cli.distill.load_config", lambda *args, **kwargs: object())

    def _raise(*args, **kwargs):
        raise _fake_preflight_error()

    monkeypatch.setattr("llama_vllm.cli.distill.validate_training_preflight", _raise)

    runner = CliRunner()
    result = runner.invoke(distill_app, ["--config", "dummy.yaml", "--auto-fix", "--show-raw"])
    assert result.exit_code == 2
    assert "Raw override suggestions:" in result.output
    assert "1) --override training.bf16=false" in result.output


def test_distill_apply_overrides_dry_run_success(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_load_config(*args, **kwargs):
        return object()

    def _validate(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise _fake_preflight_error()

    monkeypatch.setattr("llama_vllm.cli.distill.load_config", _fake_load_config)
    monkeypatch.setattr("llama_vllm.cli.distill.validate_training_preflight", _validate)

    runner = CliRunner()
    result = runner.invoke(
        distill_app,
        ["--config", "dummy.yaml", "--auto-fix", "--apply-overrides"],
    )
    assert result.exit_code == 0
    assert "Applying suggestion #1" in result.output
    assert "Preflight passed after applying suggestion #1" in result.output


def test_distill_apply_overrides_requires_auto_fix() -> None:
    runner = CliRunner()
    result = runner.invoke(distill_app, ["--config", "dummy.yaml", "--apply-overrides"])
    assert result.exit_code != 0
    assert "requires --auto-fix" in result.output


