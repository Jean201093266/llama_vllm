from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llama_vllm.cli.common import apply_first_suggestion, format_auto_fix_message
from llama_vllm.config.preflight import PreflightValidationError


def test_format_auto_fix_message_with_multiple_commands() -> None:
    exc = PreflightValidationError(
        "ignored",
        errors=["a", "b"],
        suggestions=["--override x=1", "--override y=2"],
        formatted_suggestions=["cmd1", "cmd2"],
    )
    output = format_auto_fix_message(exc)
    assert "Preflight validation failed:" in output
    assert "- a" in output
    assert "- b" in output
    assert "Recommended fix command (1):" in output
    assert "1) cmd1" in output
    assert "Other fix commands:" in output
    assert "2) cmd2" in output
    assert "Raw override suggestions:" not in output


def test_format_auto_fix_message_show_raw() -> None:
    exc = PreflightValidationError(
        "ignored",
        errors=["a"],
        suggestions=["--override x=1", "--override y=2"],
        formatted_suggestions=["cmd1"],
    )
    output = format_auto_fix_message(exc, show_raw=True)
    assert "Raw override suggestions:" in output
    assert "1) --override x=1" in output
    assert "2) --override y=2" in output


def test_apply_first_suggestion_replaces_duplicate_key() -> None:
    merged = apply_first_suggestion(
        ["training.bf16=true", "training.learning_rate=1e-4"],
        ["--override training.bf16=false", "--override training.fp16=true"],
    )
    assert "training.bf16=true" not in merged
    assert "training.bf16=false" in merged
    assert "training.learning_rate=1e-4" in merged


