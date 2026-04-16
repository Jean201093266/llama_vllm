"""CLI commands for model fine-tuning."""

from __future__ import annotations

import typer

from llama_vllm.cli.common import apply_first_suggestion, format_auto_fix_message
from llama_vllm.config.schemas import load_config
from llama_vllm.config.preflight import PreflightValidationError, validate_training_preflight

app = typer.Typer(help="Fine-tuning commands")


@app.command("run")
def run(
    config: str = typer.Option(..., "--config", "-c", help="YAML config path"),
    override: list[str] = typer.Option(None, "--override", help="Override values like key=value"),
    auto_fix: bool = typer.Option(False, "--auto-fix", help="Print remediation command suggestions and exit on preflight failure"),
    show_raw: bool = typer.Option(False, "--show-raw", help="With --auto-fix, also print raw --override suggestions"),
    apply_overrides: bool = typer.Option(False, "--apply-overrides", help="With --auto-fix, apply suggestion #1 and re-run preflight only"),
) -> None:
    if apply_overrides and not auto_fix:
        raise typer.BadParameter("--apply-overrides requires --auto-fix")

    cfg = load_config(config, config_type="finetuning", overrides=override or [])
    try:
        validate_training_preflight(
            cfg,
            base_command="llama-vllm finetune run",
            config_path=config,
            overrides=override or [],
        )
    except PreflightValidationError as exc:
        if not auto_fix:
            raise
        typer.echo(format_auto_fix_message(exc, show_raw=show_raw))

        if apply_overrides and exc.suggestions:
            merged_overrides = apply_first_suggestion(override or [], exc.suggestions)
            typer.echo("Applying suggestion #1 and re-running preflight (dry-run)...")
            cfg_retry = load_config(config, config_type="finetuning", overrides=merged_overrides)
            validate_training_preflight(
                cfg_retry,
                base_command="llama-vllm finetune run",
                config_path=config,
                overrides=merged_overrides,
            )
            typer.echo("Preflight passed after applying suggestion #1. Training is not started in --apply-overrides mode.")
            raise typer.Exit(code=0)

        raise typer.Exit(code=2)

    from llama_vllm.finetuning.trainer import run_finetuning

    run_finetuning(cfg)


@app.command("export")
def export(
    base_model: str = typer.Option(..., "--base-model", help="Base model path"),
    adapter: str = typer.Option(..., "--adapter", help="LoRA adapter path"),
    output: str = typer.Option(..., "--output", help="Merged model output path"),
    dtype: str = typer.Option("bfloat16", "--dtype", help="Output dtype"),
) -> None:
    from llama_vllm.utils.checkpoint import merge_lora_adapter

    merge_lora_adapter(base_model, adapter, output, dtype=dtype)

