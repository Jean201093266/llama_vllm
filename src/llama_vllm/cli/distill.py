"""CLI commands for knowledge distillation."""

from __future__ import annotations

import typer

from llama_vllm.config.schemas import load_config
from llama_vllm.config.preflight import validate_training_preflight

app = typer.Typer(help="Knowledge distillation commands")


@app.command("run")
def run(
    config: str = typer.Option(..., "--config", "-c", help="YAML config path"),
    override: list[str] = typer.Option(None, "--override", help="Override values like key=value"),
) -> None:
    cfg = load_config(config, config_type="distillation", overrides=override or [])
    validate_training_preflight(cfg)

    from llama_vllm.distillation.trainer import run_distillation

    run_distillation(cfg)

