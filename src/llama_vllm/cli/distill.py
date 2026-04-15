"""CLI commands for knowledge distillation."""

from __future__ import annotations

import typer

from llama_vllm.config.schemas import load_config

app = typer.Typer(help="Knowledge distillation commands")


@app.command("run")
def run(
    config: str = typer.Option(..., "--config", "-c", help="YAML config path"),
    override: list[str] = typer.Option(None, "--override", help="Override values like key=value"),
) -> None:
    from llama_vllm.distillation.trainer import run_distillation

    cfg = load_config(config, config_type="distillation", overrides=override or [])
    run_distillation(cfg)

