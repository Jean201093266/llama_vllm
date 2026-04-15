"""CLI commands for model fine-tuning."""

from __future__ import annotations

import typer

from llama_vllm.config.schemas import load_config
from llama_vllm.config.preflight import validate_training_preflight

app = typer.Typer(help="Fine-tuning commands")


@app.command("run")
def run(
    config: str = typer.Option(..., "--config", "-c", help="YAML config path"),
    override: list[str] = typer.Option(None, "--override", help="Override values like key=value"),
) -> None:
    cfg = load_config(config, config_type="finetuning", overrides=override or [])
    validate_training_preflight(cfg)

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

