"""CLI commands for vLLM inference and serving."""

from __future__ import annotations

import typer

from llama_vllm.config.schemas import load_config

app = typer.Typer(help="Inference commands")


@app.command("batch")
def batch(
    config: str = typer.Option(..., "--config", "-c"),
    input_file: str | None = typer.Option(None, "--input"),
    output_file: str | None = typer.Option(None, "--output"),
    override: list[str] = typer.Option(None, "--override"),
) -> None:
    from llama_vllm.inference.batch import run_batch_inference

    cfg = load_config(config, config_type="inference", overrides=override or [])
    if input_file:
        cfg.input_file = input_file
    if output_file:
        cfg.output_file = output_file
    run_batch_inference(cfg)


@app.command("stream")
def stream(
    config: str = typer.Option(..., "--config", "-c"),
    prompt: str = typer.Option(..., "--prompt"),
    override: list[str] = typer.Option(None, "--override"),
) -> None:
    from llama_vllm.inference.streaming import stream_text

    cfg = load_config(config, config_type="inference", overrides=override or [])
    for chunk in stream_text(cfg, prompt):
        typer.echo(chunk, nl=False)
    typer.echo()


@app.command("serve")
def serve(
    config: str = typer.Option(..., "--config", "-c"),
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
    override: list[str] = typer.Option(None, "--override"),
) -> None:
    from llama_vllm.inference.server import run_server

    cfg = load_config(config, config_type="inference", overrides=override or [])
    if host:
        cfg.server.host = host
    if port:
        cfg.server.port = port
    run_server(cfg)

