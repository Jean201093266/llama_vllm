"""Typer application root."""

from __future__ import annotations

import typer

from llama_vllm.cli.distill import app as distill_app
from llama_vllm.cli.finetune import app as finetune_app
from llama_vllm.cli.infer import app as infer_app

app = typer.Typer(
    help="Comprehensive framework for distillation, fine-tuning, and inference using vLLM + LLaMA Factory"
)

app.add_typer(distill_app, name="distill")
app.add_typer(finetune_app, name="finetune")
app.add_typer(infer_app, name="infer")


@app.command("dashboard")
def dashboard(
    host: str = typer.Option("127.0.0.1", "--host", help="Dashboard host"),
    port: int = typer.Option(7860, "--port", help="Dashboard port"),
    db_path: str = typer.Option(".dashboard/history.db", "--db-path", help="SQLite path for dashboard history"),
) -> None:
    """Launch dry-run visual dashboard for preflight and command preview."""
    import uvicorn

    from llama_vllm.dashboard.app import create_dashboard_app

    uvicorn.run(create_dashboard_app(db_path=db_path), host=host, port=port)


@app.command("serve")
def serve_alias(
    config: str = typer.Option(..., "--config", "-c"),
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
) -> None:
    from llama_vllm.cli.infer import serve

    serve(config=config, host=host, port=port, override=[])


if __name__ == "__main__":
    app()

