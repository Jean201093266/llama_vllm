"""CLI package with lazy exports."""

__all__ = ["app"]


def __getattr__(name: str):
	if name == "app":
		from llama_vllm.cli.main import app

		return app
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

