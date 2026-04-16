"""Dashboard module with lazy exports."""

__all__ = ["create_dashboard_app"]


def __getattr__(name: str):
	if name == "create_dashboard_app":
		from llama_vllm.dashboard.app import create_dashboard_app

		return create_dashboard_app
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


