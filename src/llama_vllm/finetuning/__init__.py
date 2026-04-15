"""Fine-tuning package with lazy exports."""

__all__ = ["run_finetuning"]


def __getattr__(name: str):
	if name == "run_finetuning":
		from llama_vllm.finetuning.trainer import run_finetuning

		return run_finetuning
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

