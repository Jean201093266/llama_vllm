"""Inference package with lazy exports."""

__all__ = ["VLLMEngineWrapper", "run_batch_inference", "stream_text", "create_app", "run_server"]


def __getattr__(name: str):
	if name == "VLLMEngineWrapper":
		from llama_vllm.inference.engine import VLLMEngineWrapper

		return VLLMEngineWrapper
	if name == "run_batch_inference":
		from llama_vllm.inference.batch import run_batch_inference

		return run_batch_inference
	if name == "stream_text":
		from llama_vllm.inference.streaming import stream_text

		return stream_text
	if name in {"create_app", "run_server"}:
		from llama_vllm.inference.server import create_app, run_server

		return {"create_app": create_app, "run_server": run_server}[name]
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
