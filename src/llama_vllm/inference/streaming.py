"""Simple streaming helpers for prompt inference."""

from __future__ import annotations

from typing import Generator

from llama_vllm.config.schemas import InferenceConfig
from llama_vllm.inference.engine import VLLMEngineWrapper


def stream_text(config: InferenceConfig, prompt: str):
    """Yield token text chunks. Falls back to full text if incremental streaming is unavailable."""
    engine = VLLMEngineWrapper(config)
    outputs = engine.generate([prompt])
    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    for token in text.split():
        yield token + " "

