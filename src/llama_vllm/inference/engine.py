"""vLLM engine factory and sampling helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional

from llama_vllm.config.schemas import InferenceConfig
from llama_vllm.utils.logging import get_logger

logger = get_logger(__name__)


class VLLMEngineWrapper:
    """Thin wrapper around vLLM LLM with config-driven defaults."""

    def __init__(self, config: InferenceConfig) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError("vLLM is required for inference. Install with `pip install vllm`.") from exc

        self.config = config
        self._llm = LLM(
            model=config.model_name_or_path,
            tensor_parallel_size=config.tensor_parallel_size,
            pipeline_parallel_size=config.pipeline_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            dtype=config.dtype,
            quantization=config.quantization,
            enable_lora=config.enable_lora,
            max_lora_rank=config.max_lora_rank,
            enforce_eager=config.enforce_eager,
            trust_remote_code=config.trust_remote_code,
            swap_space=config.swap_space,
        )
        logger.info(f"vLLM engine ready: {config.model_name_or_path}")

    @property
    def llm(self):
        return self._llm

    def build_sampling_params(self, overrides: Optional[Dict[str, Any]] = None):
        from vllm import SamplingParams

        values = self.config.sampling.model_dump()
        if overrides:
            values.update({k: v for k, v in overrides.items() if v is not None})
        return SamplingParams(**values)

    def generate(self, prompts: List[str], sampling_overrides: Optional[Dict[str, Any]] = None):
        params = self.build_sampling_params(sampling_overrides)
        return self._llm.generate(prompts, params)


@lru_cache(maxsize=8)
def get_engine(config_key: str, config_json: str):
    """Cache engine instances by config key/json fingerprint."""
    from llama_vllm.config.schemas import InferenceConfig

    config = InferenceConfig.model_validate_json(config_json)
    return VLLMEngineWrapper(config)

