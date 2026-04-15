"""
LLaMA-vLLM: A comprehensive framework for model distillation,
fine-tuning, and inference using vLLM and LLaMA Factory.
"""

__version__ = "0.1.0"
__author__ = "LLaMA-vLLM Team"

from llama_vllm.config.schemas import load_config

__all__ = ["load_config", "__version__"]

