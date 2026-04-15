"""Unified model loader supporting HF, LoRA, and quantization."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from llama_vllm.config.schemas import FineTuningConfig, QuantizationArgs
from llama_vllm.models.registry import detect_family, get_lora_target_modules
from llama_vllm.utils.logging import get_logger

logger = get_logger(__name__)


def build_bnb_config(quant: QuantizationArgs) -> Optional[BitsAndBytesConfig]:
    """Build BitsAndBytesConfig from QuantizationArgs."""
    if quant.bits is None:
        return None

    compute_dtype = getattr(torch, quant.compute_dtype)

    if quant.bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.quant_type,
            bnb_4bit_use_double_quant=quant.double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif quant.bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    return None


def load_base_model(
    model_name_or_path: str,
    quantization: Optional[QuantizationArgs] = None,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Union[str, Dict, None] = "auto",
    trust_remote_code: bool = False,
    attn_implementation: str = "flash_attention_2",
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load a causal LM and its tokenizer.

    Args:
        model_name_or_path: HF hub ID or local directory.
        quantization: Quantization settings (4-bit/8-bit via bitsandbytes).
        torch_dtype: Torch dtype; defaults to bfloat16 if CUDA available.
        device_map: HF device_map strategy.
        trust_remote_code: Allow custom code from hub.
        attn_implementation: 'flash_attention_2' | 'eager' | 'sdpa'.

    Returns:
        (model, tokenizer) tuple.
    """
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    bnb_config = build_bnb_config(quantization) if quantization else None

    # Determine attention implementation support
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    # Try flash attention, fall back gracefully
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = attn_implementation
    except ImportError:
        logger.debug("flash-attn not installed; using default attention.")

    logger.info(f"Loading model: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.config.use_cache = False  # Required for gradient checkpointing

    logger.info(f"Loading tokenizer: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        padding_side="right",
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            model.resize_token_embeddings(len(tokenizer))

    logger.info(
        f"Model loaded: {model.config.model_type} | "
        f"Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B"
    )
    return model, tokenizer


def wrap_lora(
    model: PreTrainedModel,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Union[list, str] = "auto",
    bias: str = "none",
    is_quantized: bool = False,
) -> PreTrainedModel:
    """Apply PEFT LoRA (or QLoRA if model is quantized)."""
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if is_quantized:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    if target_modules == "auto":
        model_name = getattr(model.config, "_name_or_path", "")
        target_modules = get_lora_target_modules(model_name)
        logger.info(f"Auto-detected LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_model_for_training(config: FineTuningConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model and optionally apply LoRA/QLoRA for fine-tuning."""
    use_quant = config.method == "qlora" or config.quantization.bits is not None
    quant_args = config.quantization if use_quant else None

    model, tokenizer = load_base_model(
        config.model_name_or_path,
        quantization=quant_args,
    )

    if config.method in ("lora", "qlora"):
        lora = config.lora
        target_mods = lora.target_modules if lora.target_modules != "auto" else "auto"
        model = wrap_lora(
            model,
            lora_rank=lora.lora_rank,
            lora_alpha=lora.lora_alpha,
            lora_dropout=lora.lora_dropout,
            target_modules=target_mods,
            bias=lora.bias,
            is_quantized=(quant_args is not None and quant_args.bits is not None),
        )

    return model, tokenizer

