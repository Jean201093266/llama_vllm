"""Model family registry: LoRA targets, chat templates, special tokens."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelFamilyInfo:
    """Metadata for a model family."""
    family: str
    lora_target_modules: List[str]
    chat_template: Optional[str] = None
    model_type: str = "causal_lm"
    rope_scaling: Optional[Dict] = None
    sliding_window: Optional[int] = None
    max_position_embeddings: int = 4096
    notes: str = ""


# ─────────────────────────────────────────────
# Family registry
# ─────────────────────────────────────────────

_REGISTRY: Dict[str, ModelFamilyInfo] = {
    "llama": ModelFamilyInfo(
        family="llama",
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        chat_template="llama-2",
        max_position_embeddings=4096,
        notes="LLaMA / LLaMA-2 / LLaMA-3 series",
    ),
    "mistral": ModelFamilyInfo(
        family="mistral",
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        chat_template="mistral",
        sliding_window=4096,
        max_position_embeddings=32768,
        notes="Mistral / Mixtral series",
    ),
    "qwen": ModelFamilyInfo(
        family="qwen",
        lora_target_modules=["c_attn", "c_proj", "w1", "w2"],
        chat_template="chatml",
        max_position_embeddings=32768,
        notes="Qwen / Qwen1.5 / Qwen2 series",
    ),
    "qwen2": ModelFamilyInfo(
        family="qwen2",
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        chat_template="chatml",
        max_position_embeddings=32768,
        notes="Qwen2 / Qwen2.5 series",
    ),
    "gemma": ModelFamilyInfo(
        family="gemma",
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        chat_template="gemma",
        max_position_embeddings=8192,
        notes="Google Gemma / Gemma 2 series",
    ),
    "phi": ModelFamilyInfo(
        family="phi",
        lora_target_modules=["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
        chat_template="phi",
        max_position_embeddings=2048,
        notes="Microsoft Phi-2 / Phi-3 series",
    ),
    "baichuan": ModelFamilyInfo(
        family="baichuan",
        lora_target_modules=["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"],
        chat_template="baichuan",
        max_position_embeddings=4096,
        notes="Baichuan / Baichuan2 series",
    ),
    "chatglm": ModelFamilyInfo(
        family="chatglm",
        lora_target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        chat_template="chatglm",
        max_position_embeddings=32768,
        notes="ChatGLM / GLM-4 series",
    ),
    "yi": ModelFamilyInfo(
        family="yi",
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        chat_template="chatml",
        max_position_embeddings=200000,
        notes="Yi / Yi-1.5 series",
    ),
    "deepseek": ModelFamilyInfo(
        family="deepseek",
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        chat_template="deepseek",
        max_position_embeddings=4096,
        notes="DeepSeek / DeepSeek-V2 / DeepSeek-R1 series",
    ),
    "internlm": ModelFamilyInfo(
        family="internlm",
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        chat_template="internlm",
        max_position_embeddings=32768,
        notes="InternLM / InternLM2 series",
    ),
    # Fallback for unknown architectures
    "default": ModelFamilyInfo(
        family="default",
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        notes="Generic fallback — targets attention projections only",
    ),
}

# Model name → family name pattern matching
_NAME_PATTERNS: List[tuple[str, str]] = [
    ("llama", "llama"),
    ("mistral", "mistral"),
    ("mixtral", "mistral"),
    ("qwen2", "qwen2"),
    ("qwen", "qwen"),
    ("gemma", "gemma"),
    ("phi", "phi"),
    ("baichuan", "baichuan"),
    ("chatglm", "chatglm"),
    ("glm", "chatglm"),
    ("yi-", "yi"),
    ("yi_", "yi"),
    ("deepseek", "deepseek"),
    ("internlm", "internlm"),
]


def detect_family(model_name_or_path: str) -> str:
    """Auto-detect model family from model name or path."""
    name_lower = model_name_or_path.lower()
    for pattern, family in _NAME_PATTERNS:
        if pattern in name_lower:
            return family
    return "default"


def get_family_info(family_or_model: str) -> ModelFamilyInfo:
    """Get ModelFamilyInfo by family name or model name/path."""
    if family_or_model in _REGISTRY:
        return _REGISTRY[family_or_model]
    # Try to detect family
    family = detect_family(family_or_model)
    return _REGISTRY.get(family, _REGISTRY["default"])


def get_lora_target_modules(model_name_or_path: str) -> List[str]:
    """Get LoRA target modules for a given model."""
    return get_family_info(model_name_or_path).lora_target_modules


def list_families() -> List[str]:
    """Return list of known model families."""
    return [k for k in _REGISTRY.keys() if k != "default"]

