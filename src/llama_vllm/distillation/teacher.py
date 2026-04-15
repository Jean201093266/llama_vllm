"""Teacher model wrapper supporting vLLM and HuggingFace backends."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llama_vllm.utils.logging import get_logger

logger = get_logger(__name__)


class HFTeacher:
    """
    HuggingFace-based teacher that provides:
      - Full logits over the vocabulary
      - Hidden states from specified layers (for feature distillation)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return raw logits [batch, seq_len, vocab_size]."""
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
            use_cache=False,
        )
        return outputs.logits

    @torch.no_grad()
    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layers: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Return hidden states from specified transformer layers.

        Returns:
            Dict mapping layer index → hidden state tensor [batch, seq_len, hidden_size]
        """
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
            output_hidden_states=True,
            use_cache=False,
        )
        all_hidden = outputs.hidden_states  # tuple of [batch, seq, hidden]
        if layers is None:
            layers = list(range(len(all_hidden)))
        return {layer: all_hidden[layer].detach() for layer in layers if layer < len(all_hidden)}


class VLLMTeacher:
    """
    vLLM-based teacher for fast logit generation.
    Note: Hidden states are NOT available from vLLM; use HFTeacher for feature distillation.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizerBase,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.45,  # Leave room for student
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is required for VLLMTeacher. Install with: pip install vllm"
            )

        logger.info(f"Initializing vLLM teacher: {model_name_or_path}")
        self.llm = LLM(
            model=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.tokenizer = tokenizer
        self._model_name = model_name_or_path

    def get_logits_from_prompts(
        self, prompts: List[str], prompt_logprobs: int = 0
    ) -> List[Any]:
        """
        Use vLLM prompt_logprobs to get per-token log probs.
        Returns list of vLLM RequestOutput objects.
        """
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=1.0,
            max_tokens=1,
            prompt_logprobs=prompt_logprobs,
        )
        return self.llm.generate(prompts, params)

    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode input_ids to text, run vLLM with prompt_logprobs,
        and reconstruct a logit tensor [batch, seq_len, vocab_size].

        Note: This is an approximation — vLLM returns log-probs for a
        limited top-k, not full vocab. For exact KL distillation use HFTeacher.
        """
        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        vocab_size = len(self.tokenizer)
        batch_size, seq_len = input_ids.shape

        outputs = self.get_logits_from_prompts(prompts, prompt_logprobs=vocab_size)

        # Reconstruct dense logit tensors (approximation from log probs)
        logits = torch.full((batch_size, seq_len, vocab_size), -1e9)
        for b, out in enumerate(outputs):
            if out.prompt_logprobs:
                for t, token_logprobs in enumerate(out.prompt_logprobs):
                    if token_logprobs and t < seq_len:
                        for token_id, logprob in token_logprobs.items():
                            if isinstance(logprob, (int, float)):
                                logits[b, t, int(token_id)] = float(logprob)
                            else:
                                logits[b, t, int(token_id)] = logprob.logprob
        return logits

    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layers: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        raise NotImplementedError("vLLM teacher does not expose hidden states. Use HF teacher.")


def build_teacher(
    model_name_or_path: str,
    use_vllm: bool = True,
    tensor_parallel_size: int = 1,
    dtype: str = "bfloat16",
    feature_distill: bool = False,
) -> Any:
    """
    Factory: build the appropriate teacher backend.
    Falls back to HF if use_vllm=True but feature_distill=True.
    """
    if feature_distill:
        logger.info("Feature distillation requires HF teacher (hidden states).")
        use_vllm = False

    if use_vllm:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right")
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            return VLLMTeacher(
                model_name_or_path,
                tokenizer=tokenizer,
                tensor_parallel_size=tensor_parallel_size,
                dtype=dtype,
            )
        except ImportError:
            logger.warning("vLLM not available; falling back to HF teacher.")

    # HF teacher
    from llama_vllm.models.loader import load_base_model
    dtype_map = {
        "auto": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model, tokenizer = load_base_model(model_name_or_path, torch_dtype=dtype_map.get(dtype, torch.bfloat16), device_map="auto")
    return HFTeacher(model, tokenizer)

