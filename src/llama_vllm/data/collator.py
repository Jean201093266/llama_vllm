"""Data collators for causal LM and preference tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from transformers import PreTrainedTokenizerBase


@dataclass
class CausalLMDataCollator:
    """Pad tokenized causal LM samples and keep labels padded with -100."""

    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        max_length = batch["input_ids"].shape[1]

        padded_labels = []
        for label in labels:
            pad_len = max_length - len(label)
            padded_labels.append(label + [self.label_pad_token_id] * pad_len)
        batch["labels"] = batch["input_ids"].new_tensor(padded_labels)
        return batch

