"""Dataset loading and preprocessing for all training modes."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from llama_vllm.config.schemas import DataArgs
from llama_vllm.utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Format converters
# ─────────────────────────────────────────────

def _convert_alpaca(example: Dict, system_prompt: Optional[str] = None) -> Dict:
    """Convert Alpaca-format to standard {'input': ..., 'output': ...}."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")
    if inp:
        text_in = f"{instruction}\n\n{inp}"
    else:
        text_in = instruction
    if system_prompt:
        text_in = f"{system_prompt}\n\n{text_in}"
    return {"input": text_in, "output": output}


def _convert_sharegpt(example: Dict, system_prompt: Optional[str] = None) -> Dict:
    """Convert ShareGPT conversations to concatenated input/output."""
    conversations: List[Dict] = example.get("conversations", [])
    turns: List[str] = []
    for turn in conversations:
        role = turn.get("from", turn.get("role", "human"))
        value = turn.get("value", turn.get("content", ""))
        if role in ("human", "user"):
            turns.append(f"User: {value}")
        elif role in ("gpt", "assistant"):
            turns.append(f"Assistant: {value}")
    full_text = "\n".join(turns)
    if system_prompt:
        full_text = f"System: {system_prompt}\n\n{full_text}"
    # Last assistant turn is the "output"; everything else is "input"
    last_assistant = ""
    for turn in reversed(conversations):
        role = turn.get("from", turn.get("role", ""))
        if role in ("gpt", "assistant"):
            last_assistant = turn.get("value", turn.get("content", ""))
            break
    # Input = all except last assistant turn
    input_text = full_text[: full_text.rfind(last_assistant)].rstrip() if last_assistant else full_text
    return {"input": input_text, "output": last_assistant}


def _convert_openai_messages(example: Dict, system_prompt: Optional[str] = None) -> Dict:
    """Convert OpenAI messages format."""
    messages: List[Dict] = example.get("messages", [])
    turns: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        turns.append(f"{role.capitalize()}: {content}")
    full = "\n".join(turns)
    if system_prompt:
        full = f"System: {system_prompt}\n\n{full}"
    # Similar split
    last_assistant = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            last_assistant = msg.get("content", "")
            break
    input_text = full[: full.rfind(last_assistant)].rstrip() if last_assistant else full
    return {"input": input_text, "output": last_assistant}


_CONVERTERS = {
    "alpaca": _convert_alpaca,
    "sharegpt": _convert_sharegpt,
    "openai": _convert_openai_messages,
    "raw": lambda ex, sp=None: ex,
    "dpo_pairs": lambda ex, sp=None: ex,  # Pass through; handled separately
}


# ─────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────

def load_and_preprocess(
    cfg: DataArgs,
    tokenizer: PreTrainedTokenizerBase,
    mode: str = "sft",  # "sft" | "dpo"
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load dataset and apply tokenization.

    Returns:
        (train_dataset, eval_dataset) tuple; eval may be None.
    """
    # Load raw dataset
    if os.path.exists(cfg.dataset_name_or_path):
        # Local file: infer format from extension
        ext = os.path.splitext(cfg.dataset_name_or_path)[-1].lower()
        fmt_map = {".json": "json", ".jsonl": "json", ".csv": "csv", ".parquet": "parquet"}
        file_fmt = fmt_map.get(ext, "json")
        raw = load_dataset(file_fmt, data_files={"train": cfg.dataset_name_or_path}, cache_dir=cfg.cache_dir)
    else:
        raw = load_dataset(cfg.dataset_name_or_path, cache_dir=cfg.cache_dir)

    # Select splits
    if isinstance(raw, DatasetDict):
        available_splits = list(raw.keys())
        train_split = cfg.train_split if cfg.train_split in raw else available_splits[0]
        if train_split != cfg.train_split:
            logger.warning(f"Requested train split '{cfg.train_split}' not found. Using '{train_split}' instead.")
        train_raw = raw[train_split]
        eval_raw = raw.get(cfg.eval_split) if cfg.eval_split and cfg.eval_split in raw else None
    else:
        train_raw = raw
        eval_raw = None

    if eval_raw is None and cfg.validation_split_ratio > 0:
        split = train_raw.train_test_split(
            test_size=cfg.validation_split_ratio,
            shuffle=cfg.shuffle_before_split,
            seed=42,
        )
        train_raw = split["train"]
        eval_raw = split["test"]
        logger.info(
            f"Validation split '{cfg.eval_split}' not found. Auto-split train set with ratio={cfg.validation_split_ratio}."
        )

    # Subsample
    if cfg.max_samples and cfg.max_samples < len(train_raw):
        train_raw = train_raw.select(range(cfg.max_samples))

    # Format conversion
    converter = _CONVERTERS.get(cfg.dataset_format, _CONVERTERS["alpaca"])

    if cfg.dataset_format != "dpo_pairs":
        # SFT / distillation
        train_raw = train_raw.map(
            lambda ex: converter(ex, cfg.system_prompt),
            num_proc=cfg.preprocessing_num_workers,
            desc="Converting format",
        )
        if eval_raw is not None:
            eval_raw = eval_raw.map(
                lambda ex: converter(ex, cfg.system_prompt),
                num_proc=cfg.preprocessing_num_workers,
                desc="Converting eval format",
            )

    # DPO typically expects raw text columns and tokenizes internally inside TRL.
    if mode == "dpo":
        train_ds = _prepare_dpo_dataset(train_raw, cfg)
        eval_ds = _prepare_dpo_dataset(eval_raw, cfg) if eval_raw is not None else None
        logger.info(f"Train samples: {len(train_ds)}" + (f" | Eval samples: {len(eval_ds)}" if eval_ds else ""))
        return train_ds, eval_ds

    # Tokenize
    if mode == "sft":
        tokenize_fn = _make_sft_tokenize_fn(cfg, tokenizer)
    else:
        tokenize_fn = _make_sft_tokenize_fn(cfg, tokenizer)

    remove_cols = list(train_raw.column_names)
    train_ds = train_raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=remove_cols,
        num_proc=cfg.preprocessing_num_workers,
        desc="Tokenizing train",
    )
    eval_ds = None
    if eval_raw is not None:
        remove_cols_eval = list(eval_raw.column_names)
        eval_ds = eval_raw.map(
            tokenize_fn,
            batched=True,
            remove_columns=remove_cols_eval,
            num_proc=cfg.preprocessing_num_workers,
            desc="Tokenizing eval",
        )

    logger.info(f"Train samples: {len(train_ds)}" + (f" | Eval samples: {len(eval_ds)}" if eval_ds else ""))
    return train_ds, eval_ds


def _make_sft_tokenize_fn(cfg: DataArgs, tokenizer: PreTrainedTokenizerBase):
    """Create a batched SFT tokenization function."""
    max_len = cfg.max_seq_length
    in_key = cfg.input_key
    out_key = cfg.output_key
    train_on_prompt = cfg.train_on_prompt

    def tokenize(examples):
        inputs = examples.get(in_key, [""] * len(examples[list(examples.keys())[0]]))
        outputs = examples.get(out_key, [""] * len(inputs))
        prompts = [f"{inp}\n\n### Response:\n" for inp in inputs]
        texts = [f"{prompt}{out}" for prompt, out in zip(prompts, outputs)]
        batch = tokenizer(texts, truncation=True, max_length=max_len, padding=False)
        labels = []
        prompt_lengths = None

        if not train_on_prompt:
            prompt_encodings = tokenizer(prompts, truncation=True, max_length=max_len, padding=False, add_special_tokens=True)
            prompt_lengths = [len(ids) for ids in prompt_encodings["input_ids"]]

        for idx, input_ids in enumerate(batch["input_ids"]):
            sample_labels = list(input_ids)
            if not train_on_prompt and prompt_lengths is not None:
                prompt_len = min(prompt_lengths[idx], len(sample_labels))
                sample_labels[:prompt_len] = [-100] * prompt_len
            labels.append(sample_labels)

        batch["labels"] = labels
        return batch

    return tokenize


def _prepare_dpo_dataset(dataset: Dataset, cfg: DataArgs) -> Dataset:
    """Normalize preference datasets to TRL-friendly prompt/chosen/rejected text fields."""
    prompt_key = cfg.input_key
    chosen_key = cfg.chosen_key
    rejected_key = cfg.rejected_key

    return dataset.map(
        lambda ex: {
            "prompt": ex.get(prompt_key, ex.get("prompt", "")),
            "chosen": ex.get(chosen_key, ex.get("chosen", "")),
            "rejected": ex.get(rejected_key, ex.get("rejected", "")),
        },
        remove_columns=list(dataset.column_names),
        desc="Preparing DPO text pairs",
    )

