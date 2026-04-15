"""Unified fine-tuning factory for SFT, LoRA, QLoRA, DPO and RLHF."""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any, Dict

import yaml
from transformers import Trainer, TrainingArguments, default_data_collator

from llama_vllm.config.schemas import FineTuningConfig
from llama_vllm.data.dataset import load_and_preprocess
from llama_vllm.models.loader import load_base_model, load_model_for_training
from llama_vllm.utils.logging import get_logger

logger = get_logger(__name__)


def _build_training_arguments(config: FineTuningConfig) -> TrainingArguments:
    ta = config.training
    return TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=ta.learning_rate,
        num_train_epochs=ta.num_train_epochs,
        per_device_train_batch_size=ta.per_device_train_batch_size,
        per_device_eval_batch_size=ta.per_device_eval_batch_size,
        gradient_accumulation_steps=ta.gradient_accumulation_steps,
        gradient_checkpointing=ta.gradient_checkpointing,
        warmup_ratio=ta.warmup_ratio,
        lr_scheduler_type=ta.lr_scheduler_type,
        weight_decay=ta.weight_decay,
        max_grad_norm=ta.max_grad_norm,
        bf16=ta.bf16,
        fp16=ta.fp16,
        logging_steps=ta.logging_steps,
        evaluation_strategy=ta.eval_strategy,
        eval_steps=ta.eval_steps,
        save_strategy=ta.save_strategy,
        save_steps=ta.save_steps,
        save_total_limit=ta.save_total_limit,
        load_best_model_at_end=ta.load_best_model_at_end,
        metric_for_best_model=ta.metric_for_best_model,
        report_to=ta.report_to,
        dataloader_num_workers=ta.dataloader_num_workers,
        group_by_length=ta.group_by_length,
        seed=ta.seed,
        deepspeed=ta.deepspeed,
        remove_unused_columns=False,
    )


def _maybe_run_llamafactory(config: FineTuningConfig) -> bool:
    """Run LLaMA Factory externally if requested and available."""
    if not config.use_llamafactory:
        return False

    payload: Dict[str, Any] = {
        "model_name_or_path": config.model_name_or_path,
        "stage": config.method,
        "output_dir": config.output_dir,
        **config.llamafactory_args,
    }

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, sort_keys=False, allow_unicode=True)
        temp_path = fp.name

    logger.info(f"Dispatching to LLaMA Factory with temp config: {temp_path}")
    try:
        subprocess.run(["llamafactory-cli", "train", temp_path], check=True)
        return True
    except FileNotFoundError:
        logger.warning("`llamafactory-cli` not found. Falling back to internal TRL pipeline.")
        return False
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def _run_sft_like(config: FineTuningConfig) -> None:
    """Run SFT / LoRA / QLoRA using the standard HF Trainer over tokenized datasets."""

    model, tokenizer = load_model_for_training(config)
    train_ds, eval_ds = load_and_preprocess(config.data, tokenizer, mode="sft")
    args = _build_training_arguments(config)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"✓ {config.method.upper()} complete → {config.output_dir}")


def _run_dpo(config: FineTuningConfig) -> None:
    """Run preference optimization with TRL DPOTrainer."""
    from trl import DPOTrainer

    model, tokenizer = load_model_for_training(config)
    ref_model = None
    if config.ref_model_path:
        ref_model, _ = load_base_model(config.ref_model_path)

    train_ds, eval_ds = load_and_preprocess(config.data, tokenizer, mode="dpo")
    args = _build_training_arguments(config)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        beta=config.dpo_beta,
        loss_type=config.dpo_loss_type,
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"✓ DPO complete → {config.output_dir}")


def _run_rlhf(config: FineTuningConfig) -> None:
    """Run PPO-based RLHF when TRL PPOTrainer is available."""
    from datasets import Dataset
    from transformers import pipeline

    try:
        from trl import PPOConfig, PPOTrainer
    except ImportError as exc:
        raise ImportError("RLHF requires TRL PPO support. Please upgrade `trl`.") from exc

    if not config.reward_model_path:
        raise ValueError("RLHF requires `reward_model_path` in the config.")

    model, tokenizer = load_model_for_training(config)
    train_ds, _ = load_and_preprocess(config.data, tokenizer, mode="sft")

    queries = []
    for sample in train_ds:
        token_ids = sample.get("input_ids", [])
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        queries.append({"query": text})
    query_dataset = Dataset.from_list(queries)

    ppo_config = PPOConfig(
        model_name=config.model_name_or_path,
        learning_rate=config.training.learning_rate,
        batch_size=config.training.per_device_train_batch_size,
        mini_batch_size=max(1, config.training.per_device_train_batch_size // 2),
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        log_with=None,
    )
    trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer, dataset=query_dataset)
    reward_pipe = pipeline("text-classification", model=config.reward_model_path, tokenizer=config.reward_model_path)

    for _epoch in range(config.ppo_epochs):
        for batch in trainer.dataloader:
            query_tensors = batch["input_ids"]
            response_tensors = trainer.generate(query_tensors)
            responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            rewards_raw = reward_pipe(responses, truncation=True)
            rewards = [float(item[0]["score"] if isinstance(item, list) else item["score"]) for item in rewards_raw]
            trainer.step(query_tensors, response_tensors, rewards)

    trainer.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"✓ RLHF complete → {config.output_dir}")


def run_finetuning(config: FineTuningConfig) -> None:
    """Main entry for model fine-tuning."""
    os.makedirs(config.output_dir, exist_ok=True)
    logger.info("=" * 60)
    logger.info("  Fine-tuning")
    logger.info(f"  Model  : {config.model_name_or_path}")
    logger.info(f"  Method : {config.method}")
    logger.info("=" * 60)

    if _maybe_run_llamafactory(config):
        logger.info(f"✓ LLaMA Factory run complete → {config.output_dir}")
        return

    if config.method in {"sft", "lora", "qlora"}:
        _run_sft_like(config)
    elif config.method == "dpo":
        _run_dpo(config)
    elif config.method == "rlhf":
        _run_rlhf(config)
    else:
        raise ValueError(f"Unsupported fine-tuning method: {config.method}")

