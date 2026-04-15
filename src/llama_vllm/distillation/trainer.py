"""Distillation trainer: HuggingFace Trainer subclass with combined losses."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers import (
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from llama_vllm.config.schemas import DistillationConfig
from llama_vllm.data.dataset import load_and_preprocess
from llama_vllm.distillation.feature_distill import FeatureDistillationLoss
from llama_vllm.distillation.logit_distill import CombinedDistillationLoss
from llama_vllm.distillation.teacher import build_teacher
from llama_vllm.models.loader import wrap_lora
from llama_vllm.utils.logging import get_logger

logger = get_logger(__name__)


class DistillationTrainer(Trainer):
    """
    Custom HF Trainer that injects teacher logits and computes
    combined distillation + task losses.
    """

    def __init__(
        self,
        *args,
        teacher: Any,
        distill_config: DistillationConfig,
        feature_loss: Optional[FeatureDistillationLoss] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.distill_config = distill_config
        self.feature_loss = feature_loss

        self.logit_loss_fn = CombinedDistillationLoss(
            temperature=distill_config.temperature,
            alpha=distill_config.alpha,
        )

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels", input_ids)

        # ─── Student forward ───────────────────────────────────────
        distill_type = self.distill_config.distill_type
        student_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=(distill_type in ("feature", "combined")),
            use_cache=False,
        )

        # ─── Teacher logits ────────────────────────────────────────
        with torch.no_grad():
            teacher_logits = self.teacher.get_logits(input_ids, attention_mask)

        # ─── Logit distillation loss ───────────────────────────────
        total_loss, metrics = self.logit_loss_fn(
            student_out.logits,
            teacher_logits,
            labels,
            attention_mask,
        )

        # ─── Feature distillation loss ─────────────────────────────
        if self.feature_loss is not None and distill_type in ("feature", "combined"):
            teacher_hidden = self.teacher.get_hidden_states(input_ids, attention_mask, layers=self.distill_config.feature_layers)
            student_hidden = {
                i: h for i, h in enumerate(student_out.hidden_states or [])
                if i in self.distill_config.feature_layers
            }
            feat_loss = self.feature_loss(student_hidden, teacher_hidden, attention_mask)
            total_loss = total_loss + feat_loss
            metrics["loss_feature"] = feat_loss.item()

        # Log every N steps
        if self.state.global_step % self.args.logging_steps == 0:
            for k, v in metrics.items():
                self.log({k: v})

        return (total_loss, student_out) if return_outputs else total_loss


def run_distillation(config: DistillationConfig) -> None:
    """Entry point: set up teacher, student, data, and run distillation training."""
    logger.info("=" * 60)
    logger.info("  Knowledge Distillation")
    logger.info(f"  Teacher : {config.teacher_model}")
    logger.info(f"  Student : {config.student_model}")
    logger.info(f"  Type    : {config.distill_type}")
    logger.info("=" * 60)

    os.makedirs(config.output_dir, exist_ok=True)

    # ─── Build teacher ─────────────────────────────────────────────
    feature_distill = config.distill_type in ("feature", "combined")
    teacher = build_teacher(
        config.teacher_model,
        use_vllm=config.use_vllm_teacher and not feature_distill,
        tensor_parallel_size=config.teacher_tensor_parallel_size,
        dtype=config.teacher_dtype,
        feature_distill=feature_distill,
    )

    # ─── Build student ─────────────────────────────────────────────
    from llama_vllm.models.loader import load_base_model

    student_model, tokenizer = load_base_model(
        config.student_model,
        quantization=config.quantization if config.use_lora_student else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if config.use_lora_student:
        lora = config.lora
        student_model = wrap_lora(
            student_model,
            lora_rank=lora.lora_rank,
            lora_alpha=lora.lora_alpha,
            lora_dropout=lora.lora_dropout,
            target_modules=lora.target_modules,
            is_quantized=config.quantization.bits is not None,
        )

    # ─── Build feature loss if needed ─────────────────────────────
    feature_loss_module = None
    if feature_distill and config.feature_layers:
        teacher_hidden = getattr(getattr(teacher, "model", None), "config", None)
        teacher_hidden_size = getattr(teacher_hidden, "hidden_size", None)
        student_hidden_size = getattr(student_model.config, "hidden_size", None)
        if teacher_hidden_size and student_hidden_size:
            feature_loss_module = FeatureDistillationLoss(
                student_hidden_size=student_hidden_size,
                teacher_hidden_size=teacher_hidden_size,
                layers=config.feature_layers,
                loss_type=config.feature_loss_type,
                project_hidden=config.project_hidden,
            )
        else:
            logger.warning("Could not infer hidden sizes for feature distillation; feature loss disabled.")

    # ─── Load data ─────────────────────────────────────────────────
    train_ds, eval_ds = load_and_preprocess(config.data, tokenizer, mode="sft")

    # ─── Build TrainingArguments ───────────────────────────────────
    ta = config.training
    training_args = TrainingArguments(
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

    # ─── Trainer ───────────────────────────────────────────────────
    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        teacher=teacher,
        distill_config=config,
        feature_loss=feature_loss_module,
    )

    resume = config.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"✓ Distillation complete → {config.output_dir}")

