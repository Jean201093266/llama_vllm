"""Logit-based knowledge distillation loss (KL divergence with temperature)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from llama_vllm.utils.logging import get_logger

logger = get_logger(__name__)


class LogitDistillationLoss(nn.Module):
    """
    KL-divergence distillation loss between teacher and student logits.

    Loss = KL(softmax(T_logits / τ) || softmax(S_logits / τ)) * τ²

    Args:
        temperature: Softening temperature τ. Higher = softer distributions.
        reduction: 'batchmean' | 'mean' | 'sum'
    """

    def __init__(self, temperature: float = 4.0, reduction: str = "batchmean") -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: [batch, seq_len, vocab_size]
            teacher_logits: [batch, seq_len, vocab_size] (same device)
            attention_mask: [batch, seq_len] optional mask for padding

        Returns:
            Scalar distillation loss.
        """
        T = self.temperature

        # Move teacher to same device as student
        teacher_logits = teacher_logits.to(student_logits.device)

        # Compute soft distributions
        student_soft = F.log_softmax(student_logits / T, dim=-1)  # log for KLDivLoss
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)

        # Flatten seq dimension
        batch, seq_len, vocab = student_logits.shape
        student_flat = student_soft.view(-1, vocab)  # [B*S, V]
        teacher_flat = teacher_soft.view(-1, vocab)  # [B*S, V]

        loss = F.kl_div(student_flat, teacher_flat, reduction="none").sum(dim=-1)  # [B*S]

        if attention_mask is not None:
            mask_flat = attention_mask.view(-1).float()
            loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        else:
            if self.reduction == "batchmean":
                loss = loss.sum() / batch
            elif self.reduction == "mean":
                loss = loss.mean()
            else:
                loss = loss.sum()

        # Multiply by T² to keep loss scale consistent with label loss
        return loss * (T ** 2)


class CombinedDistillationLoss(nn.Module):
    """
    Combined task loss + distillation loss:
    L = α * L_distill + (1 - α) * L_task
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7) -> None:
        super().__init__()
        self.alpha = alpha
        self.distill_loss = LogitDistillationLoss(temperature=temperature)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            (total_loss, metrics_dict)
        """
        # Distillation loss
        l_distill = self.distill_loss(student_logits, teacher_logits, attention_mask)

        # Task loss (cross-entropy with ground-truth labels)
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        l_task = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        total = self.alpha * l_distill + (1.0 - self.alpha) * l_task

        metrics = {
            "loss_distill": l_distill.item(),
            "loss_task": l_task.item(),
            "loss_total": total.item(),
        }
        return total, metrics

