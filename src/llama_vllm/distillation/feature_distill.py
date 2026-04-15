"""Feature/layer-wise distillation loss (MSE or cosine similarity)."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llama_vllm.utils.logging import get_logger

logger = get_logger(__name__)


class LayerProjection(nn.Module):
    """Optional linear projection to align hidden dimensions."""

    def __init__(self, student_dim: int, teacher_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)
        if student_dim == teacher_dim:
            nn.init.eye_(self.proj.weight)
        else:
            nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FeatureDistillationLoss(nn.Module):
    """
    Layer-wise feature distillation between teacher and student hidden states.

    Supports:
      - MSE loss between hidden state tensors
      - Cosine similarity loss (1 - cos_sim)
      - Optional linear projection when teacher_dim != student_dim
    """

    def __init__(
        self,
        student_hidden_size: int,
        teacher_hidden_size: int,
        layers: List[int],
        loss_type: str = "mse",
        project_hidden: bool = False,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.loss_type = loss_type
        self.project_hidden = project_hidden

        self.projections = nn.ModuleDict()
        if project_hidden and student_hidden_size != teacher_hidden_size:
            for layer in layers:
                self.projections[str(layer)] = LayerProjection(
                    student_hidden_size, teacher_hidden_size
                )
            logger.info(
                f"Feature distillation: added {len(layers)} projection layers "
                f"({student_hidden_size} → {teacher_hidden_size})"
            )

    def forward(
        self,
        student_hidden: Dict[int, torch.Tensor],
        teacher_hidden: Dict[int, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            student_hidden: Dict[layer_idx → tensor [B, S, H_student]]
            teacher_hidden: Dict[layer_idx → tensor [B, S, H_teacher]]
            attention_mask: [B, S] optional padding mask

        Returns:
            Averaged layer-wise feature distillation loss.
        """
        losses = []
        for layer in self.layers:
            if layer not in student_hidden or layer not in teacher_hidden:
                logger.debug(f"Layer {layer} not found in hidden states; skipping.")
                continue

            s_feat = student_hidden[layer]  # [B, S, H_s]
            t_feat = teacher_hidden[layer].to(s_feat.device).detach()  # [B, S, H_t]

            # Optional projection
            if str(layer) in self.projections:
                s_feat = self.projections[str(layer)](s_feat)

            if self.loss_type == "mse":
                layer_loss = self._mse_loss(s_feat, t_feat, attention_mask)
            elif self.loss_type == "cosine":
                layer_loss = self._cosine_loss(s_feat, t_feat, attention_mask)
            else:
                raise ValueError(f"Unknown feature loss type: {self.loss_type}")

            losses.append(layer_loss)

        if not losses:
            ref = next(iter(student_hidden.values()), None)
            device = ref.device if ref is not None else None
            return torch.tensor(0.0, requires_grad=True, device=device)
        return torch.stack(losses).mean()

    def _mse_loss(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MSE loss with optional masking."""
        loss = F.mse_loss(s, t, reduction="none").mean(dim=-1)  # [B, S]
        if mask is not None:
            mask_f = mask.float()
            return (loss * mask_f).sum() / (mask_f.sum() + 1e-8)
        return loss.mean()

    def _cosine_loss(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """1 - cosine_similarity loss with optional masking."""
        cos = F.cosine_similarity(s, t, dim=-1)  # [B, S]
        loss = 1.0 - cos
        if mask is not None:
            mask_f = mask.float()
            return (loss * mask_f).sum() / (mask_f.sum() + 1e-8)
        return loss.mean()

