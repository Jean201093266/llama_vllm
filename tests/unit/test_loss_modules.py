from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

torch = pytest.importorskip("torch")

from llama_vllm.distillation.feature_distill import FeatureDistillationLoss
from llama_vllm.distillation.logit_distill import LogitDistillationLoss


def test_logit_distillation_loss_scalar() -> None:
    loss_fn = LogitDistillationLoss(temperature=2.0)
    student = torch.randn(2, 4, 8)
    teacher = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4)
    loss = loss_fn(student, teacher, mask)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_feature_distillation_loss_scalar() -> None:
    loss_fn = FeatureDistillationLoss(
        student_hidden_size=6,
        teacher_hidden_size=6,
        layers=[1, 2],
        loss_type="mse",
        project_hidden=False,
    )
    student_hidden = {1: torch.randn(2, 3, 6), 2: torch.randn(2, 3, 6)}
    teacher_hidden = {1: torch.randn(2, 3, 6), 2: torch.randn(2, 3, 6)}
    mask = torch.ones(2, 3)
    loss = loss_fn(student_hidden, teacher_hidden, mask)
    assert loss.dim() == 0
    assert torch.isfinite(loss)

