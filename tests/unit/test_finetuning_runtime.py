from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llama_vllm.finetuning.runtime import build_dpo_trainer_kwargs, resolve_resume_checkpoint


class FakeDPOTrainerLegacy:
    def __init__(self, model=None, ref_model=None, args=None, tokenizer=None, train_dataset=None, eval_dataset=None, beta=None, loss_type=None):
        pass


class FakeDPOTrainerNew:
    def __init__(self, model=None, ref_model=None, args=None, processing_class=None, train_dataset=None, eval_dataset=None):
        pass


def test_resolve_resume_checkpoint_prefers_requested_path(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir()
    resolved = resolve_resume_checkpoint(str(tmp_path), requested_checkpoint=str(ckpt), auto_resume_from_last_checkpoint=True)
    assert resolved == str(ckpt)


def test_resolve_resume_checkpoint_uses_latest_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-1").mkdir()
    (tmp_path / "checkpoint-7").mkdir()
    resolved = resolve_resume_checkpoint(str(tmp_path), requested_checkpoint=None, auto_resume_from_last_checkpoint=True)
    assert resolved.endswith("checkpoint-7")


def test_build_dpo_kwargs_legacy_signature() -> None:
    kwargs = build_dpo_trainer_kwargs(
        FakeDPOTrainerLegacy,
        model="m",
        ref_model="r",
        args="a",
        tokenizer="tok",
        train_dataset="train",
        eval_dataset="eval",
        beta=0.1,
        loss_type="sigmoid",
    )
    assert kwargs["tokenizer"] == "tok"
    assert kwargs["beta"] == 0.1
    assert kwargs["loss_type"] == "sigmoid"


def test_build_dpo_kwargs_new_signature() -> None:
    kwargs = build_dpo_trainer_kwargs(
        FakeDPOTrainerNew,
        model="m",
        ref_model="r",
        args="a",
        tokenizer="tok",
        train_dataset="train",
        eval_dataset=None,
        beta=0.1,
        loss_type="sigmoid",
    )
    assert kwargs["processing_class"] == "tok"
    assert "tokenizer" not in kwargs
    assert "beta" not in kwargs
    assert "loss_type" not in kwargs

