"""Distillation package with lazy exports."""

__all__ = [
    "HFTeacher",
    "VLLMTeacher",
    "build_teacher",
    "LogitDistillationLoss",
    "CombinedDistillationLoss",
    "FeatureDistillationLoss",
    "DistillationTrainer",
    "run_distillation",
]


def __getattr__(name: str):
    if name in {"LogitDistillationLoss", "CombinedDistillationLoss"}:
        from llama_vllm.distillation.logit_distill import CombinedDistillationLoss, LogitDistillationLoss

        return {"LogitDistillationLoss": LogitDistillationLoss, "CombinedDistillationLoss": CombinedDistillationLoss}[name]
    if name == "FeatureDistillationLoss":
        from llama_vllm.distillation.feature_distill import FeatureDistillationLoss

        return FeatureDistillationLoss
    if name in {"HFTeacher", "VLLMTeacher", "build_teacher"}:
        from llama_vllm.distillation.teacher import HFTeacher, VLLMTeacher, build_teacher

        return {"HFTeacher": HFTeacher, "VLLMTeacher": VLLMTeacher, "build_teacher": build_teacher}[name]
    if name in {"DistillationTrainer", "run_distillation"}:
        from llama_vllm.distillation.trainer import DistillationTrainer, run_distillation

        return {"DistillationTrainer": DistillationTrainer, "run_distillation": run_distillation}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

