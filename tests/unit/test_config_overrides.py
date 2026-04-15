from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llama_vllm.config.schemas import load_config


def test_override_nested_config_values() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(
        root / "configs" / "finetuning" / "lora.yaml",
        config_type="finetuning",
        overrides=[
            "training.learning_rate=0.0001",
            "output_dir=./outputs/test_run",
            "lora.lora_rank=8",
        ],
    )
    assert cfg.training.learning_rate == 0.0001
    assert cfg.output_dir == "./outputs/test_run"
    assert cfg.lora.lora_rank == 8

