from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llama_vllm.config.schemas import load_config


def test_load_finetuning_config() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "configs" / "finetuning" / "lora.yaml", config_type="finetuning")
    assert cfg.method == "lora"
    assert cfg.lora.lora_rank == 16
    assert cfg.training.per_device_train_batch_size == 2


def test_load_inference_config() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "configs" / "inference" / "server.yaml", config_type="inference")
    assert cfg.mode == "server"
    assert cfg.server.port == 8000

