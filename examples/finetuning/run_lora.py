from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llama_vllm.config.schemas import load_config
from llama_vllm.finetuning import run_finetuning


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs" / "finetuning" / "lora.yaml"
    cfg = load_config(config_path, config_type="finetuning")
    run_finetuning(cfg)

