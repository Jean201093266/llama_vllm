from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llama_vllm.config.schemas import load_config
from llama_vllm.inference.batch import run_batch_inference


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs" / "inference" / "batch.yaml"
    cfg = load_config(config_path, config_type="inference")
    run_batch_inference(cfg)

