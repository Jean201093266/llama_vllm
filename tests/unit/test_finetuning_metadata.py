from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llama_vllm.config.schemas import load_config
from llama_vllm.finetuning.metadata import build_run_metadata, write_run_metadata


def test_build_and_write_run_metadata(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "configs" / "finetuning" / "lora.yaml", config_type="finetuning")
    payload = build_run_metadata(stage="lora", config=cfg, resume_from_checkpoint="checkpoint-1", status="started")
    path = write_run_metadata(str(tmp_path), "run_start.json", payload)
    assert Path(path).exists()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["stage"] == "lora"
    assert data["resume_from_checkpoint"] == "checkpoint-1"
    assert "config" in data

