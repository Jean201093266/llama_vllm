"""Batch inference utilities over JSONL/CSV inputs."""

from __future__ import annotations

import json
import os
from typing import Dict, List

import pandas as pd

from llama_vllm.config.schemas import InferenceConfig
from llama_vllm.inference.engine import VLLMEngineWrapper
from llama_vllm.utils.logging import get_logger

logger = get_logger(__name__)


def _load_prompts(path: str, prompt_key: str) -> List[Dict[str, str]]:
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    if ext == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    raise ValueError(f"Unsupported input format: {ext}")


def run_batch_inference(config: InferenceConfig) -> str:
    if not config.input_file:
        raise ValueError("Batch inference requires `input_file` in config or CLI.")

    records = _load_prompts(config.input_file, config.prompt_key)
    prompts = [str(item.get(config.prompt_key, "")) for item in records]
    engine = VLLMEngineWrapper(config)
    outputs = engine.generate(prompts)

    output_path = config.output_file or "./outputs/inference/results.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record, output in zip(records, outputs):
            text = output.outputs[0].text if output.outputs else ""
            payload = {**record, "response": text}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    logger.info(f"✓ Batch inference complete → {output_path}")
    return output_path

