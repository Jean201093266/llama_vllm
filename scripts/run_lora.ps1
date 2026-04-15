$ErrorActionPreference = 'Stop'

python -m pip install -e .[train]
llama-vllm finetune run --config configs/finetuning/lora.yaml

