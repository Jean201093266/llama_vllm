# llama-vllm

基于 **vLLM** 与 **LLaMA Factory** 的统一大模型框架，覆盖：

- **模型蒸馏**：Logit Distillation / Feature Distillation / Combined Distillation
- **模型微调**：SFT / LoRA / QLoRA / DPO / RLHF
- **模型推理**：vLLM 批量推理 / 流式推理 / OpenAI 兼容服务

## 项目结构

```text
llama_vllm/
├── configs/
│   ├── distillation/
│   ├── finetuning/
│   └── inference/
├── docker/
├── examples/
├── scripts/
├── src/llama_vllm/
│   ├── cli/
│   ├── config/
│   ├── data/
│   ├── distillation/
│   ├── finetuning/
│   ├── inference/
│   ├── models/
│   └── utils/
└── tests/
```

## 安装

最小安装：

```powershell
cd D:\projects\github\llama_vllm
pip install -e .
```

训练能力：

```powershell
pip install -e .[train]
```

推理服务能力：

```powershell
pip install -e .[inference]
```

完整开发依赖：

```powershell
pip install -r requirements-dev.txt
```

## 快速开始

### 1) LoRA 微调

```powershell
llama-vllm finetune run --config configs/finetuning/lora.yaml
```

### 2) 知识蒸馏

```powershell
llama-vllm distill run --config configs/distillation/logit_distill.yaml
```

### 3) 批量推理

```powershell
llama-vllm infer batch --config configs/inference/batch.yaml --input .\examples\inference\prompts.jsonl --output .\outputs\results.jsonl
```

### 4) 启动服务

```powershell
llama-vllm serve --config configs/inference/server.yaml --port 8000
```

### 5) 启动可视化后台（dry-run）

```powershell
llama-vllm dashboard --host 127.0.0.1 --port 7860 --db-path .dashboard/history.db
```

默认只提供 preflight 与命令预览，不会自动启动训练。
后台请求历史会持久化到 SQLite（默认 `.dashboard/history.db`）。

## 主要命令

```powershell
llama-vllm distill run --config <yaml>
llama-vllm finetune run --config <yaml>
llama-vllm finetune export --base-model <base> --adapter <adapter> --output <merged>
llama-vllm infer batch --config <yaml> --input <jsonl/csv>
llama-vllm infer stream --config <yaml> --prompt "hello"
llama-vllm serve --config <yaml>
llama-vllm dashboard --host 127.0.0.1 --port 7860
```

## 配置说明

- 蒸馏配置：`configs/distillation/*.yaml`
- 微调配置：`configs/finetuning/*.yaml`
- 推理配置：`configs/inference/*.yaml`

支持 `--override key=value` 动态覆盖，例如：

```powershell
llama-vllm finetune run --config configs/finetuning/lora.yaml --override training.learning_rate=1e-4 --override output_dir=./outputs/debug
```

当 preflight 校验失败时，可使用 `--auto-fix` 仅打印可复制修复命令并退出：

```powershell
llama-vllm distill run --config configs/distillation/feature_distill.yaml --auto-fix
llama-vllm finetune run --config configs/finetuning/lora.yaml --auto-fix
```

`--auto-fix` 会输出编号建议，并将第 1 条作为推荐修复命令。
如需同时查看底层 `--override` 列表，可追加 `--show-raw`。
如需自动应用第 1 条建议并仅重跑 preflight（不启动训练），可追加 `--apply-overrides`。

训练任务默认支持以下生产化能力：

- 自动从最新 `checkpoint-*` 恢复（`training.auto_resume_from_last_checkpoint: true`）
- 训练开始/完成时写入 `run_start.json` 与 `run_complete.json`
- 写入 `latest_checkpoint.json` / `best_checkpoint.json` 作为 checkpoint 生命周期标记
- 可选早停（`training.early_stopping_patience`）
- DPO 针对不同 `trl` 版本做参数兼容处理
- preflight 失败时会附带 `Quick override suggestions`（可直接复制 `--override` 参数进行降级修复）

## Docker

```powershell
docker build -f docker/Dockerfile -t llama-vllm:gpu .
docker run --gpus all -p 8000:8000 -v ${PWD}\configs:/app/configs llama-vllm:gpu serve --config /app/configs/inference/server.yaml
```

## 注意事项

1. `vllm` 通常需要 Linux + CUDA 环境；Windows 下更适合做配置开发与代码组织。
2. `bitsandbytes`、`deepspeed`、`vllm` 对 CUDA / 驱动版本有要求。
3. `LLaMA Factory` 集成入口已预留，可通过 `use_llamafactory: true` 启用外部命令调度。

## 当前实现状态

- 已完成统一代码骨架、配置系统、CLI、蒸馏 / 微调 / 推理主流程
- 已补充 Docker、示例配置、基础测试
- 具体训练与推理效果仍依赖你的模型、数据集与运行环境

