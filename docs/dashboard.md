# Dashboard

`llama-vllm` 提供了一个轻量可视化后台，用于：

- 配置文件加载与校验
- 训练 preflight 检查
- 可执行命令预览

> 该后台默认是 dry-run，不会直接启动训练任务。

## 启动

```powershell
cd D:\projects\github\llama_vllm
$env:PYTHONPATH = "D:\projects\github\llama_vllm\src"
python -m llama_vllm.cli.main dashboard --host 127.0.0.1 --port 7860 --db-path .dashboard/history.db
```

然后访问：`http://127.0.0.1:7860`

## API

- `POST /api/preflight`
- `POST /api/command-preview`
- `GET /api/history?limit=50`
- `GET /api/history/{id}`
- `DELETE /api/history`

`/api/history` 支持可选过滤参数：

- `action=preflight|command-preview`
- `task_type=distill|finetune|infer`
- `ok=true|false`

按过滤条件清理示例：

```text
DELETE /api/history?action=preflight&task_type=distill
```

请求体示例：

```json
{
  "task_type": "distill",
  "config_path": "configs/distillation/feature_distill.yaml",
  "overrides": ["training.bf16=false"],
  "shell_style": "auto"
}
```

