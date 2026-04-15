"""FastAPI server with OpenAI-compatible endpoints backed by vLLM."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from llama_vllm.config.schemas import InferenceConfig
from llama_vllm.inference.engine import VLLMEngineWrapper
from llama_vllm.inference.streaming import stream_text
from llama_vllm.utils.logging import get_logger

logger = get_logger(__name__)


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False


def _build_prompt_from_messages(messages: List[ChatMessage]) -> str:
    parts = []
    for msg in messages:
        parts.append(f"{msg.role.capitalize()}: {msg.content}")
    parts.append("Assistant:")
    return "\n".join(parts)


def create_app(config: InferenceConfig) -> FastAPI:
    engine = VLLMEngineWrapper(config)
    app = FastAPI(title="llama-vllm server", version="0.1.0")

    def _check_auth(auth_header: Optional[str]) -> None:
        api_key = config.server.api_key
        if not api_key:
            return
        if not auth_header or not auth_header.startswith("Bearer ") or auth_header[7:] != api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "model": config.model_name_or_path}

    @app.get("/metrics")
    def metrics() -> Dict[str, Any]:
        return {
            "model": config.model_name_or_path,
            "tensor_parallel_size": config.tensor_parallel_size,
            "mode": config.mode,
        }

    @app.post("/v1/completions")
    def completions(request: CompletionRequest, authorization: Optional[str] = Header(default=None)):
        _check_auth(authorization)
        overrides = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.stream:
            def event_gen():
                for chunk in stream_text(config, request.prompt):
                    payload = {
                        "id": f"cmpl-{uuid.uuid4().hex}",
                        "object": "text_completion",
                        "choices": [{"text": chunk, "index": 0, "finish_reason": None}],
                    }
                    yield {"data": json.dumps(payload, ensure_ascii=False)}
                yield {"data": "[DONE]"}

            return EventSourceResponse(event_gen())

        outputs = engine.generate([request.prompt], sampling_overrides=overrides)
        text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model or config.model_name_or_path,
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest, authorization: Optional[str] = Header(default=None)):
        _check_auth(authorization)
        prompt = _build_prompt_from_messages(request.messages)
        completion_request = CompletionRequest(
            model=request.model,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream,
        )
        if request.stream:
            return completions(completion_request, authorization)

        response = completions(completion_request, authorization)
        content = response["choices"][0]["text"]
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model or config.model_name_or_path,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
        }

    return app


def run_server(config: InferenceConfig) -> None:
    import uvicorn

    app = create_app(config)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
    )

