#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat server using transformers and optional PEFT adapters."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 8
    temperature: float = 0.0


def create_app(model_name: str, adapter_path: str | None, served_model_name: str, device: str) -> FastAPI:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=device,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest) -> dict[str, Any]:
        messages = [{"role": item.role, "content": item.content} for item in req.messages]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        do_sample = float(req.temperature) > 0.0
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=int(req.max_tokens),
                do_sample=do_sample,
                temperature=max(float(req.temperature), 1e-5) if do_sample else None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        content = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "model": served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve a local transformers chat model behind an OpenAI-compatible API")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--served-model-name", default="local-transformers")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "auto":
        device = "auto"
    elif args.device == "cuda":
        device = "cuda:0"
    else:
        device = "cpu"
    app = create_app(args.model, args.adapter, args.served_model_name, device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
