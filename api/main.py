"""Optional local HTTP API. Do not expose publicly without authentication/rate limits."""

import os
from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from inference.generate import generate_text, load_model

app = FastAPI(title="JakeLLM local API", version="0.2.0")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4096)
    max_new_tokens: int = Field(default=50, ge=1, le=256)
    temperature: float = Field(default=1.0, ge=0, le=2)
    top_k: int = Field(default=20, ge=0, le=100)


class GenerateResponse(BaseModel):
    text: str


@lru_cache(maxsize=1)
def runtime():
    return load_model(
        checkpoint_path=os.getenv("JAKELLM_CHECKPOINT", "checkpoints/jakellm.pt"),
        tokenizer_dir=os.getenv("JAKELLM_TOKENIZER_DIR", "tokenizer"),
        device=os.getenv("JAKELLM_DEVICE", "auto"),
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).resolve().parent.parent / "web" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": runtime.cache_info().currsize > 0}


@app.get("/ready")
def ready():
    try:
        runtime()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready"}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    try:
        model, tokenizer = runtime()
        return GenerateResponse(text=generate_text(
            model, tokenizer, request.prompt, request.max_new_tokens,
            request.temperature, request.top_k,
        ))
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
