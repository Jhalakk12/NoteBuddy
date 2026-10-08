"""LLM Service: the only service that talks to Ollama generation APIs."""

import httpx

from fastapi import Depends, FastAPI, HTTPException, status

from app.config import settings
from app.ollama_client import OllamaClient, OllamaError
from services.contracts import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelListResponse,
    ModelOption,
)


app = FastAPI(title="NoteBuddy — LLM Service", version="0.4.0")
MODEL_LABELS = {
    "codellama": "Code Llama",
    "starcoder2:3b": "StarCoder2 3B",
    "qwen2.5-coder:1.5b": "Qwen2.5 Coder 1.5B",
}


def get_llm() -> OllamaClient:
    return OllamaClient(
        settings.ollama_base_url,
        settings.ollama_model,
        settings.ollama_timeout_seconds,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/version")
            response.raise_for_status()
            version = str(response.json().get("version", "unknown"))
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ollama is unavailable: {exc}",
        ) from exc
    return HealthResponse(
        service="llm",
        details={"runtime": "Ollama", "version": version, "model": settings.ollama_model},
    )


@app.get("/models", response_model=ModelListResponse)
async def models() -> ModelListResponse:
    approved = list(dict.fromkeys((settings.ollama_model, *settings.week4_models)))
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
            installed = {
                str(item.get("name", "")).removesuffix(":latest")
                for item in response.json().get("models", [])
            }
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot list Ollama models: {exc}",
        ) from exc
    return ModelListResponse(
        default_model=settings.ollama_model,
        models=[
            ModelOption(
                id=model,
                label=MODEL_LABELS.get(model, model),
                available=model.removesuffix(":latest") in installed,
            )
            for model in approved
        ],
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    llm: OllamaClient = Depends(get_llm),
) -> GenerateResponse:
    selected_model = request.model or settings.ollama_model
    allowed_models = {settings.ollama_model, *settings.week4_models}
    if selected_model not in allowed_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model is not approved for this evaluation. Allowed: {', '.join(sorted(allowed_models))}",
        )
    try:
        result = await llm.generate_detailed(
            request.prompt,
            model=selected_model,
            max_tokens=request.max_tokens,
        )
    except OllamaError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return GenerateResponse(**result.__dict__)
