"""Small async client for Ollama's HTTP generation API."""

from dataclasses import dataclass
from typing import Any

import httpx


class OllamaError(RuntimeError):
    """Raised when Ollama cannot produce a valid response."""


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    model: str
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    eval_duration_ns: int | None = None
    model_memory_bytes: int | None = None
    gpu_memory_bytes: int | None = None


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: float = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def generate(self, prompt: str) -> str:
        return (await self.generate_detailed(prompt)).answer

    async def generate_detailed(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 256,
    ) -> GenerationResult:
        selected_model = model or self.model
        payload: dict[str, Any] = {
            "model": selected_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": max_tokens},
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise OllamaError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure `ollama serve` is running."
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaError("Ollama timed out while generating the response.") from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise OllamaError(
                f"Ollama returned HTTP {exc.response.status_code}: {detail}"
            ) from exc

        try:
            response_payload = response.json()
            answer = response_payload["response"]
        except (ValueError, KeyError, TypeError) as exc:
            raise OllamaError("Ollama returned an invalid response payload.") from exc

        if not isinstance(answer, str) or not answer.strip():
            raise OllamaError("Ollama returned an empty response.")
        memory, gpu_memory = await self._loaded_model_memory(selected_model)
        return GenerationResult(
            answer=answer.strip(),
            model=str(response_payload.get("model", selected_model)),
            total_duration_ns=_optional_int(response_payload.get("total_duration")),
            load_duration_ns=_optional_int(response_payload.get("load_duration")),
            prompt_tokens=_optional_int(response_payload.get("prompt_eval_count")),
            completion_tokens=_optional_int(response_payload.get("eval_count")),
            eval_duration_ns=_optional_int(response_payload.get("eval_duration")),
            model_memory_bytes=memory,
            gpu_memory_bytes=gpu_memory,
        )

    async def _loaded_model_memory(self, model: str) -> tuple[int | None, int | None]:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/ps")
                response.raise_for_status()
                for row in response.json().get("models", []):
                    name = str(row.get("name", ""))
                    if name == model or name.split(":", 1)[0] == model.split(":", 1)[0]:
                        return _optional_int(row.get("size")), _optional_int(row.get("size_vram"))
        except (httpx.HTTPError, ValueError, TypeError):
            pass
        return None, None


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None
