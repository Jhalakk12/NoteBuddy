"""Typed async HTTP client for downstream Exercise 4 services."""

import httpx

from services.contracts import (
    DocumentDeleteResponse,
    GenerateResponse,
    IngestResponse,
    KnowledgeStatus,
    RetrieveResponse,
    ServiceStatus,
    UploadResponse,
)


class DownstreamServiceError(RuntimeError):
    """Raised when an orchestrated service call fails."""


class ServiceClient:
    def __init__(
        self,
        retrieval_url: str,
        llm_url: str,
        data_url: str,
        timeout_seconds: float = 300,
    ) -> None:
        self.retrieval_url = retrieval_url.rstrip("/")
        self.llm_url = llm_url.rstrip("/")
        self.data_url = data_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def retrieve(self, question: str, top_k: int = 4) -> RetrieveResponse:
        payload = await self._post(
            f"{self.retrieval_url}/retrieve",
            {"question": question, "top_k": top_k},
        )
        return RetrieveResponse.model_validate(payload)

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 256,
    ) -> GenerateResponse:
        payload = await self._post(
            f"{self.llm_url}/generate",
            {"prompt": prompt, "model": model, "max_tokens": max_tokens},
        )
        return GenerateResponse.model_validate(payload)

    async def retrieve_repository(self, question: str, top_k: int = 4) -> RetrieveResponse:
        payload = await self._post(f'{self.retrieval_url}/retrieve/repository', {'question': question, 'top_k': top_k})
        return RetrieveResponse.model_validate(payload)

    async def knowledge_status(self) -> KnowledgeStatus:
        payload = await self._get(f"{self.data_url}/documents")
        return KnowledgeStatus.model_validate(payload)

    async def ingest(self, reset: bool = False) -> IngestResponse:
        payload = await self._post(f"{self.data_url}/ingest", {"reset": reset})
        return IngestResponse.model_validate(payload)

    async def upload_documents(
        self,
        files: list[tuple[str, bytes, str]],
    ) -> UploadResponse:
        multipart = [
            ("files", (name, content, content_type))
            for name, content, content_type in files
        ]
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.data_url}/documents/upload",
                    files=multipart,
                )
                response.raise_for_status()
                return UploadResponse.model_validate(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            raise DownstreamServiceError(
                f"Document upload failed at {self.data_url}: {exc}"
            ) from exc

    async def delete_document(self, filename: str) -> DocumentDeleteResponse:
        from urllib.parse import quote

        payload = await self._delete(f"{self.data_url}/documents/{quote(filename, safe='')}")
        return DocumentDeleteResponse.model_validate(payload)

    async def service_status(self, name: str, url: str) -> ServiceStatus:
        endpoint = f"{url.rstrip('/')}/health"
        try:
            payload = await self._get(endpoint)
            detail_parts = [f"{key}: {value}" for key, value in payload.get("details", {}).items()]
            return ServiceStatus(
                name=name,
                url="internal",
                status=str(payload.get("status", "ok")),
                detail=", ".join(detail_parts) or None,
            )
        except DownstreamServiceError as exc:
            return ServiceStatus(
                name=name,
                url="internal",
                status="unavailable",
                detail=f"{name} is not responding",
            )

    async def _get(self, url: str) -> dict:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise DownstreamServiceError(f"Downstream call failed at {url}: {exc}") from exc

    async def _post(self, url: str, json: dict) -> dict:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, json=json)
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise DownstreamServiceError(f"Downstream call failed at {url}: {exc}") from exc

    async def _delete(self, url: str) -> dict:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.delete(url)
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise DownstreamServiceError(f"Downstream delete failed at {url}: {exc}") from exc
