"""Environment-based configuration for Exercise 1."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "codellama")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    ollama_timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
    chroma_path: Path = Path(os.getenv("CHROMA_PATH", "data/chroma"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    week4_models: tuple[str, ...] = tuple(
        model.strip()
        for model in os.getenv(
            "WEEK4_MODELS",
            "codellama,starcoder2:3b,qwen2.5-coder:1.5b",
        ).split(",")
        if model.strip()
    )


settings = Settings()
