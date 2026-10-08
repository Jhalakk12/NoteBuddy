#!/usr/bin/env python3
"""Ingest course materials into the persistent Chroma knowledge base."""

import argparse
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge.embeddings import OllamaEmbeddingClient
from knowledge.ingest import ingest_directory
from knowledge.store import KnowledgeStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--documents",
        type=Path,
        default=PROJECT_ROOT / "data" / "documents",
        help="Directory containing PDF, TXT, and Markdown course files.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "chroma",
        help="Persistent Chroma directory.",
    )
    parser.add_argument("--reset", action="store_true", help="Rebuild the collection.")
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    embedder = OllamaEmbeddingClient(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
    )
    store = KnowledgeStore(args.database, embedder)
    report = ingest_directory(args.documents, store, reset=args.reset)
    print(json.dumps(report.__dict__, indent=2))


if __name__ == "__main__":
    main()
