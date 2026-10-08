"""Repository indexing for multi-file codebase questions."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from knowledge.models import Chunk
from knowledge.store import KnowledgeStore


CODE_EXTENSIONS = {".py", ".js", ".html", ".css", ".yaml", ".yml", ".toml", ".md"}
IGNORED_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "chroma",
    "evaluations",
}


@dataclass(frozen=True)
class RepositoryIngestionReport:
    files: int
    chunks: int
    collection_total: int


def discover_repository_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in CODE_EXTENSIONS
        and not any(part in IGNORED_PARTS or part.startswith(".") for part in path.relative_to(root).parts)
        and path.stat().st_size <= 500_000
    )


def repository_chunks(
    root: Path,
    *,
    lines_per_chunk: int = 120,
    overlap_lines: int = 20,
) -> tuple[list[Path], list[Chunk]]:
    files = discover_repository_files(root)
    chunks: list[Chunk] = []
    stride = lines_per_chunk - overlap_lines
    for path in files:
        relative = path.relative_to(root).as_posix()
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for chunk_index, start in enumerate(range(0, len(lines), stride), start=1):
            selected = lines[start : start + lines_per_chunk]
            if not selected:
                continue
            end = start + len(selected)
            section = f"lines {start + 1}-{end}"
            citation = f"{relative}, {section}"
            text = f"FILE: {relative}\n{section}\n" + "\n".join(selected)
            stable_key = f"repository|{relative}|{start}|{text}"
            chunks.append(
                Chunk(
                    id=sha256(stable_key.encode("utf-8")).hexdigest(),
                    text=text,
                    source=relative,
                    page=None,
                    section=section,
                    chunk_index=chunk_index,
                    citation=citation,
                )
            )
    return files, chunks


def ingest_repository(
    root: Path,
    store: KnowledgeStore,
    *,
    reset: bool = True,
) -> RepositoryIngestionReport:
    files, chunks = repository_chunks(root)
    if not files:
        raise FileNotFoundError(f"No repository source files found in {root}")
    if reset:
        store.reset()
    store.upsert(chunks)
    return RepositoryIngestionReport(
        files=len(files),
        chunks=len(chunks),
        collection_total=store.count(),
    )
