"""Load PDF, text, and Markdown documents without crossing citation boundaries."""

from pathlib import Path
import re

import pymupdf

from knowledge.models import DocumentUnit


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def discover_documents(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_document(path: Path, document_root: Path) -> list[DocumentUnit]:
    source = path.relative_to(document_root).as_posix()
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path, source)
    if suffix in {".txt", ".md"}:
        return _load_text(path, source)
    raise ValueError(f"Unsupported document type: {path.suffix}")


def _load_pdf(path: Path, source: str) -> list[DocumentUnit]:
    units: list[DocumentUnit] = []
    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True).strip()
            if text:
                units.append(
                    DocumentUnit(
                        text=text,
                        source=source,
                        page=page_number,
                        section=f"Page {page_number}",
                    )
                )
    return units


def _load_text(path: Path, source: str) -> list[DocumentUnit]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    units: list[DocumentUnit] = []
    current_heading = "Document"
    current_lines: list[str] = []

    def flush() -> None:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            units.append(
                DocumentUnit(
                    text=section_text,
                    source=source,
                    page=None,
                    section=current_heading,
                )
            )

    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            flush()
            current_lines = []
            current_heading = match.group(2).strip()
        else:
            current_lines.append(line)
    flush()
    return units
