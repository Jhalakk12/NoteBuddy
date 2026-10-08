"""Paragraph-aware chunking that never crosses a page or section boundary."""

from hashlib import sha256
import re

from knowledge.models import Chunk, DocumentUnit


PARAGRAPH_RE = re.compile(r"\n\s*\n+")


def chunk_units(
    units: list[DocumentUnit],
    max_characters: int = 1_200,
    overlap_characters: int = 200,
) -> list[Chunk]:
    if max_characters < 100:
        raise ValueError("max_characters must be at least 100")
    if overlap_characters < 0 or overlap_characters >= max_characters:
        raise ValueError("overlap_characters must be between 0 and max_characters")

    chunks: list[Chunk] = []
    for unit in units:
        pieces = _split_text(unit.text, max_characters, overlap_characters)
        for chunk_index, text in enumerate(pieces, start=1):
            citation = _citation(unit)
            stable_key = (
                f"{unit.source}|{unit.page}|{unit.section}|{chunk_index}|{text}"
            )
            chunks.append(
                Chunk(
                    id=sha256(stable_key.encode("utf-8")).hexdigest(),
                    text=text,
                    source=unit.source,
                    page=unit.page,
                    section=unit.section,
                    chunk_index=chunk_index,
                    citation=citation,
                )
            )
    return chunks


def _citation(unit: DocumentUnit) -> str:
    if unit.page is not None:
        return f"{unit.source}, page/slide {unit.page}"
    return f"{unit.source}, section: {unit.section}"


def _split_text(text: str, limit: int, overlap: int) -> list[str]:
    paragraphs = [part.strip() for part in PARAGRAPH_RE.split(text) if part.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        for piece in _split_oversized(paragraph, limit):
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if len(candidate) <= limit:
                current = candidate
                continue
            if current:
                chunks.append(current)
                prefix = _overlap_tail(current, overlap)
                current = f"{prefix}\n\n{piece}".strip() if prefix else piece
                if len(current) > limit:
                    current = piece
            else:
                chunks.append(piece)
    if current:
        chunks.append(current)
    return chunks


def _split_oversized(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        added = len(word) + (1 if current else 0)
        if current and length + added > limit:
            pieces.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += added
    if current:
        pieces.append(" ".join(current))
    return pieces


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap == 0:
        return ""
    tail = text[-overlap:]
    first_space = tail.find(" ")
    return tail[first_space + 1 :].strip() if first_space >= 0 else tail.strip()

