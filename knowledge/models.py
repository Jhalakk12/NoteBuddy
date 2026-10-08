"""Internal document and chunk models."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DocumentUnit:
    """Text contained by a citation-safe source boundary."""

    text: str
    source: str
    page: int | None
    section: str


@dataclass(frozen=True)
class Chunk:
    """A vector-store document with traceable source metadata."""

    id: str
    text: str
    source: str
    page: int | None
    section: str
    chunk_index: int
    citation: str

    def metadata(self) -> dict[str, Any]:
        # Chroma metadata values cannot be None.
        return {
            "source": self.source,
            "page": self.page if self.page is not None else 0,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "citation": self.citation,
        }


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    page: int | None
    section: str
    citation: str
    distance: float

    @property
    def relevance_score(self) -> float:
        # Cosine distance is 0 for identical vectors and approaches 2 for opposites.
        return max(0.0, min(1.0, 1.0 - self.distance / 2.0))
