"""Plain data types shared across the package."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """A passage of source text. ``id`` is unique across the knowledge base."""

    id: str
    doc_id: str
    text: str


@dataclass(frozen=True)
class Triple:
    """A (subject, relation, object) fact together with the chunk it came from."""

    subject: str
    relation: str
    object: str
    chunk_id: str


@dataclass
class Evidence:
    """Everything retrieved for one question."""

    entities: list[str] = field(default_factory=list)
    triples: list[Triple] = field(default_factory=list)
    chunks: list[tuple[Chunk, float]] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.chunks and not self.triples


@dataclass
class Answer:
    text: str
    evidence: Evidence

    @property
    def sources(self) -> list[str]:
        return [chunk.id for chunk, _ in self.evidence.chunks]


@dataclass
class VoiceReply:
    transcript: str
    answer: Answer
    audio: bytes
    content_type: str
