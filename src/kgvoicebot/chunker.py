"""Sentence-aware chunking with sentence overlap."""

from __future__ import annotations

from .models import Chunk
from .text import split_sentences


def chunk_text(
    text: str, doc_id: str, max_chars: int = 500, overlap_sentences: int = 1
) -> list[Chunk]:
    """Split ``text`` into chunks of at most ~``max_chars`` characters.

    Chunks always end on a sentence boundary. A single sentence longer than
    ``max_chars`` becomes its own chunk rather than being cut mid-sentence.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")

    sentences = split_sentences(text)
    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        added = len(sentence) + (1 if current else 0)
        if current and size + added > max_chars:
            groups.append(current)
            # Carry the tail over, but never so much that we stall.
            current = current[-overlap_sentences:] if overlap_sentences else []
            if current and sum(len(s) + 1 for s in current) + len(sentence) > max_chars:
                current = []
            size = sum(len(s) + 1 for s in current) - 1 if current else 0
            added = len(sentence) + (1 if current else 0)
        current.append(sentence)
        size += added
    if current:
        groups.append(current)

    # Drop a trailing group that only repeats overlap from the previous one.
    if (
        len(groups) > 1
        and overlap_sentences
        and len(groups[-1]) <= overlap_sentences
        and groups[-1] == groups[-2][-len(groups[-1]) :]
    ):
        groups.pop()

    return [
        Chunk(id=f"{doc_id}:{i}", doc_id=doc_id, text=" ".join(g)) for i, g in enumerate(groups)
    ]
