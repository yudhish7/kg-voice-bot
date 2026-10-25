"""Document ingestion: text -> chunks -> triples -> knowledge graph."""

from __future__ import annotations

import logging
from pathlib import Path

from .chunker import chunk_text
from .extract import TripleExtractor
from .graph import KnowledgeGraph

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".txt", ".md"}


class Ingestor:
    def __init__(
        self,
        kg: KnowledgeGraph,
        extractor: TripleExtractor,
        max_chars: int = 500,
        overlap_sentences: int = 1,
    ) -> None:
        self.kg = kg
        self.extractor = extractor
        self.max_chars = max_chars
        self.overlap_sentences = overlap_sentences

    def ingest_text(self, doc_id: str, text: str) -> dict[str, int]:
        """Add (or replace) one document. Re-ingesting the same ``doc_id`` is idempotent."""
        if not doc_id.strip():
            raise ValueError("doc_id must not be empty")
        chunks = chunk_text(text, doc_id, self.max_chars, self.overlap_sentences)
        # Extraction may call an LLM, so do it before taking the graph lock.
        extracted = [(chunk, self.extractor.extract(chunk)) for chunk in chunks]
        facts = 0
        with self.kg.lock:
            self.kg.remove_document(doc_id)
            for chunk, triples in extracted:
                self.kg.add_chunk(chunk)
                facts += sum(self.kg.add_triple(t) for t in triples)
        log.info("Ingested %s: %d chunks, %d facts", doc_id, len(chunks), facts)
        return {"chunks": len(chunks), "facts": facts}

    def ingest_path(self, path: Path) -> dict[str, int]:
        """Ingest a file, or every ``.txt``/``.md`` file under a directory."""
        if path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
        elif path.is_file():
            files = [path]
        else:
            raise FileNotFoundError(path)
        totals = {"documents": 0, "chunks": 0, "facts": 0}
        for file in files:
            doc_id = file.relative_to(path).as_posix() if path.is_dir() else file.name
            result = self.ingest_text(doc_id, file.read_text(encoding="utf-8"))
            totals["documents"] += 1
            totals["chunks"] += result["chunks"]
            totals["facts"] += result["facts"]
        return totals
