"""The knowledge graph: entities as nodes, relations as edges, chunks as provenance."""

from __future__ import annotations

import json
import re
import threading
from collections import deque
from pathlib import Path

import networkx as nx

from .models import Chunk, Triple
from .text import content_tokens, normalize

_FORMAT_VERSION = 1


class KnowledgeGraph:
    """A directed multigraph of entities backed by the chunks that mention them.

    Nodes are keyed by the normalised entity name, so "Home Loan" and "home loan." merge.
    Every node records which chunks mention it; every edge records which chunks assert it.
    """

    def __init__(self) -> None:
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.chunks: dict[str, Chunk] = {}
        self._triples: set[Triple] = set()
        self.lock = threading.RLock()  # guards mutation vs. retrieval when shared across threads
        self.version = 0  # bumped on every mutation; lets indexes detect staleness

    # ------------------------------------------------------------------ building
    def add_chunk(self, chunk: Chunk) -> None:
        if self.chunks.get(chunk.id) != chunk:
            self.chunks[chunk.id] = chunk
            self.version += 1

    def add_triple(self, triple: Triple) -> bool:
        """Add a fact. Returns False if the exact fact is already present."""
        if triple in self._triples:
            return False
        subj, obj = normalize(triple.subject), normalize(triple.object)
        if not subj or not obj:
            return False
        self._triples.add(triple)
        for key, label in ((subj, triple.subject), (obj, triple.object)):
            if key not in self.graph:
                self.graph.add_node(key, label=label.strip(), mentions=set())
            self.graph.nodes[key]["mentions"].add(triple.chunk_id)
        if self.graph.has_edge(subj, obj, key=triple.relation):
            self.graph[subj][obj][triple.relation]["chunks"].add(triple.chunk_id)
        else:
            self.graph.add_edge(subj, obj, key=triple.relation, chunks={triple.chunk_id})
        self.version += 1
        return True

    def remove_document(self, doc_id: str) -> None:
        """Drop a document's chunks and facts (used to make re-ingestion idempotent)."""
        gone = {cid for cid, c in self.chunks.items() if c.doc_id == doc_id}
        if not gone:
            return
        keep_chunks = [c for cid, c in self.chunks.items() if cid not in gone]
        keep_triples = [t for t in self._triples if t.chunk_id not in gone]
        version = self.version
        self.graph = nx.MultiDiGraph()
        self.chunks, self._triples = {}, set()
        for chunk in keep_chunks:
            self.add_chunk(chunk)
        for triple in keep_triples:
            self.add_triple(triple)
        self.version = version + 1

    # ------------------------------------------------------------------- queries
    @property
    def triples(self) -> list[Triple]:
        return sorted(self._triples, key=lambda t: (t.chunk_id, t.subject, t.relation, t.object))

    def label(self, node: str) -> str:
        return str(self.graph.nodes[node]["label"])

    def mentions(self, node: str) -> set[str]:
        return set(self.graph.nodes[node]["mentions"])

    def distances(self, seeds: list[str], hops: int) -> dict[str, int]:
        """Undirected BFS distance from the nearest seed, up to ``hops``."""
        dist = {s: 0 for s in seeds if s in self.graph}
        queue = deque(dist)
        while queue:
            node = queue.popleft()
            if dist[node] >= hops:
                continue
            for nbr in set(self.graph.successors(node)) | set(self.graph.predecessors(node)):
                if nbr not in dist:
                    dist[nbr] = dist[node] + 1
                    queue.append(nbr)
        return dist

    def link_entities(self, text: str, min_overlap: float = 0.6) -> list[str]:
        """Find graph nodes named in free text.

        Pass 1 matches whole entity names (longest first). Pass 2 is a fuzzy fallback for
        multi-word entities the user abbreviated ("Premium Savings" for "Premium Savings
        account"): a node links when at least ``min_overlap`` of its words appear in the
        text not already consumed by pass 1.
        """
        haystack = f" {normalize(text)} "
        found: list[str] = []
        # Longer names first so "premium savings account" wins over "premium".
        for node in sorted(self.graph.nodes, key=len, reverse=True):
            pattern = rf"(?<![a-z0-9]){re.escape(node)}(?![a-z0-9])"
            match = re.search(pattern, haystack)
            if match:
                found.append(node)
                # Blank out the match so shorter names inside it are not double counted.
                haystack = haystack[: match.start()] + " " * len(node) + haystack[match.end() :]

        remaining = set(content_tokens(haystack))
        for node in self.graph.nodes:
            if node in found:
                continue
            words = content_tokens(node)
            if len(words) >= 2 and sum(w in remaining for w in words) / len(words) >= min_overlap:
                found.append(node)
        return found

    def triples_among(self, nodes: set[str]) -> list[Triple]:
        return [
            t
            for t in self.triples
            if normalize(t.subject) in nodes and normalize(t.object) in nodes
        ]

    def stats(self) -> dict[str, int]:
        return {
            "chunks": len(self.chunks),
            "entities": self.graph.number_of_nodes(),
            "relations": len(self._triples),
        }

    # --------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "format": _FORMAT_VERSION,
            "chunks": [vars(c) for c in self.chunks.values()],
            "triples": [vars(t) for t in self.triples],
        }

    @classmethod
    def from_dict(cls, data: dict) -> KnowledgeGraph:
        if data.get("format") != _FORMAT_VERSION:
            raise ValueError(f"Unsupported graph format: {data.get('format')!r}")
        kg = cls()
        for c in data["chunks"]:
            kg.add_chunk(Chunk(**c))
        for t in data["triples"]:
            kg.add_triple(Triple(**t))
        return kg

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(path)  # atomic: a crash never leaves a half-written graph

    @classmethod
    def load(cls, path: Path) -> KnowledgeGraph:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
