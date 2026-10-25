"""Graph-aware hybrid retrieval.

Pipeline for one question:

1. **Entity linking** - find graph nodes named in the question (the *seeds*).
2. **Graph expansion** - breadth-first walk ``hops`` steps out from the seeds, so facts about
   related entities (the router a plan includes, what that router supports) come along.
3. **Chunk ranking** - blend lexical BM25 with graph proximity: a chunk scores higher when it
   mentions entities close to the seeds. With no seeds this degrades to plain BM25.
4. **Fact selection** - keep the facts inside the neighbourhood, nearest first.
"""

from __future__ import annotations

from .bm25 import BM25
from .graph import KnowledgeGraph
from .models import Chunk, Evidence, Triple
from .text import content_tokens, normalize


class GraphRetriever:
    def __init__(
        self,
        kg: KnowledgeGraph,
        hops: int = 2,
        top_k_chunks: int = 4,
        max_facts: int = 15,
        graph_weight: float = 0.5,
    ) -> None:
        if not 0.0 <= graph_weight <= 1.0:
            raise ValueError("graph_weight must be between 0 and 1")
        self.kg = kg
        self.hops = hops
        self.top_k_chunks = top_k_chunks
        self.max_facts = max_facts
        self.graph_weight = graph_weight
        self._index: BM25 | None = None
        self._index_version = -1

    def _bm25(self) -> BM25:
        if self._index is None or self._index_version != self.kg.version:
            self._index = BM25({cid: c.text for cid, c in self.kg.chunks.items()})
            self._index_version = self.kg.version
        return self._index

    def retrieve(self, question: str) -> Evidence:
        with self.kg.lock:
            return self._retrieve(question)

    def _retrieve(self, question: str) -> Evidence:
        kg = self.kg
        seeds = kg.link_entities(question)
        dist = kg.distances(seeds, self.hops)

        lexical = self._bm25().scores(question)
        top = max(lexical.values(), default=0.0)
        lexical = {cid: s / top for cid, s in lexical.items()} if top else {}

        graph_scores: dict[str, float] = {}
        for node, d in dist.items():
            for cid in kg.mentions(node):
                graph_scores[cid] = max(graph_scores.get(cid, 0.0), 1.0 / (1 + d))

        w = self.graph_weight if seeds else 0.0
        combined = {
            cid: (1 - w) * lexical.get(cid, 0.0) + w * graph_scores.get(cid, 0.0)
            for cid in set(lexical) | set(graph_scores)
        }
        ranked = sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))
        chunks: list[tuple[Chunk, float]] = [
            (kg.chunks[cid], score) for cid, score in ranked[: self.top_k_chunks] if score > 0
        ]

        return Evidence(
            entities=[kg.label(n) for n in seeds],
            triples=self._facts(question, dist),
            chunks=chunks,
        )

    def _facts(self, question: str, dist: dict[str, int]) -> list[Triple]:
        if not dist:
            return []
        q_terms = set(content_tokens(question))

        def rank(t: Triple) -> tuple[int, int, str, str, str]:
            nearest = min(dist[normalize(t.subject)], dist[normalize(t.object)])
            overlap = len(q_terms & set(content_tokens(f"{t.relation} {t.object}")))
            return (nearest, -overlap, t.subject, t.relation, t.object)

        inside = self.kg.triples_among(set(dist))
        return sorted(inside, key=rank)[: self.max_facts]
