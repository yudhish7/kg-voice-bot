"""Answer generation from retrieved evidence."""

from __future__ import annotations

from typing import Protocol

from .llm import LLM
from .models import Evidence
from .text import content_tokens, normalize, split_sentences

FALLBACK_ANSWER = "I'm sorry, I don't have information about that."

SYSTEM_PROMPT = (
    "You are a friendly voice assistant for a service desk. Answer the customer's question "
    "using ONLY the facts and passages provided. Your answer will be read aloud, so reply in "
    "one to three short, natural sentences with no markdown, lists or citations. If the "
    f"context does not contain the answer, reply exactly: {FALLBACK_ANSWER}"
)


class Generator(Protocol):
    def generate(self, question: str, evidence: Evidence) -> str: ...


def build_prompt(question: str, evidence: Evidence) -> str:
    facts = "\n".join(
        f"- {t.subject} {t.relation.replace('_', ' ')} {t.object}" for t in evidence.triples
    )
    passages = "\n".join(f"[{c.id}] {c.text}" for c, _ in evidence.chunks)
    return (
        f"Facts from the knowledge graph:\n{facts or '(none)'}\n\n"
        f"Passages:\n{passages or '(none)'}\n\n"
        f"Question: {question}"
    )


class LLMGenerator:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def generate(self, question: str, evidence: Evidence) -> str:
        return (
            self._llm.complete(SYSTEM_PROMPT, build_prompt(question, evidence)) or FALLBACK_ANSWER
        )


class ExtractiveGenerator:
    """Offline generator: returns the retrieved sentences that best match the question.

    Useful without an API key, for tests, and as a transparent baseline.
    """

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, evidence: Evidence) -> str:
        q_terms = set(content_tokens(question))
        entities = [normalize(e) for e in evidence.entities]
        scored: list[tuple[float, int, str]] = []
        order = 0
        for chunk, _ in evidence.chunks:
            for sentence in split_sentences(chunk.text):
                order += 1
                overlap = len(q_terms & set(content_tokens(sentence)))
                if not overlap:
                    continue
                bonus = 0.5 * sum(e in normalize(sentence) for e in entities)
                scored.append((overlap + bonus, order, sentence))
        if not scored:
            return FALLBACK_ANSWER
        ranked = sorted(scored, key=lambda x: (-x[0], x[1]))
        # Keep runner-up sentences only if they are nearly as relevant as the best one.
        best = [r for r in ranked if r[0] >= 0.8 * ranked[0][0]][: self.max_sentences]
        return " ".join(s for _, _, s in sorted(best, key=lambda x: x[1]))
