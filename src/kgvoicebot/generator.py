"""Answer generation from retrieved evidence."""

from __future__ import annotations

import math
from typing import Protocol

from .llm import LLM
from .models import Evidence
from .text import content_tokens, normalize, split_sentences

_MIN_WORDS = 5  # shorter lines are treated as headings, not answers

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

        # 1. Collect each distinct candidate sentence once. Overlapping chunks repeat
        #    sentences, and saying the same thing twice sounds silly when spoken aloud.
        sentences: list[str] = []
        seen: set[str] = set()
        for chunk, _ in evidence.chunks:
            for sentence in split_sentences(chunk.text):
                if sentence not in seen:
                    seen.add(sentence)
                    sentences.append(sentence)

        # 2. Rare words matter more. "card" is in most sentences, "lost" in only one,
        #    so "lost" tells us more about which sentence the user needs.
        words = [set(content_tokens(sentence)) for sentence in sentences]
        total = len(sentences)

        def weight(term: str) -> float:
            containing = sum(term in w for w in words)
            return math.log(1 + total / containing) if containing else 0.0

        scored: list[tuple[float, int, str]] = []
        for order, (sentence, sentence_words) in enumerate(zip(sentences, words, strict=True)):
            # Very short "sentences" are headings ("Acme Bank help centre."), not facts.
            if len(sentence.split()) < _MIN_WORDS:
                continue
            overlap = sum(weight(t) for t in q_terms & sentence_words)
            if not overlap:
                continue
            bonus = 0.5 * sum(e in normalize(sentence) for e in entities)
            scored.append((overlap + bonus, order, sentence))
        if not scored:
            return FALLBACK_ANSWER
        ranked = sorted(scored, key=lambda x: (-x[0], x[1]))
        # Keep runner-up sentences only if they are nearly as relevant as the best one.
        best = [r for r in ranked if r[0] >= 0.9 * ranked[0][0]][: self.max_sentences]
        return " ".join(s for _, _, s in sorted(best, key=lambda x: x[1]))