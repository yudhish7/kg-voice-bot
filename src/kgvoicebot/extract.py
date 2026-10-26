"""Triple extraction: turn chunks of prose into (subject, relation, object) facts.

Two interchangeable extractors implement :class:`TripleExtractor`:

* :class:`RuleBasedExtractor` - offline, deterministic, pattern based. A baseline that
  works well on short declarative sentences ("The Home Loan requires a credit score of 700.").
* :class:`LLMExtractor` - asks an LLM for JSON triples. Much better recall on free text.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

from .llm import LLM
from .models import Chunk, Triple
from .text import split_sentences

log = logging.getLogger(__name__)


class TripleExtractor(Protocol):
    def extract(self, chunk: Chunk) -> list[Triple]: ...


_RELATIONS = [
    "is eligible for",
    "is available in",
    "is located in",
    "is part of",
    "is included in",
    "is a type of",
    "is a",
    "is an",
    "is covered by",
    "is managed by",
    "depends on",
    "belongs to",
    "includes",
    "include",
    "costs",
    "requires",
    "provides",
    "supports",
    "offers",
    "covers",
    "contains",
    "uses",
    "has",
    "have",
    "replaces",
    "unlocks",
    "limits",
    "charges",
    "refunds",
]
_REL_RE = re.compile(
    r"^(?P<s>.+?)\s+(?P<r>"
    + "|".join(re.escape(r) for r in sorted(_RELATIONS, key=len, reverse=True))
    + r")\s+(?P<o>.+)$",
    re.IGNORECASE,
)
_LEADING = re.compile(r"^(?:the|a|an|our|each|every|all|any|your|its|their|some)\s+", re.I)
_OBJECT_SPLIT = re.compile(r"\s*,\s*(?:and\s+)?|\s+and\s+", re.I)
_MAX_WORDS = 7


def _clean_entity(raw: str) -> str:
    raw = raw.strip(" \t.,;:!?\"'()")
    raw = _LEADING.sub("", raw)
    raw = re.sub(r"\s+", " ", raw)
    if not raw or len(raw.split()) > _MAX_WORDS:
        return ""
    return raw


def _relation_name(raw: str) -> str:
    return re.sub(r"\s+", "_", raw.lower().strip())


class RuleBasedExtractor:
    """Pattern-based extractor for simple declarative sentences."""

    def extract(self, chunk: Chunk) -> list[Triple]:
        triples: list[Triple] = []
        for sentence in split_sentences(chunk.text):
            match = _REL_RE.match(sentence.rstrip(" ."))
            if not match:
                continue
            subject = _clean_entity(match["s"])
            relation = _relation_name(match["r"])
            if not subject:
                continue
            for part in _OBJECT_SPLIT.split(match["o"]):
                obj = _clean_entity(part)
                if obj and obj.lower() != subject.lower():
                    triples.append(Triple(subject, relation, obj, chunk.id))
        return triples


_SYSTEM = (
    "You extract knowledge-graph facts from text. Reply with ONLY a JSON array of "
    '[subject, relation, object] string triples, e.g. [["Home Loan", "requires", "credit score"]]. '
    "Use short noun phrases for entities, lower_snake_case verbs for relations, and resolve "
    "pronouns to the entity they refer to. Return [] if there are no facts."
)


class LLMExtractor:
    """Extractor that prompts an :class:`~kgvoicebot.llm.LLM` for JSON triples."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def extract(self, chunk: Chunk) -> list[Triple]:
        try:
            reply = self._llm.complete(_SYSTEM, chunk.text, max_tokens=1024)
        except Exception:
            log.exception("LLM extraction failed for chunk %s", chunk.id)
            return []
        return [Triple(s, r, o, chunk.id) for s, r, o in parse_triples(reply)]


def parse_triples(reply: str) -> list[tuple[str, str, str]]:
    """Leniently parse an LLM reply into triples, ignoring malformed entries."""
    start, end = reply.find("["), reply.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        log.warning("Could not parse triples from LLM reply")
        return []
    out: list[tuple[str, str, str]] = []
    if not isinstance(data, list):
        return out
    for item in data:
        if (
            isinstance(item, list)
            and len(item) == 3
            and all(isinstance(x, str) and x.strip() for x in item)
        ):
            s, r, o = (x.strip() for x in item)
            out.append((s, _relation_name(r), o))
    return out
