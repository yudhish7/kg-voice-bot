"""Small text utilities: normalisation, tokenisation, sentence splitting."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from how i if in is it its my of on or
    our so than that the their then there these this to us was we what when where which
    who whom why will with you your""".split()  # noqa: SIM905 - keeps the list readable
)


def normalize(text: str) -> str:
    """Lower-case, collapse whitespace and trim surrounding punctuation."""
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return text.strip(" .,;:!?\"'()[]")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def stem(token: str) -> str:
    """Very light plural/3rd-person stemming so 'includes' matches 'include'."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(text: str) -> list[str]:
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS]


_HEADING = re.compile(r"^\s*#{1,6}\s+")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


def _units(text: str) -> list[str]:
    """Break text into paragraph-like units; headings and bullets stand alone."""
    units: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            units.append(" ".join(buffer))
            buffer.clear()

    for raw in text.splitlines():
        line = raw.replace("**", "").replace("__", "").replace("`", "").strip()
        if not line:
            flush()
        elif _HEADING.match(line) or _BULLET.match(line):
            flush()
            units.append(_BULLET.sub("", _HEADING.sub("", line)))
        else:
            buffer.append(line)
    flush()
    return units


def split_sentences(text: str) -> list[str]:
    """Split into sentences; markdown headings/bullets and paragraph ends count as boundaries."""
    sentences: list[str] = []
    for unit in _units(text):
        unit = re.sub(r"\s+", " ", unit).strip()
        if not unit:
            continue
        if unit[-1] not in ".!?":
            unit += "."
        sentences.extend(s.strip() for s in _SENTENCE_END.split(unit) if s.strip())
    return sentences
