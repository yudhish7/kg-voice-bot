from __future__ import annotations

from kgvoicebot.extract import LLMExtractor, RuleBasedExtractor, parse_triples
from kgvoicebot.models import Chunk


def _chunk(text: str) -> Chunk:
    return Chunk("c:0", "c", text)


def test_rule_extractor_splits_list_objects_and_strips_determiners():
    triples = RuleBasedExtractor().extract(
        _chunk(
            "The Fiber 300 plan includes a free router, a static IP address and priority support."
        )
    )
    assert {(t.subject, t.relation, t.object) for t in triples} == {
        ("Fiber 300 plan", "includes", "free router"),
        ("Fiber 300 plan", "includes", "static IP address"),
        ("Fiber 300 plan", "includes", "priority support"),
    }
    assert all(t.chunk_id == "c:0" for t in triples)


def test_rule_extractor_prefers_longest_relation():
    (triple,) = RuleBasedExtractor().extract(_chunk("Basic is eligible for refunds."))
    assert triple.relation == "is_eligible_for"


def test_rule_extractor_ignores_unmatched_and_overlong_entities():
    text = "Nothing to see here. " + "word " * 12 + "includes a router."
    assert RuleBasedExtractor().extract(_chunk(text)) == []


def test_parse_triples_is_lenient_about_wrapping_and_bad_entries():
    reply = 'Sure!\n```json\n[["A", "Is Part Of", "B"], ["x", "y"], ["", "r", "o"], [1, 2, 3]]\n```'
    assert parse_triples(reply) == [("A", "is_part_of", "B")]


def test_parse_triples_handles_garbage():
    assert parse_triples("no json here") == []
    assert parse_triples("[not json]") == []
    assert parse_triples('{"a": 1}') == []


class _FakeLLM:
    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply

    def complete(self, system: str, prompt: str, max_tokens: int = 512) -> str:
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_llm_extractor_builds_triples_with_provenance():
    extractor = LLMExtractor(_FakeLLM('[["Fiber 300", "includes", "router"]]'))
    (triple,) = extractor.extract(_chunk("anything"))
    assert (triple.subject, triple.relation, triple.object, triple.chunk_id) == (
        "Fiber 300",
        "includes",
        "router",
        "c:0",
    )


def test_llm_extractor_survives_llm_failure():
    assert LLMExtractor(_FakeLLM(RuntimeError("boom"))).extract(_chunk("x")) == []
