from __future__ import annotations

import io
import wave

import pytest

from kgvoicebot.audio.asr import TextPassthroughASR
from kgvoicebot.audio.tts import ToneTTS
from kgvoicebot.bm25 import BM25
from kgvoicebot.bot import NOT_HEARD, VoiceBot
from kgvoicebot.extract import RuleBasedExtractor
from kgvoicebot.generator import FALLBACK_ANSWER, ExtractiveGenerator, LLMGenerator, build_prompt
from kgvoicebot.graph import KnowledgeGraph
from kgvoicebot.ingest import Ingestor
from kgvoicebot.models import Evidence
from kgvoicebot.retriever import GraphRetriever


def test_bm25_ranks_matching_document_first_and_skips_zero_scores():
    index = BM25({"a": "router wifi speed", "b": "refund policy", "c": "router router router"})
    scores = index.scores("router speed")
    assert "b" not in scores
    assert max(scores, key=scores.get) in {"a", "c"}
    assert BM25({}).scores("anything") == {}


def test_retriever_links_entities_and_expands_multi_hop(retriever: GraphRetriever):
    ev = retriever.retrieve("What is the ATM limit on the card that comes with Premium Savings?")
    assert "Premium Savings account" in ev.entities
    facts = {(t.subject, t.relation, t.object) for t in ev.triples}
    # Hop 1: what the account includes. Hop 2: that card's ATM limit.
    assert ("Premium Savings account", "includes", "Platinum debit card") in facts
    assert ("Platinum debit card", "has", "daily ATM limit of 100000 rupees") in facts


def test_one_hop_limit_excludes_second_hop_facts(kg: KnowledgeGraph):
    ev = GraphRetriever(kg, hops=1).retrieve("What does the Premium Savings account include?")
    facts = {(t.subject, t.relation, t.object) for t in ev.triples}
    assert ("Platinum debit card", "has", "daily ATM limit of 100000 rupees") not in facts


def test_retriever_falls_back_to_bm25_without_entities(retriever: GraphRetriever):
    ev = retriever.retrieve("block lost card")
    assert ev.entities == []
    assert ev.chunks and "block a lost card" in ev.chunks[0][0].text


def test_retriever_returns_empty_evidence_for_unrelated_question(retriever: GraphRetriever):
    assert retriever.retrieve("Who won the cricket world cup?").empty


def test_retriever_sees_documents_ingested_after_construction(kg: KnowledgeGraph):
    retriever = GraphRetriever(kg)
    assert retriever.retrieve("Do you sell smart speakers?").empty
    Ingestor(kg, RuleBasedExtractor()).ingest_text("new", "The Gold plan includes a smart speaker.")
    assert not retriever.retrieve("Do you sell smart speakers?").empty


def test_retriever_validates_graph_weight(kg: KnowledgeGraph):
    with pytest.raises(ValueError):
        GraphRetriever(kg, graph_weight=1.5)


def test_reingesting_a_document_replaces_it(kg: KnowledgeGraph):
    ingestor = Ingestor(kg, RuleBasedExtractor())
    ingestor.ingest_text("promo", "The Silver plan costs 10 dollars.")
    ingestor.ingest_text("promo", "The Silver plan costs 12 dollars.")
    texts = [c.text for c in kg.chunks.values() if c.doc_id == "promo"]
    assert texts == ["The Silver plan costs 12 dollars."]
    assert not any(t.object == "10 dollars" for t in kg.triples)


def test_ingest_rejects_blank_doc_id(kg: KnowledgeGraph):
    with pytest.raises(ValueError):
        Ingestor(kg, RuleBasedExtractor()).ingest_text("  ", "text.")


def test_ingest_path_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        Ingestor(KnowledgeGraph(), RuleBasedExtractor()).ingest_path(tmp_path / "nope")


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the minimum balance for the Premium Savings account?", "25000 rupees"),
        ("What credit score does the Home Loan need?", "700"),
        ("How do I block a lost card?", "mobile app"),
    ],
)
def test_bot_answers_from_the_knowledge_base(bot: VoiceBot, question: str, expected: str):
    answer = bot.ask_text(question)
    assert expected in answer.text
    assert answer.sources


def test_bot_refuses_when_nothing_relevant_and_never_calls_the_model(retriever: GraphRetriever):
    class Exploding:
        def generate(self, question, evidence):
            raise AssertionError("generator must not run without evidence")

    bot = VoiceBot(retriever, Exploding(), TextPassthroughASR(), ToneTTS())
    assert bot.ask_text("Who won the cricket world cup?").text == FALLBACK_ANSWER


def test_bot_asks_to_repeat_on_empty_input(bot: VoiceBot):
    assert bot.ask_text("   ").text == NOT_HEARD


def test_voice_round_trip_returns_transcript_and_valid_wav(bot: VoiceBot):
    reply = bot.ask_voice(b"What credit score does the Home Loan need?")
    assert reply.transcript == "What credit score does the Home Loan need?"
    assert "700" in reply.answer.text
    assert reply.content_type == "audio/wav"
    with wave.open(io.BytesIO(reply.audio)) as wav:
        assert wav.getnframes() > 0 and wav.getframerate() == 16000


def test_llm_generator_sends_facts_and_passages_in_prompt(retriever: GraphRetriever):
    seen: dict[str, str] = {}

    class FakeLLM:
        def complete(self, system: str, prompt: str, max_tokens: int = 512) -> str:
            seen["system"], seen["prompt"] = system, prompt
            return "The Home Loan needs a credit score of 700."

    question = "What credit score does the Home Loan need?"
    ev = retriever.retrieve(question)
    assert LLMGenerator(FakeLLM()).generate(question, ev).startswith("The Home Loan")
    assert "ONLY" in seen["system"]
    assert "Passages:" in seen["prompt"] and question in seen["prompt"]
    assert "(none)" in build_prompt("q", Evidence())


def test_extractive_generator_fallback_when_no_overlap():
    assert ExtractiveGenerator().generate("zebra", Evidence()) == FALLBACK_ANSWER
