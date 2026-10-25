from __future__ import annotations

from pathlib import Path

import pytest

from kgvoicebot.audio.asr import TextPassthroughASR
from kgvoicebot.audio.tts import ToneTTS
from kgvoicebot.bot import VoiceBot
from kgvoicebot.extract import RuleBasedExtractor
from kgvoicebot.generator import ExtractiveGenerator
from kgvoicebot.graph import KnowledgeGraph
from kgvoicebot.ingest import Ingestor
from kgvoicebot.retriever import GraphRetriever

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


@pytest.fixture
def kg() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    Ingestor(graph, RuleBasedExtractor()).ingest_path(SAMPLE_DIR)
    return graph


@pytest.fixture
def retriever(kg: KnowledgeGraph) -> GraphRetriever:
    return GraphRetriever(kg)


@pytest.fixture
def bot(retriever: GraphRetriever) -> VoiceBot:
    return VoiceBot(retriever, ExtractiveGenerator(), TextPassthroughASR(), ToneTTS())
