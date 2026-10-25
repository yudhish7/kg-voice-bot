"""Wire concrete components together according to :class:`Settings`."""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil

from .audio.asr import ASR, FasterWhisperASR, TextPassthroughASR
from .audio.tts import TTS, EspeakTTS, ToneTTS
from .bot import VoiceBot
from .config import Settings
from .extract import LLMExtractor, RuleBasedExtractor, TripleExtractor
from .generator import ExtractiveGenerator, Generator, LLMGenerator
from .graph import KnowledgeGraph
from .ingest import Ingestor
from .llm import LLM, AnthropicLLM
from .retriever import GraphRetriever

log = logging.getLogger(__name__)


def _llm_available() -> bool:
    return (
        bool(os.environ.get("ANTHROPIC_API_KEY"))
        and importlib.util.find_spec("anthropic") is not None
    )


def build_llm(settings: Settings) -> LLM:
    return AnthropicLLM(settings.llm_model)


def build_extractor(settings: Settings) -> TripleExtractor:
    use_llm = settings.extractor == "llm" or (settings.extractor == "auto" and _llm_available())
    if settings.extractor not in {"auto", "rules", "llm"}:
        raise ValueError(f"Unknown extractor: {settings.extractor!r}")
    return LLMExtractor(build_llm(settings)) if use_llm else RuleBasedExtractor()


def build_generator(settings: Settings) -> Generator:
    if settings.generator not in {"auto", "llm", "extractive"}:
        raise ValueError(f"Unknown generator: {settings.generator!r}")
    use_llm = settings.generator == "llm" or (settings.generator == "auto" and _llm_available())
    return LLMGenerator(build_llm(settings)) if use_llm else ExtractiveGenerator()


def build_asr(settings: Settings) -> ASR:
    if settings.asr == "text":
        return TextPassthroughASR()
    if settings.asr in {"auto", "whisper"}:
        return FasterWhisperASR(settings.whisper_model)
    raise ValueError(f"Unknown asr backend: {settings.asr!r}")


def build_tts(settings: Settings) -> TTS:
    if settings.tts == "stub":
        return ToneTTS()
    if (
        settings.tts == "auto"
        and shutil.which("espeak-ng") is None
        and shutil.which("espeak") is None
    ):
        log.warning("No speech engine found; falling back to placeholder tones (KGV_TTS=stub)")
        return ToneTTS()
    if settings.tts in {"auto", "espeak"}:
        return EspeakTTS()
    raise ValueError(f"Unknown tts backend: {settings.tts!r}")


def load_graph(settings: Settings) -> KnowledgeGraph:
    path = settings.graph_path
    return KnowledgeGraph.load(path) if path.exists() else KnowledgeGraph()


def build_retriever(kg: KnowledgeGraph, settings: Settings) -> GraphRetriever:
    return GraphRetriever(
        kg, hops=settings.hops, top_k_chunks=settings.top_k_chunks, max_facts=settings.max_facts
    )


def build_ingestor(kg: KnowledgeGraph, settings: Settings) -> Ingestor:
    return Ingestor(kg, build_extractor(settings))


def build_bot(kg: KnowledgeGraph, settings: Settings, with_audio: bool = True) -> VoiceBot:
    asr: ASR = build_asr(settings) if with_audio else TextPassthroughASR()
    tts: TTS = build_tts(settings) if with_audio else ToneTTS()
    return VoiceBot(build_retriever(kg, settings), build_generator(settings), asr, tts)
