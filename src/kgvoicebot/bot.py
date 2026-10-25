"""The voice bot: audio -> transcript -> retrieval -> answer -> audio."""

from __future__ import annotations

import logging

from .audio.asr import ASR
from .audio.tts import TTS
from .generator import FALLBACK_ANSWER, Generator
from .models import Answer, Evidence, VoiceReply
from .retriever import GraphRetriever

log = logging.getLogger(__name__)

NOT_HEARD = "Sorry, I didn't catch that. Could you say it again?"


class VoiceBot:
    def __init__(
        self,
        retriever: GraphRetriever,
        generator: Generator,
        asr: ASR,
        tts: TTS,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.asr = asr
        self.tts = tts

    def ask_text(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            return Answer(NOT_HEARD, Evidence())
        evidence = self.retriever.retrieve(question)
        if evidence.empty:
            # Nothing relevant in the knowledge base: don't let a model improvise.
            return Answer(FALLBACK_ANSWER, evidence)
        return Answer(self.generator.generate(question, evidence), evidence)

    def ask_voice(self, audio: bytes) -> VoiceReply:
        transcript = self.asr.transcribe(audio)
        log.info("Transcript: %r", transcript)
        answer = self.ask_text(transcript)
        return VoiceReply(
            transcript=transcript,
            answer=answer,
            audio=self.tts.synthesize(answer.text),
            content_type=self.tts.content_type,
        )
