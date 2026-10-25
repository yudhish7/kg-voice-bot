"""Automatic speech recognition backends."""

from __future__ import annotations

import os
import tempfile
from typing import Protocol


class ASR(Protocol):
    def transcribe(self, audio: bytes) -> str: ...


class FasterWhisperASR:
    """Local Whisper via ``faster-whisper`` (``pip install kgvoicebot[whisper]``).

    Accepts anything ffmpeg can decode (wav, mp3, m4a, webm, ogg...). The model is loaded
    lazily on first use so the service starts fast.
    """

    def __init__(self, model_size: str = "base") -> None:
        try:
            import faster_whisper  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "faster-whisper is required: pip install 'kgvoicebot[whisper]'"
            ) from exc
        self._model_size = model_size
        self._model = None

    def transcribe(self, audio: bytes) -> str:  # pragma: no cover - needs model download
        from faster_whisper import WhisperModel

        if self._model is None:
            self._model = WhisperModel(self._model_size, compute_type="int8")
        fd, path = tempfile.mkstemp(suffix=".audio")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(audio)
            segments, _ = self._model.transcribe(path, vad_filter=True)
            return " ".join(seg.text.strip() for seg in segments).strip()
        finally:
            os.unlink(path)


class TextPassthroughASR:
    """Treats the 'audio' bytes as UTF-8 text. For tests and demos only."""

    def transcribe(self, audio: bytes) -> str:
        return audio.decode("utf-8", errors="replace").strip()
