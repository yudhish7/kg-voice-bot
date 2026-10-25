"""Text-to-speech backends. All return a complete WAV file as bytes."""

from __future__ import annotations

import io
import math
import shutil
import struct
import subprocess
import wave
from typing import Protocol

CONTENT_TYPE = "audio/wav"


class TTS(Protocol):
    content_type: str

    def synthesize(self, text: str) -> bytes: ...


class EspeakTTS:
    """Offline synthesis through the ``espeak-ng`` (or ``espeak``) command-line tool."""

    content_type = CONTENT_TYPE

    def __init__(self, words_per_minute: int = 165, timeout: float = 30.0) -> None:
        binary = shutil.which("espeak-ng") or shutil.which("espeak")
        if binary is None:
            raise RuntimeError("espeak-ng not found; install it (e.g. apt install espeak-ng)")
        self._binary = binary
        self._wpm = words_per_minute
        self._timeout = timeout

    def synthesize(self, text: str) -> bytes:
        result = subprocess.run(  # noqa: S603 - fixed binary, text passed via stdin
            [self._binary, "--stdout", "-s", str(self._wpm), "--stdin"],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=self._timeout,
            check=True,
        )
        return result.stdout


class ToneTTS:
    """Deterministic placeholder: a short tone whose length scales with the text.

    Lets the full pipeline run (and be tested) on machines without a speech engine.
    """

    content_type = CONTENT_TYPE

    def __init__(self, sample_rate: int = 16000) -> None:
        self._rate = sample_rate

    def synthesize(self, text: str) -> bytes:
        seconds = min(max(len(text) / 60, 0.2), 10.0)
        n = int(self._rate * seconds)
        frames = b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / self._rate)))
            for i in range(n)
        )
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self._rate)
            wav.writeframes(frames)
        return buf.getvalue()
