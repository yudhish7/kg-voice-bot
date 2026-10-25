"""Runtime configuration, read from ``KGV_*`` environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(".kgvoicebot")
    extractor: str = "auto"  # auto | rules | llm
    generator: str = "auto"  # auto | llm | extractive
    asr: str = "auto"  # auto | whisper | text
    tts: str = "auto"  # auto | espeak | stub
    llm_model: str = "claude-sonnet-5-5"
    whisper_model: str = "base"
    hops: int = 2
    top_k_chunks: int = 4
    max_facts: int = 15
    max_audio_bytes: int = 10 * 1024 * 1024
    api_key: str | None = None

    @property
    def graph_path(self) -> Path:
        return self.data_dir / "graph.json"

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        return cls(
            data_dir=Path(env("KGV_DATA_DIR", ".kgvoicebot")),
            extractor=env("KGV_EXTRACTOR", "auto"),
            generator=env("KGV_GENERATOR", "auto"),
            asr=env("KGV_ASR", "auto"),
            tts=env("KGV_TTS", "auto"),
            llm_model=env("KGV_LLM_MODEL", cls.llm_model),
            whisper_model=env("KGV_WHISPER_MODEL", cls.whisper_model),
            hops=_int("KGV_HOPS", cls.hops),
            top_k_chunks=_int("KGV_TOP_K", cls.top_k_chunks),
            max_facts=_int("KGV_MAX_FACTS", cls.max_facts),
            max_audio_bytes=_int("KGV_MAX_AUDIO_BYTES", cls.max_audio_bytes),
            api_key=env("KGV_API_KEY") or None,
        )
