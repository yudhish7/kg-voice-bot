"""Thin LLM abstraction so the rest of the code never imports a vendor SDK."""

from __future__ import annotations

from typing import Protocol


class LLM(Protocol):
    def complete(self, system: str, prompt: str, max_tokens: int = 512) -> str: ...


class AnthropicLLM:
    """Claude via the official ``anthropic`` SDK (``pip install kgvoicebot[llm]``).

    Reads ``ANTHROPIC_API_KEY`` from the environment.
    """

    def __init__(self, model: str) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "The 'anthropic' package is required: pip install 'kgvoicebot[llm]'"
            ) from exc
        self._client = anthropic.Anthropic()
        self._model = model

    def complete(self, system: str, prompt: str, max_tokens: int = 512) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in message.content if block.type == "text").strip()
