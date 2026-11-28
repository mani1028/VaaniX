"""TTS provider base — the plugin interface every voice engine implements.

A provider is constructed with ONE validated ``VoicePack`` and owns the
loaded model for that voice for its whole lifetime. It is created lazily
by ``core.engine`` on first use of the voice, and ``close()`` must free
the model (ONNX session, file handles) so idle unload actually frees RAM.

Contract:
  * ``synthesize(text, voice, style="conversational") -> bytes``
    Full audio for one (already chunked) piece of text. WAV bytes.
  * ``synthesize_stream(text, voice, style="conversational") -> Iterator[bytes]``
    Yields audio chunks as produced. Default implementation calls
    ``synthesize`` once and yields it — providers with true streaming
    override this.
  * ``close()`` — idempotent, never raises for a half-initialised backend.

Providers must NOT import each other, must NOT read the registry, and
must NOT import FastAPI. They may import ``telugu_speech.tts.infer`` /
``telugu_speech.tts.runtime`` / ``telugu_speech.tts.model`` — the legacy
backends this package adapts — lazily, inside ``__init__``, so importing
the provider package loads no model.
"""

from __future__ import annotations

from typing import Iterator

from telugu_speech.core.errors import EngineNotLoadedError
from telugu_speech.core.registry import VoicePack


class BaseTTSBackend:
    """Base class for TTS providers. Subclass + implement ``synthesize``."""

    engine_name = "base"

    def __init__(self, pack: VoicePack) -> None:
        self.pack = pack
        self._closed = False

    # -- the contract ------------------------------------------------------
    def synthesize(self, text: str, voice: VoicePack,
                   style: str = "conversational") -> bytes:
        raise NotImplementedError

    def synthesize_stream(self, text: str, voice: VoicePack,
                          style: str = "conversational") -> Iterator[bytes]:
        yield self.synthesize(text, voice, style)

    def close(self) -> None:
        self._closed = True

    # -- helpers for adapters ----------------------------------------------
    @staticmethod
    def _require_legacy(module_name: str, *names: str):
        """Import a legacy module and return the first named attribute found.

        Raises EngineNotLoadedError naming exactly what was tried — a
        missing legacy entry point must be a loud, diagnosable error,
        never a silent fallback to another voice.
        """
        import importlib

        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise EngineNotLoadedError(
                f"legacy backend module '{module_name}' is not importable: {exc}"
            )
        for name in names:
            attr = getattr(module, name, None)
            if attr is not None:
                return attr
        raise EngineNotLoadedError(
            f"legacy backend module '{module_name}' exposes none of "
            f"{', '.join(names)} — the adapter cannot delegate to it"
        )
