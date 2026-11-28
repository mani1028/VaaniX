"""Piper TTS provider (engine: ``piper``) — ONNX VITS, CPU-only.

Piper is the recommended low-resource upgrade path from the Part 1
research: a Piper Telugu voice (``te_IN-padmavathi-medium``, ~62 MB)
runs in real time on a Pi 4 / 1-2 GB RAM with no GPU. This provider
adapts a Piper voice pack to the same plugin interface as hima-nano,
so a Piper pack can be plugged in as a folder + manifest and used with
zero core or API changes.

Implementation: delegates to the ``piper`` Python package when it is
installed (``pip install piper-tts``), constructing
``piper.PiperVoice.load(model_path, config_path=...)``. If the package
is not installed, construction fails with a clear EngineNotLoadedError
telling the operator exactly what to install — it never silently falls
back to another voice.

Licence caution (from the research, UNVERIFIED for the specific voice):
each Piper voice carries its own licence; check the voice card in the
manifest's ``licence_url`` before any commercial use. The manifest's
``licence`` field is the operator's attestation, not the core's.
"""

from __future__ import annotations

from typing import Iterator

from telugu_speech.core.errors import EngineNotLoadedError
from telugu_speech.core.registry import VoicePack

from .base import BaseTTSBackend


class PiperBackend(BaseTTSBackend):
    """Adapter for Piper (ONNX VITS) voice packs."""

    engine_name = "piper"

    def __init__(self, pack: VoicePack) -> None:
        super().__init__(pack)
        try:
            from piper import PiperVoice  # type: ignore
        except ImportError as exc:
            raise EngineNotLoadedError(
                f"voice '{pack.id}': piper engine needs the 'piper-tts' package: "
                f"pip install piper-tts ({exc})"
            )
        try:
            self._voice = PiperVoice.load(
                str(pack.model_path),
                config_path=str(pack.config_path),
            )
        except Exception as exc:
            raise EngineNotLoadedError(
                f"voice '{pack.id}': Piper failed to load {pack.model_path}: {exc}"
            )

    def synthesize(self, text: str, voice: VoicePack,
                   style: str = "conversational") -> bytes:
        # Piper's synthesize() yields AudioChunk objects; collect them
        # into one WAV byte string via the first chunk's sample rate.
        chunks = list(self._voice.synthesize(text))
        if not chunks:
            return b""
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(chunks[0].sample_rate)
            for chunk in chunks:
                wav.writeframes(chunk.audio_int16_bytes)
        return buf.getvalue()

    def synthesize_stream(self, text: str, voice: VoicePack,
                          style: str = "conversational") -> Iterator[bytes]:
        # Stream per Piper chunk: each chunk is re-wrapped as a tiny WAV
        # so a streaming client can play it as it arrives. (A future
        # optimisation is raw PCM streaming with a single WAV header.)
        import io
        import wave

        for chunk in self._voice.synthesize(text):
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(chunk.sample_rate)
                wav.writeframes(chunk.audio_int16_bytes)
            yield buf.getvalue()

    def close(self) -> None:
        # PiperVoice holds no explicit close in the public API; dropping
        # the reference frees the ONNX session.
        self._voice = None  # type: ignore[assignment]
        super().close()
