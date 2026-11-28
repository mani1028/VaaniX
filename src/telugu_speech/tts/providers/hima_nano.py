"""Hima Nano TTS provider (engine: ``hima-nano``) — the shipped default.

Wraps the project's own ``HimaTeluguBackend`` (``tts/voices.py``) —
the lite pack ``lite/models/hima-tts-nano-te`` (Prahaladha
hima-tts-nano-te, 62M params, ONNX int8, licence: other, 24 kHz).
Both shipped voices share this ONE int8 pack; the speaker is selected
by name ("Srinivas" / "Sravani", seed 0), not by a separate model file.
Do NOT re-implement the model here, and do NOT delete or replace
``tts/voices.py`` — this provider is a thin plugin adapter over it.

Real backend API (read from ``tts/voices.py`` 2026-11-21):
  * ``HimaTeluguBackend(checkpoint, speaker, seed)``
    — checkpoint is the pack dir (``lite/models/hima-tts-nano-te``).
  * ``synthesize(text, reference_audio, reference_text, output_path) -> Path``
    — writes a WAV (via ``tts/chunks.py``: 180-char pieces, 0.12 s gaps,
    peak normalised to 0.9) and returns the path.
  * ``iter_pcm(text)`` — yields float32 numpy arrays per piece while the
    engine stays loaded (via ``tts/runtime.py::borrow_engine``).

This adapter therefore:
  * ``synthesize()`` calls the real ``synthesize()`` into a temp file
    and returns its bytes (WAV, 24 kHz).
  * ``synthesize_stream()`` calls ``iter_pcm()`` and yields each piece
    as 16-bit PCM WAV bytes, so a streaming client can play it as it
    arrives. The stream is float32 -> int16 converted here, with the
    same 0.9 peak normalisation ``write_chunked_speech`` applies.
"""

from __future__ import annotations

import io
import tempfile
import wave
from pathlib import Path
from typing import Iterator

import numpy as np

from telugu_speech.core.errors import EngineNotLoadedError
from telugu_speech.core.registry import VoicePack

from .base import BaseTTSBackend


class HimaNanoBackend(BaseTTSBackend):
    """Adapter for the hima-nano lite TTS engine (ENGINE_ID hima-onnx-int8)."""

    engine_name = "hima-nano"
    sample_rate_hz = 24000

    def __init__(self, pack: VoicePack) -> None:
        super().__init__(pack)
        # Speaker name is the pack's display name: "Srinivas" / "Sravani".
        # The legacy backend takes (checkpoint_dir, speaker, seed=0), where
        # checkpoint_dir is the PACK ROOT (lite/models/hima-tts-nano-te),
        # not the voice folder: model_path is
        #   <pack root>/onnx/voices/step_int8.onnx
        # so the root is three levels up from the model file. If a future
        # pack lays its files out differently, fall back to the pack dir.
        model = pack.model_path.resolve()
        checkpoint = (
            model.parents[2]
            if len(model.parents) > 2 and (model.parents[2] / "voices" / "voices.json").is_file()
            else pack.pack_dir
        )
        from telugu_speech.tts.voices import HimaTeluguBackend

        try:
            self._impl = HimaTeluguBackend(str(checkpoint), pack.name, 0)
        except Exception as exc:
            raise EngineNotLoadedError(
                f"hima-nano backend for voice '{pack.id}' failed to initialise: {exc}"
            )

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _float32_to_wav(pcm: np.ndarray, sample_rate: int) -> bytes:
        """One float32 array -> one WAV byte string (16-bit PCM, peak 0.9)."""
        audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-4:
            audio = audio * (0.9 / peak)
        pcm16 = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm16 * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm16.tobytes())
        return buf.getvalue()

    # -- the contract --------------------------------------------------------
    def synthesize(self, text: str, voice: VoicePack,
                   style: str = "conversational") -> bytes:
        # The real backend writes a WAV file; read it back as bytes.
        # (Temp file, not the API output dir — the pool owns caching.)
        with tempfile.TemporaryDirectory(prefix="vaani-tts-") as directory:
            out = Path(directory) / "out.wav"
            try:
                self._impl.synthesize(text, None, "", out)
            except Exception as exc:
                raise EngineNotLoadedError(
                    f"hima-nano backend for voice '{voice.id}' failed to synthesise: {exc}"
                )
            return out.read_bytes()

    def synthesize_stream(self, text: str, voice: VoicePack,
                          style: str = "conversational") -> Iterator[bytes]:
        try:
            for piece in self._impl.iter_pcm(text):
                yield self._float32_to_wav(piece, self.sample_rate_hz)
        except Exception as exc:
            raise EngineNotLoadedError(
                f"hima-nano backend for voice '{voice.id}' failed to stream: {exc}"
            )

    def close(self) -> None:
        # HimaTeluguBackend holds no explicit close; the engine itself is
        # owned by tts/runtime.py's pool (borrow_engine / shared_engine)
        # and is freed by idle unload there. Dropping our reference is enough.
        self._impl = None  # type: ignore[assignment]
        super().close()
