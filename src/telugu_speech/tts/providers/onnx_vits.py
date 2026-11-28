"""Generic ONNX VITS provider (engine: ``onnx-vits``) — any VITS ONNX model.

This is the escape hatch for VITS-family ONNX packs that are not Piper
and not hima-nano (e.g. Meta MMS-TTS conversions via sherpa-onnx, or a
future Telugu VITS voice). It loads the model with ONNX Runtime directly
and runs the standard VITS inference path.

Honest scope note: a VITS ONNX graph needs its phonemiser / text
front-end, which varies by model (espeak-ng, MMS tokeniser, per-model
symbol table in ``config.json``). This provider implements the common
path — read ``config.json`` for the symbol table and sample rate, run
the graph — and raises EngineNotLoadedError with the exact missing
piece if a pack's config does not describe its front-end. It does NOT
guess a phonemiser: a wrong phonemiser produces wrong Telugu, silently.
Wire the per-model front-end in ``_phonemise()`` once the pack's config
format is known; until then the provider refuses to synthesise rather
than producing garbage audio.
"""

from __future__ import annotations

import json
from typing import Iterator

from telugu_speech.core.errors import EngineNotLoadedError
from telugu_speech.core.registry import VoicePack

from .base import BaseTTSBackend


class OnnxVitsBackend(BaseTTSBackend):
    """Adapter for generic VITS ONNX voice packs."""

    engine_name = "onnx-vits"

    def __init__(self, pack: VoicePack) -> None:
        super().__init__(pack)
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise EngineNotLoadedError(
                f"voice '{pack.id}': onnx-vits engine needs onnxruntime: "
                f"pip install onnxruntime ({exc})"
            )
        try:
            self._config = json.loads(pack.config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise EngineNotLoadedError(
                f"voice '{pack.id}': cannot read config {pack.config_path}: {exc}"
            )
        from telugu_speech.core.config import settings

        options = ort.SessionOptions()
        options.intra_op_num_threads = settings.effective_threads()
        options.inter_op_num_threads = 1
        try:
            self._session = ort.InferenceSession(str(pack.model_path), options)
        except Exception as exc:
            raise EngineNotLoadedError(
                f"voice '{pack.id}': ONNX Runtime failed to load {pack.model_path}: {exc}"
            )

    def _phonemise(self, text: str) -> list[int]:
        """Map text -> input ids using the pack's config.

        Supported config shapes (checked in order):
          * ``{"symbols": [...]}`` — direct character/symbol table lookup
          * ``{"phoneme_map": {...}}`` — explicit text -> ids map
        Anything else raises EngineNotLoadedError naming the config file.
        """
        if isinstance(self._config.get("symbols"), list):
            symbols = list(self._config["symbols"])
            index = {s: i for i, s in enumerate(symbols)}
            ids = [index[ch] for ch in text if ch in index]
            if not ids:
                raise EngineNotLoadedError(
                    f"voice '{self.pack.id}': no characters of the input text "
                    f"are in the pack's symbol table ({self.pack.config_path})"
                )
            return ids
        if isinstance(self._config.get("phoneme_map"), dict):
            mapping = self._config["phoneme_map"]
            return [int(mapping[ch]) for ch in text if ch in mapping]
        raise EngineNotLoadedError(
            f"voice '{self.pack.id}': config {self.pack.config_path} does not "
            "describe a front-end (need 'symbols' list or 'phoneme_map'); "
            "refusing to synthesise rather than guessing a phonemiser"
        )

    def synthesize(self, text: str, voice: VoicePack,
                   style: str = "conversational") -> bytes:
        import io

        import numpy as np
        import wave

        input_ids = self._phonemise(text)
        inputs = {self._session.get_inputs()[0].name: np.array([input_ids], dtype=np.int64)}
        # VITS graphs commonly take (x, x_lengths, scales); provide the
        # minimal inputs the graph declares, in declared order.
        for inp in self._session.get_inputs()[1:]:
            if "length" in inp.name:
                inputs[inp.name] = np.array([len(input_ids)], dtype=np.int64)
            elif "scale" in inp.name:
                inputs[inp.name] = np.array([1.0], dtype=np.float32)
        outputs = self._session.run(None, inputs)
        audio = outputs[0]
        pcm = (np.asarray(audio).flatten() * 32767).astype(np.int16).tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(int(voice.sample_rate))
            wav.writeframes(pcm)
        return buf.getvalue()

    def synthesize_stream(self, text: str, voice: VoicePack,
                          style: str = "conversational") -> Iterator[bytes]:
        yield self.synthesize(text, voice, style)

    def close(self) -> None:
        self._session = None  # type: ignore[assignment]
        super().close()
