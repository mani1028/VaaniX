"""Telugu voices that stay clean enough to play.

The Hima pack's extra middle seed carried a noise floor about ten times louder
than the trained speakers, so it is not offered.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from telugu_speech.asr.infer import ModelUnavailable

PACK = Path("lite/models/hima-tts-nano-te")
SAMPLE_DIR = Path("lite/voices/samples")
DOWNLOAD_ID = "hima-tts-nano-te"
ENGINE_ID = "hima-onnx-int8"
VOICES = {
    "telugu_srinivas": {
        "speaker": "Srinivas",
        "seed": 0,
        "gender": "male",
        "sample": "srinivas.wav",
        "note": "Deep trained male, about 100 to 115 Hz. Karthik and Pranay match this pitch.",
    },
    "telugu_sravani": {
        "speaker": "Sravani",
        "seed": 0,
        "gender": "female",
        "sample": "sravani.wav",
        "note": "Trained female, about 185 to 225 Hz. Bhavana and Lakshmi match this pitch.",
    },
}


def cpu_threads() -> int:
    raw = os.environ.get("TELUGU_CPU_THREADS", "2")
    try:
        threads = int(raw)
    except ValueError:
        return 2
    return min(4, max(1, threads))


def _pack_bytes(base: Path) -> int | None:
    path = base / PACK
    if not path.is_dir():
        return None
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def pack_ready(root: Path | None = None) -> bool:
    path = (root or Path(".")) / PACK
    return (path / "onnx" / "voices" / "step_int8.onnx").is_file() and (path / "voices" / "voices.json").is_file()


SUPPORTED_STYLES = frozenset({"conversational"})


def sample_names() -> set[str]:
    return {spec["sample"] for spec in VOICES.values()}


def _marker(root: Path) -> dict[str, object]:
    path = root / "voice_pack.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def selected_voices(root: Path | None = None) -> frozenset[str] | None:
    """Voices a downloaded folder may speak. None means every installed voice."""
    payload = _marker(root or Path("."))
    listed = payload.get("voices")
    if isinstance(listed, list):
        chosen = [item for item in listed if isinstance(item, str) and item in VOICES]
        if chosen:
            return frozenset(chosen)
    voice = str(payload.get("voice", ""))
    if voice in VOICES:
        return frozenset(VOICES)
    return None


def is_known_voice(voice: str, root: Path | None = None) -> bool:
    if voice not in VOICES:
        return False
    allowed = selected_voices(root)
    return allowed is None or voice in allowed


def list_installed_voices(root: Path | None = None) -> list[dict[str, object]]:
    base = root or Path(".")
    allowed = selected_voices(base)
    ready = pack_ready(base)
    sample_root = base / SAMPLE_DIR
    voices: list[dict[str, object]] = []
    for voice_id, spec in VOICES.items():
        if allowed is not None and voice_id not in allowed:
            continue
        sample = spec["sample"]
        voices.append(
            {
                "id": voice_id,
                "name": spec["speaker"],
                "speaker": spec["speaker"] or voice_id.removeprefix("telugu_"),
                "language": "te",
                "accent": "unlabeled",
                "tier": "lite",
                "gender": spec["gender"],
                "styles": ["conversational"],
                "license": "other (see lite/models/hima-tts-nano-te/README.md)",
                "backend": ENGINE_ID,
                "model_path": str(base / PACK),
                "model_version": "int8",
                "size_bytes": _pack_bytes(base),
                "compatibility": [ENGINE_ID],
                "available": ready,
                "shared_weights": True,
                "download_id": DOWNLOAD_ID,
                "samples": [sample] if (sample_root / sample).is_file() else [],
                "note": spec["note"] + " Both default voices share this one int8 pack. Not a Grok or Sarvam voice.",
                "paths": [str(base / PACK)],
            }
        )
    return voices


class HimaTeluguBackend:
    """One shipped Telugu voice, either a named speaker or a stable seed."""

    def __init__(self, checkpoint: str, speaker: str | None, seed: int) -> None:
        self.checkpoint = checkpoint
        self.speaker = speaker
        self.seed = seed
        label = speaker if speaker else f"seed{seed}"
        self.name = f"hima-onnx-int8:{label}"
        self.requires_reference = False
        self.sample_rate_hz = 24000
        self._engine = None

    def _load(self):
        root = Path(self.checkpoint)
        if not (root / "onnx" / "voices" / "step_int8.onnx").is_file():
            raise ModelUnavailable("The Hima Telugu voice pack is not in lite/models/hima-tts-nano-te.")
        from telugu_speech.tts.runtime import shared_engine

        def build() -> object:
            import sys

            sys.path.insert(0, str(root))
            try:
                from hima_onnx import OnnxTTS
            except Exception as exc:
                raise ModelUnavailable(
                    "The voice pack needs onnxruntime, numpy, safetensors, sentencepiece, and pyyaml."
                ) from exc
            return OnnxTTS(str(root), precision="int8", threads=cpu_threads())

        self._engine = shared_engine(("hima-int8", str(root.resolve()), cpu_threads()), build)
        return self._engine

    def synthesize(self, text: str, reference_audio: Path | None, reference_text: str, output_path: Path) -> Path:
        del reference_audio, reference_text
        root = Path(self.checkpoint)
        if not (root / "onnx" / "voices" / "step_int8.onnx").is_file():
            raise ModelUnavailable("The Hima Telugu voice pack is not in lite/models/hima-tts-nano-te.")
        from telugu_speech.tts.chunks import write_chunked_speech
        from telugu_speech.tts.runtime import borrow_engine

        def build() -> object:
            import sys

            sys.path.insert(0, str(root))
            try:
                from hima_onnx import OnnxTTS
            except Exception as exc:
                raise ModelUnavailable(
                    "The voice pack needs onnxruntime, numpy, safetensors, sentencepiece, and pyyaml."
                ) from exc
            return OnnxTTS(str(root), precision="int8", threads=cpu_threads())

        with borrow_engine(("hima-int8", str(root.resolve()), cpu_threads()), build) as engine:
            chunks = list(self._speak(engine, text))
        return write_chunked_speech(chunks, self.sample_rate_hz, output_path)

    def iter_pcm(self, text: str):
        """Yield float32 audio for each piece while the engine stays loaded."""
        root = Path(self.checkpoint)
        if not (root / "onnx" / "voices" / "step_int8.onnx").is_file():
            raise ModelUnavailable("The Hima Telugu voice pack is not in lite/models/hima-tts-nano-te.")
        from telugu_speech.tts.runtime import borrow_engine

        def build() -> object:
            import sys

            sys.path.insert(0, str(root))
            try:
                from hima_onnx import OnnxTTS
            except Exception as exc:
                raise ModelUnavailable(
                    "The voice pack needs onnxruntime, numpy, safetensors, sentencepiece, and pyyaml."
                ) from exc
            return OnnxTTS(str(root), precision="int8", threads=cpu_threads())

        with borrow_engine(("hima-int8", str(root.resolve()), cpu_threads()), build) as engine:
            yield from self._speak(engine, text)

    def _speak(self, engine: object, text: str):
        from telugu_speech.tts.chunks import speech_pieces

        for piece in speech_pieces(text):
            if self.speaker:
                yield engine.say(piece, voice=self.speaker, seed=self.seed)  # type: ignore[attr-defined]
            else:
                yield engine.say(piece, seed=self.seed)  # type: ignore[attr-defined]


def preferred_voice(root: Path | None = None) -> str:
    """Voice a download should speak first. The full tree keeps Srinivas."""
    base = root or Path(".")
    payload = _marker(base)
    allowed = selected_voices(base)
    voice = str(payload.get("voice", ""))
    listed = payload.get("voices")
    if isinstance(listed, list) and allowed is not None:
        chosen = [item for item in listed if isinstance(item, str) and item in allowed]
        if voice in chosen:
            return voice
        if chosen:
            return chosen[0]
    if allowed is not None and voice in allowed:
        return voice
    return "telugu_srinivas"


def backend_for_voice(voice: str, root: Path | None = None):
    """Return a shipped-pack backend. telugu_default follows the installed pack."""
    base = root or Path(".")
    if voice == "telugu_default":
        voice = preferred_voice(base)
    allowed = selected_voices(base)
    if allowed is not None and voice not in allowed:
        return None
    spec = VOICES.get(voice)
    if spec is None:
        return None
    return HimaTeluguBackend(str(base / PACK), spec["speaker"], int(spec["seed"]))
