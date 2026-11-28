"""Telugu voices that stay clean enough to play.

The Hima pack's extra middle seed carried a noise floor about ten times louder
than the trained speakers, so it is not offered.

v2 Part 1 (2026-11-21): this file is UNCHANGED in behaviour — every function
below keeps its original signature and return value. What was added is a
registry layer on top: ``backend_for_voice()`` and ``list_installed_voices()``
first consult the core voice registry (``core/registry.py``), so a voice
pack plugged in as a folder + ``manifest.yaml`` with a non-Hima engine
(Piper, ONNX-VITS) works with no code change. Hima voices (Srinivas /
Sravani) take the original path exactly as before, including the shared
int8 pack, speaker names, seed 0, 24 kHz, and ``conversational`` only.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from telugu_speech.asr.infer import ModelUnavailable

log = logging.getLogger("vaani.tts.voices")

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

# v2: the registry is the source of truth for plugged-in voices; the
# dict above stays as the legacy fallback for the two shipped Hima voices.
DEFAULT_VOICE = "telugu_srinivas"


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
        return _registry_has(voice)
    allowed = selected_voices(root)
    return allowed is None or voice in allowed


def list_installed_voices(root: Path | None = None) -> list[dict[str, object]]:
    base = root or Path(".")
    allowed = selected_voices(base)
    ready = pack_ready(base)
    sample_root = base / SAMPLE_DIR
    voices: list[dict[str, object]] = []
    # v2: registry packs first (plugged-in voices, any engine).
    for pack in _registry_packs():
        if pack.id in VOICES:
            continue  # shipped Hima voices are listed below, unchanged
        voices.append(
            {
                "id": pack.id,
                "name": pack.name,
                "speaker": pack.name,
                "language": pack.language,
                "accent": "unlabeled",
                "tier": "registry",
                "gender": pack.gender,
                "styles": ["conversational"],
                "license": pack.licence,
                "backend": pack.engine,
                "model_path": str(pack.model_path),
                "model_version": pack.version,
                "size_bytes": pack.model_path.stat().st_size if pack.model_path.is_file() else None,
                "compatibility": [pack.engine],
                "available": pack.model_path.is_file() and pack.config_path.is_file(),
                "shared_weights": False,
                "download_id": None,
                "samples": [],
                "note": f"Voice pack registered from {pack.pack_dir}.",
                "paths": [str(pack.pack_dir)],
            }
        )
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
            yield self._say_present(engine, piece)

    def _say_present(self, engine: object, piece: str):
        """Speak one piece again when the take is silent or far too short.

        The Hima sampler sometimes stops after the first word. The same seed
        repeats that take, so a short take is spoken again with the next seed
        and the longer take is kept. A take that is already long enough stays
        on the original seed, so a good line does not change.
        """
        import numpy as np

        letters = sum(1 for char in piece if not char.isspace())
        rate = self.sample_rate_hz
        # About 90 ms per letter is a normal take. Much shorter means the
        # sampler stopped. A longer take is not better: it often says other words.
        expected = int(0.09 * max(letters, 1) * rate)
        floor = int(0.02 * max(letters, 1) * rate)
        best = np.zeros(0, np.float32)
        best_distance = 10**12
        for attempt in range(3):
            if self.speaker:
                audio = engine.say(piece, voice=self.speaker, seed=self.seed + attempt)  # type: ignore[attr-defined]
            else:
                audio = engine.say(piece, seed=self.seed + attempt)  # type: ignore[attr-defined]
            taken = np.asarray(audio, dtype=np.float32).reshape(-1)
            distance = abs(taken.size - expected) if taken.size >= floor else 10**12
            if distance < best_distance:
                best = taken
                best_distance = distance
            if attempt == 0 and floor <= taken.size <= int(expected * 1.6):
                return taken
        return best


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


# ---------------------------------------------------------------------------
# v2 Part 1 — registry layer (additive only; nothing above changed)
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _registry_packs() -> list:
    """Registered voice packs, or [] if the registry cannot be read.

    Never raises: a registry failure must not break the legacy voice list.
    """
    try:
        from telugu_speech.core.registry import get_registry

        return get_registry(_project_root()).list_voices()
    except Exception:
        log.exception("voice registry unavailable; using legacy voices only")
        return []


def _registry_has(voice_id: str) -> bool:
    return any(pack.id == voice_id for pack in _registry_packs())


class _LegacyFileAdapter:
    """Wrap a core provider backend in the legacy file-based interface.

    Legacy callers (api/main.py warmup, routes) call
    ``synthesize(text, reference_audio, reference_text, output_path) -> Path``.
    A plugged-in non-Hima voice speaks through the core engine pool instead,
    so this adapter writes the pool's audio bytes to ``output_path``.
    """

    def __init__(self, pack, pool) -> None:
        self.pack = pack
        self._pool = pool
        self.name = f"{pack.engine}:{pack.id}"
        self.requires_reference = False
        self.sample_rate_hz = pack.sample_rate

    def synthesize(self, text: str, reference_audio: Path | None, reference_text: str, output_path: Path) -> Path:
        del reference_audio, reference_text
        audio = self._pool.synthesize(text, self.pack.id, "conversational")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio)
        return output_path

    def iter_pcm(self, text: str):
        """Yield float32 audio per piece, like HimaTeluguBackend.iter_pcm.

        The pool streams WAV bytes per chunk; decode each chunk's frames
        back to float32 so Part 2's SSE path (which expects float32
        arrays, as Hima's iter_pcm yields) works unchanged.
        """
        import io
        import wave

        import numpy as np

        for chunk in self._pool.synthesize_stream(text, self.pack.id, "conversational"):
            with wave.open(io.BytesIO(chunk), "rb") as wav:
                frames = wav.readframes(wav.getnframes())
            pcm16 = np.frombuffer(frames, dtype=np.int16)
            yield (pcm16.astype(np.float32) / 32767.0)


def backend_for_voice(voice: str, root: Path | None = None):
    """Return a shipped-pack backend. telugu_default follows the installed pack.

    Unchanged for the two Hima voices (returns ``HimaTeluguBackend`` or
    ``None`` exactly as before). A registry pack with a non-Hima engine
    returns a ``_LegacyFileAdapter`` over the core engine pool instead.
    """
    base = root or Path(".")
    if voice == "telugu_default":
        voice = preferred_voice(base)
    # v2: plugged-in voice with a non-Hima engine.
    for pack in _registry_packs():
        if pack.id == voice and pack.engine != "hima-nano":
            from telugu_speech.core.engine import get_pool

            return _LegacyFileAdapter(pack, get_pool())
    allowed = selected_voices(base)
    if allowed is not None and voice not in allowed:
        return None
    spec = VOICES.get(voice)
    if spec is None:
        return None
    return HimaTeluguBackend(str(base / PACK), spec["speaker"], int(spec["seed"]))


def default_voice() -> str:
    """v2 helper: the default voice id (unchanged: telugu_srinivas)."""
    return DEFAULT_VOICE
