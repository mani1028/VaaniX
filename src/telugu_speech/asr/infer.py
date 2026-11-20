"""ASR backends. Heavy libraries are imported only when a backend is constructed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Protocol


class AsrBackend(Protocol):
    name: str

    def transcribe(self, audio_path: Path) -> str:
        """Return a transcript for one audio file."""


class ModelUnavailable(RuntimeError):
    """A requested checkpoint or library is not installed."""


@dataclass
class CallableBackend:
    name: str
    function: object

    def transcribe(self, audio_path: Path) -> str:
        return str(self.function(audio_path))  # type: ignore[operator]


class IndicConformerBackend:
    def __init__(self, checkpoint: str, decoding: str) -> None:
        if decoding not in {"ctc", "rnnt"}:
            raise ValueError("decoding must be ctc or rnnt")
        self.name = f"indicconformer:{decoding}"
        self.checkpoint = checkpoint
        self.decoding = decoding
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from nemo.collections.asr.models import ASRModel
        except Exception as exc:
            raise ModelUnavailable(
                "IndicConformer requires AI4Bharat NeMo (nemo-v2) and a gated Hugging Face download."
            ) from exc
        model = ASRModel.from_pretrained(self.checkpoint)
        model.freeze()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(device)
        self._device = device
        return self._model

    def transcribe(self, audio_path: Path) -> str:
        model = self._load()
        if self.decoding == "ctc":
            model.cur_decoder = "ctc"
        else:
            model.cur_decoder = "rnnt"
        hypotheses = model.transcribe([str(audio_path)], batch_size=1)
        first = hypotheses[0]
        if isinstance(first, str):
            return first
        return str(getattr(first, "text", first))


def accepted_transcript(segments: object) -> str:
    """Keep each spoken sentence and drop a piece the decoder marks as unlikely."""
    kept: list[str] = []
    for segment in segments:
        if not _segment_kept(segment):
            continue
        words = getattr(segment, "words", None)
        lines = _sentences_from_words(words) if words else [_collapse_repeated_phrase(str(getattr(segment, "text", "")).strip())]
        for line in lines:
            cleaned = _collapse_repeated_phrase(line).strip()
            if cleaned and (not kept or kept[-1] != cleaned):
                kept.append(cleaned)
    return "\n".join(kept)


def _segment_kept(segment: object) -> bool:
    text = str(getattr(segment, "text", "")).strip()
    words = getattr(segment, "words", None)
    if not text and not words:
        return False
    logprob = float(getattr(segment, "avg_logprob", 0.0))
    no_speech = float(getattr(segment, "no_speech_prob", 0.0))
    compression = float(getattr(segment, "compression_ratio", 1.0))
    if logprob < -1.0 or compression > 2.4:
        return False
    if no_speech > 0.6 and logprob < -0.5:
        return False
    return True


def _sentences_from_words(words: object) -> list[str]:
    """One sentence per line. A long silent gap starts the next sentence."""
    sentences: list[list[str]] = []
    current: list[str] = []
    last_end: float | None = None
    for word in words:
        token = str(getattr(word, "word", "")).strip()
        probability = float(getattr(word, "probability", 1.0))
        start = float(getattr(word, "start", 0.0))
        end = float(getattr(word, "end", start))
        if not token or probability < 0.6:
            continue
        if last_end is not None and start - last_end > 0.7 and current:
            sentences.append(current)
            current = []
        if current and current[-1] == token:
            last_end = end
            continue
        current.append(token)
        last_end = end
        if token.endswith(("।", ".", "?", "!")):
            sentences.append(current)
            current = []
            last_end = None
    if current:
        sentences.append(current)
    return [" ".join(parts) for parts in sentences if parts]


def _collapse_repeated_phrase(text: str) -> str:
    parts = text.split()
    changed = True
    while changed and len(parts) >= 8:
        changed = False
        for size in range(len(parts) // 2, 3, -1):
            for start in range(0, len(parts) - (2 * size) + 1):
                if parts[start : start + size] == parts[start + size : start + (2 * size)]:
                    del parts[start + size : start + (2 * size)]
                    changed = True
                    break
            if changed:
                break
    return " ".join(parts)


class FasterWhisperBackend:
    def __init__(self, model_size: str = "models/telugu-asr-micro") -> None:
        self.name = f"faster-whisper:{model_size}"
        self.model_size = model_size
        self._model = None
        self._infer_lock = threading.RLock()

    def _load(self):
        with self._infer_lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel
            except Exception as exc:
                raise ModelUnavailable("faster-whisper is not installed") from exc
            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=2,
                num_workers=1,
                local_files_only=True,
            )
            return self._model

    def transcribe(self, audio_path: Path) -> str:
        model = self._load()
        with self._infer_lock:
            segments, _info = model.transcribe(
                str(audio_path),
                language="te",
                beam_size=5,
                temperature=0.0,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                    "min_silence_duration_ms": 400,
                    "speech_pad_ms": 160,
                },
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4,
                word_timestamps=True,
                hallucination_silence_threshold=1.0,
            )
            return accepted_transcript(segments)


def build_backend(spec: dict[str, object]) -> AsrBackend:
    backend = str(spec.get("backend", ""))
    if backend == "indicconformer":
        return IndicConformerBackend(
            checkpoint=str(spec.get("checkpoint", "ai4bharat/indicconformer_stt_te_hybrid_ctc_rnnt_large")),
            decoding=str(spec.get("decoding", "ctc")),
        )
    if backend == "faster_whisper":
        return FasterWhisperBackend(model_size=str(spec.get("model_size", "small")))
    raise ValueError(f"unknown ASR backend {backend}")
