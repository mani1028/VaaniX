"""ASR backends. Heavy libraries are imported only when a backend is constructed."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Protocol

from telugu_speech.asr.decode import DecodingConfig, segment_quality, word_quality
from telugu_speech.text.postprocess import clean_transcript


class AsrBackend(Protocol):
    name: str

    def transcribe(self, audio_path: Path) -> str:
        """Return a transcript for one audio file."""


@dataclass(frozen=True)
class TranscribedSegment:
    """One decoded segment, with the quality the backend supplied.

    ``quality`` is None when the backend supplied no quality signal
    (for example IndicConformer) — callers must omit confidence then,
    never invent it. ``words`` entries are dicts with text/start/end
    and, only when supplied, probability.
    """

    text: str
    start: float = 0.0
    end: float = 0.0
    quality: dict[str, float] | None = None
    words: list[dict[str, object]] = field(default_factory=list)


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
        return clean_transcript(self.transcribe_with_segments(audio_path)[0].text)

    def transcribe_with_segments(self, audio_path: Path) -> list[TranscribedSegment]:
        """One segment for the whole clip; quality is None (not supplied)."""
        model = self._load()
        if self.decoding == "ctc":
            model.cur_decoder = "ctc"
        else:
            model.cur_decoder = "rnnt"
        hypotheses = model.transcribe([str(audio_path)], batch_size=1)
        first = hypotheses[0]
        text = first if isinstance(first, str) else str(getattr(first, "text", first))
        return [TranscribedSegment(text=text, quality=None)]


def accepted_transcript(segments: object, decoding: DecodingConfig | None = None) -> str:
    """Keep each spoken sentence and drop a piece the decoder marks as unlikely."""
    config = decoding or DecodingConfig()
    kept: list[str] = []
    for segment in segments:
        if not _segment_kept(segment, config):
            continue
        words = getattr(segment, "words", None)
        lines = _sentences_from_words(words, config) if words else [_collapse_repeated_phrase(str(getattr(segment, "text", "")).strip())]
        for line in lines:
            cleaned = _collapse_repeated_phrase(line).strip()
            if cleaned and (not kept or kept[-1] != cleaned):
                kept.append(cleaned)
    # NFC + whitespace cleanup on the way out, so the API text itself is
    # clean — not just the bench scores (which normalise separately).
    return clean_transcript("\n".join(kept))


def _segment_kept(segment: object, config: DecodingConfig) -> bool:
    text = str(getattr(segment, "text", "")).strip()
    words = getattr(segment, "words", None)
    if not text and not words:
        return False
    logprob = float(getattr(segment, "avg_logprob", 0.0))
    no_speech = float(getattr(segment, "no_speech_prob", 0.0))
    compression = float(getattr(segment, "compression_ratio", 1.0))
    if logprob < config.logprob_threshold or compression > config.compression_ratio_threshold:
        return False
    if no_speech > config.no_speech_threshold and logprob < -0.5:
        return False
    return True


def _sentences_from_words(words: object, config: DecodingConfig | None = None) -> list[str]:
    """One sentence per line. A long silent gap starts the next sentence."""
    threshold = (config or DecodingConfig()).word_probability_threshold
    sentences: list[list[str]] = []
    current: list[str] = []
    last_end: float | None = None
    for word in words:
        token = str(getattr(word, "word", "")).strip()
        probability = float(getattr(word, "probability", 1.0))
        if probability < threshold:
            continue
        start = float(getattr(word, "start", 0.0))
        end = float(getattr(word, "end", start))
        if not token:
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
    def __init__(
        self,
        model_size: str = "models/telugu-asr-micro",
        decoding: DecodingConfig | dict[str, object] | None = None,
    ) -> None:
        self.name = f"faster-whisper:{model_size}"
        self.model_size = model_size
        self.decoding = decoding if isinstance(decoding, DecodingConfig) else DecodingConfig.from_dict(decoding)
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
                compute_type=self.decoding.compute_type,
                cpu_threads=self.decoding.cpu_threads,
                num_workers=1,
                local_files_only=True,
            )
            return self._model

    def transcribe(self, audio_path: Path) -> str:
        return clean_transcript(
            "\n".join(segment.text for segment in self.transcribe_with_segments(audio_path))
        )

    def transcribe_with_segments(self, audio_path: Path) -> list[TranscribedSegment]:
        """Decode with the configured settings; keep only accepted segments.

        Each returned segment carries the quality the backend supplied
        (avg_logprob / no_speech_prob / compression_ratio) or None.
        Latin code-mixed words are preserved: the word filter drops
        low-probability words, never Latin words as a class.
        """
        model = self._load()
        config = self.decoding
        with self._infer_lock:
            segments, _info = model.transcribe(
                str(audio_path),
                language=config.language,  # locked to "te" by DecodingConfig
                beam_size=config.beam_size,
                temperature=config.temperature,
                vad_filter=config.vad_enabled,
                vad_parameters={
                    "threshold": config.vad_threshold,
                    "min_speech_duration_ms": config.vad_min_speech_duration_ms,
                    "min_silence_duration_ms": config.vad_min_silence_duration_ms,
                    "speech_pad_ms": config.vad_speech_pad_ms,
                },
                condition_on_previous_text=config.condition_on_previous_text,
                log_prob_threshold=config.logprob_threshold,
                no_speech_threshold=config.no_speech_threshold,
                compression_ratio_threshold=config.compression_ratio_threshold,
                word_timestamps=True,
                hallucination_silence_threshold=config.hallucination_silence_threshold,
            )
            kept: list[TranscribedSegment] = []
            for segment in segments:
                if not _segment_kept(segment, config):
                    continue
                words = getattr(segment, "words", None) or []
                cleaned_words: list[dict[str, object]] = []
                for word in words:
                    token = str(getattr(word, "word", "")).strip()
                    probability = word_quality(word)
                    if not token:
                        continue
                    if probability is not None and probability < config.word_probability_threshold:
                        continue
                    entry: dict[str, object] = {
                        "text": token,
                        "start": float(getattr(word, "start", 0.0)),
                        "end": float(getattr(word, "end", 0.0)),
                    }
                    if probability is not None:
                        entry["probability"] = probability
                    cleaned_words.append(entry)
                # Rebuild the text from the kept words when word timestamps
                # exist, so a word dropped for low probability does not
                # silently reappear in the segment text (this matches the
                # old accepted_transcript behaviour). Without words, use
                # the raw segment text, de-duplicated for repeated phrases.
                if cleaned_words:
                    text = _collapse_repeated_phrase(" ".join(str(w["text"]) for w in cleaned_words))
                elif words:
                    # Words existed but every one was dropped for low
                    # probability: do not resurrect them from the raw text.
                    continue
                else:
                    text = _collapse_repeated_phrase(str(getattr(segment, "text", "")).strip())
                if not text:
                    continue
                kept.append(
                    TranscribedSegment(
                        text=text,
                        start=float(getattr(segment, "start", 0.0)),
                        end=float(getattr(segment, "end", 0.0)),
                        quality=segment_quality(segment),
                        words=cleaned_words,
                    )
                )
            return kept


def build_backend(spec: dict[str, object]) -> AsrBackend:
    backend = str(spec.get("backend", ""))
    if backend == "indicconformer":
        return IndicConformerBackend(
            checkpoint=str(spec.get("checkpoint", "ai4bharat/indicconformer_stt_te_hybrid_ctc_rnnt_large")),
            decoding=str(spec.get("decoding", "ctc")),
        )
    if backend == "faster_whisper":
        decoding = spec.get("decoding_config")
        if decoding is not None and not isinstance(decoding, dict):
            raise ValueError("decoding_config must be a mapping")
        return FasterWhisperBackend(
            model_size=str(spec.get("model_size", "small")),
            decoding=decoding,  # type: ignore[arg-type]
        )
    raise ValueError(f"unknown ASR backend {backend}")


def decoding_from_env() -> DecodingConfig:
    """Build a DecodingConfig from VAANI_ASR_* env vars; defaults are the measured ones."""
    import os

    values: dict[str, object] = {}
    if os.environ.get("VAANI_ASR_BEAM"):
        values["beam_size"] = int(os.environ["VAANI_ASR_BEAM"])
    if os.environ.get("VAANI_ASR_TEMPERATURE"):
        values["temperature"] = float(os.environ["VAANI_ASR_TEMPERATURE"])
    if os.environ.get("VAANI_ASR_VAD") in {"0", "false", "off"}:
        values["vad_enabled"] = False
    if os.environ.get("VAANI_ASR_NO_SPEECH_THRESHOLD"):
        values["no_speech_threshold"] = float(os.environ["VAANI_ASR_NO_SPEECH_THRESHOLD"])
    return DecodingConfig.from_dict(values)
