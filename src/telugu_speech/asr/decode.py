"""Greedy CTC collapse + the shared ASR decoding configuration.

Decoding defaults are the measured live-decoder settings (20 Nov 2026):
language locked to Telugu (``te`` — never auto-detect into English),
beam 5, temperature 0.0, VAD on, no-speech threshold 0.6, logprob
threshold -1.0, compression-ratio threshold 2.4, word-probability
threshold 0.6. Changing a default here changes the measured numbers in
docs/benchmark_results.md — change it only with a paired bench re-run
and record the new config in the bench JSON.
"""

from __future__ import annotations

from dataclasses import dataclass


def greedy_ctc(token_ids: list[int], blank_id: int) -> list[int]:
    collapsed: list[int] = []
    previous: int | None = None
    for token_id in token_ids:
        if token_id == blank_id:
            previous = token_id
            continue
        if token_id != previous:
            collapsed.append(token_id)
        previous = token_id
    return collapsed


@dataclass(frozen=True)
class DecodingConfig:
    """Every knob that changes what the recogniser hears.

    ``language`` is deliberately not optional: Telugu is locked in.
    Auto-detection collapsed into English on the Andhra code-mix set
    (and VAD off / auto language both stayed at mixed WER 0.5458), so
    there is no auto-detect path to fall back to.
    """

    language: str = "te"
    beam_size: int = 5
    temperature: float = 0.0
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    vad_min_speech_duration_ms: int = 250
    vad_min_silence_duration_ms: int = 400
    vad_speech_pad_ms: int = 160
    no_speech_threshold: float = 0.6
    logprob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    word_probability_threshold: float = 0.6
    hallucination_silence_threshold: float = 1.0
    condition_on_previous_text: bool = False
    cpu_threads: int = 2
    compute_type: str = "int8"

    def __post_init__(self) -> None:
        if self.language != "te":
            raise ValueError("language is locked to 'te' for Vaani X; do not auto-detect")
        if self.beam_size < 1 or self.beam_size > 10:
            raise ValueError("beam_size must be 1..10")
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("temperature must be 0.0..1.0")
        if not 0.0 <= self.vad_threshold <= 1.0:
            raise ValueError("vad_threshold must be 0.0..1.0")
        if not 0.0 <= self.no_speech_threshold <= 1.0:
            raise ValueError("no_speech_threshold must be 0.0..1.0")
        if self.cpu_threads < 1 or self.cpu_threads > 4:
            raise ValueError("cpu_threads must be 1..4")
        if self.compute_type not in {"int8", "int8_float16", "float16", "float32"}:
            raise ValueError(f"unknown compute_type {self.compute_type!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "beam_size": self.beam_size,
            "temperature": self.temperature,
            "vad_enabled": self.vad_enabled,
            "vad_threshold": self.vad_threshold,
            "vad_min_speech_duration_ms": self.vad_min_speech_duration_ms,
            "vad_min_silence_duration_ms": self.vad_min_silence_duration_ms,
            "vad_speech_pad_ms": self.vad_speech_pad_ms,
            "no_speech_threshold": self.no_speech_threshold,
            "logprob_threshold": self.logprob_threshold,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "word_probability_threshold": self.word_probability_threshold,
            "hallucination_silence_threshold": self.hallucination_silence_threshold,
            "condition_on_previous_text": self.condition_on_previous_text,
            "cpu_threads": self.cpu_threads,
            "compute_type": self.compute_type,
        }

    @classmethod
    def from_dict(cls, values: dict[str, object] | None) -> "DecodingConfig":
        if not values:
            return cls()
        known = set(cls().__dict__.keys()) | {
            "language", "beam_size", "temperature", "vad_enabled", "vad_threshold",
            "vad_min_speech_duration_ms", "vad_min_silence_duration_ms", "vad_speech_pad_ms",
            "no_speech_threshold", "logprob_threshold", "compression_ratio_threshold",
            "word_probability_threshold", "hallucination_silence_threshold",
            "condition_on_previous_text", "cpu_threads", "compute_type",
        }
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown decoding keys: {unknown}")
        return cls(**{k: v for k, v in values.items() if k in known})  # type: ignore[arg-type]


def segment_quality(segment: object) -> dict[str, float] | None:
    """Return the quality signals a backend actually supplied, or None.

    faster-whisper segments carry ``avg_logprob``, ``no_speech_prob``
    and ``compression_ratio``. No confidence number is invented here:
    if the backend did not supply a field, that field is omitted, and
    a segment with none of them returns None. Callers (verbose_json)
    must omit the field in that case, never fabricate one.
    """
    quality: dict[str, float] = {}
    for attr in ("avg_logprob", "no_speech_prob", "compression_ratio"):
        value = getattr(segment, attr, None)
        if value is not None:
            try:
                quality[attr] = float(value)
            except (TypeError, ValueError):
                continue
    return quality or None


def word_quality(word: object) -> float | None:
    """Per-word probability from faster-whisper, or None if not supplied."""
    value = getattr(word, "probability", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
