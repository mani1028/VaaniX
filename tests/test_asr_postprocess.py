"""ASR transcript post-processing + decoding-config wiring.

Pure functions and config objects only: fixed strings, no model
weights, no network, no files on disk. The strings below are literal
Telugu / Latin text, not mock data — they assert what clean_transcript
and accepted_transcript do to a given input.
"""

from __future__ import annotations

import unicodedata

import pytest

from telugu_speech.asr.decode import DecodingConfig
from telugu_speech.asr.infer import FasterWhisperBackend, accepted_transcript, decoding_from_env
from telugu_speech.text.postprocess import clean_transcript


def test_clean_transcript_nfc_normalises_decomposed_text() -> None:
    decomposed = unicodedata.normalize("NFD", "café నమస్కారం")
    assert decomposed != "café నమస్కారం"  # the test input really is decomposed
    assert clean_transcript(decomposed) == "café నమస్కారం"


def test_clean_transcript_keeps_telugu_text_unchanged() -> None:
    text = "నమస్కారం ఎలా ఉన్నారు"
    assert clean_transcript(text) == text


def test_clean_transcript_collapses_whitespace_and_drops_empty_lines() -> None:
    assert clean_transcript("నమస్కారం   ఎలా   ఉన్నారు") == "నమస్కారం ఎలా ఉన్నారు"
    assert clean_transcript("  నమస్కారం  \n\n  ఎలా ఉన్నారు  \n") == "నమస్కారం\nఎలా ఉన్నారు"
    assert clean_transcript("") == ""
    assert clean_transcript("   \n  ") == ""


def test_clean_transcript_keeps_latin_code_mix_and_digits_as_spoken() -> None:
    # Latin words keep their casing; digits are NOT expanded to words
    # here — the bench normalizer expands digits for scoring, but the
    # API must return what was said.
    assert clean_transcript("light వెలిగింది") == "light వెలిగింది"
    assert clean_transcript("నాకు 5 పుస్తకాలు") == "నాకు 5 పుస్తకాలు"
    assert clean_transcript("Hello WORLD") == "Hello WORLD"


def test_clean_transcript_is_idempotent() -> None:
    text = "నమస్కారం   ఎలా ఉన్నారు\n\nlight 5"
    once = clean_transcript(text)
    assert clean_transcript(once) == once


class _Piece:
    def __init__(self, text: str, logprob: float = -0.2, no_speech: float = 0.1, compression: float = 1.4) -> None:
        self.text = text
        self.avg_logprob = logprob
        self.no_speech_prob = no_speech
        self.compression_ratio = compression


def test_accepted_transcript_returns_cleaned_text() -> None:
    decomposed = unicodedata.normalize("NFD", "నమస్కారం")
    text = accepted_transcript([_Piece(f"{decomposed}   ఎలా   ఉన్నారు")])
    assert text == "నమస్కారం ఎలా ఉన్నారు"


def test_decoding_config_locks_language_to_telugu() -> None:
    assert DecodingConfig().language == "te"
    with pytest.raises(ValueError):
        DecodingConfig(language="en")
    with pytest.raises(ValueError):
        DecodingConfig(language="auto")


def test_decoding_config_measured_defaults() -> None:
    config = DecodingConfig()
    assert (config.beam_size, config.temperature) == (5, 0.0)
    assert config.vad_enabled is True
    assert (config.vad_threshold, config.vad_min_speech_duration_ms) == (0.5, 250)
    assert (config.vad_min_silence_duration_ms, config.vad_speech_pad_ms) == (400, 160)
    assert (config.no_speech_threshold, config.logprob_threshold) == (0.6, -1.0)
    assert (config.compression_ratio_threshold, config.word_probability_threshold) == (2.4, 0.6)
    assert config.condition_on_previous_text is False


def test_faster_whisper_backend_uses_measured_defaults_without_loading() -> None:
    # Constructing the backend loads no weights; the decode config is
    # available immediately and is what the live routes now read.
    backend = FasterWhisperBackend("models/telugu-asr-micro")
    assert backend.decoding == DecodingConfig()


def test_decoding_from_env_overrides_beam_vad_and_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAANI_ASR_BEAM", "3")
    monkeypatch.setenv("VAANI_ASR_VAD", "off")
    monkeypatch.setenv("VAANI_ASR_NO_SPEECH_THRESHOLD", "0.7")
    config = decoding_from_env()
    assert config.beam_size == 3
    assert config.vad_enabled is False
    assert config.no_speech_threshold == 0.7
    # Everything else stays at the measured defaults.
    assert config.language == "te" and config.temperature == 0.0


def test_decoding_from_env_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("VAANI_ASR_BEAM", "VAANI_ASR_TEMPERATURE", "VAANI_ASR_VAD", "VAANI_ASR_NO_SPEECH_THRESHOLD"):
        monkeypatch.delenv(name, raising=False)
    assert decoding_from_env() == DecodingConfig()


def test_decoding_from_env_rejects_bad_beam(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAANI_ASR_BEAM", "99")
    with pytest.raises(ValueError):
        decoding_from_env()
