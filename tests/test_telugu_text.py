"""Telugu text normalization and TTS text preparation.

Pure functions only: no model weights, no network, no files on disk.
Expectations below are read from src/telugu_speech/text/normalization.py,
not guessed: digit words, lakh/crore expansion, and the spoken forms for
money, dates, times, and ordinals.
"""

import pytest

from telugu_speech.text.normalization import (
    NormalizationConfig,
    contains_telugu,
    find_code_mix_spans,
    has_code_mix,
    normalize_text,
    prepare_tts_text,
    speak_integer,
    tokens,
)


def test_digits_are_spoken_by_default() -> None:
    _raw, norm = normalize_text("నాకు 5 పుస్తకాలు")
    assert norm == "నాకు ఐదు పుస్తకాలు"
    _raw, norm = normalize_text("100 మంది")
    assert norm == "వంద మంది"


def test_digit_policies_keep_and_ascii() -> None:
    _raw, kept = normalize_text("5", NormalizationConfig(digit_policy="keep"))
    assert kept == "5"
    _raw, ascii_norm = normalize_text("౫", NormalizationConfig(digit_policy="ascii"))
    assert ascii_norm == "5"
    with pytest.raises(ValueError):
        NormalizationConfig(digit_policy="shout")


def test_lakh_and_crore_expansion() -> None:
    assert speak_integer(1000) == "వెయ్యి"
    assert speak_integer(100000) == "లక్ష"
    assert speak_integer(10000000) == "కోటి"
    assert speak_integer(2026) == "రెండు వేలు ఇరవై ఆరు"
    with pytest.raises(ValueError):
        speak_integer(-1)


def test_normalization_is_nfc_and_strips_punctuation() -> None:
    raw, norm = normalize_text("నమస్కారం!  ఎలా ఉన్నారు?")
    assert raw == "నమస్కారం! ఎలా ఉన్నారు?"
    assert norm == "నమస్కారం ఎలా ఉన్నారు"
    assert "!" not in norm and "?" not in norm


def test_latin_spans_stay_latin_and_are_tagged() -> None:
    text = "లైట్లు light కూడా వెలుగుతూ ఉన్నాయి"
    assert has_code_mix(text) is True
    spans = find_code_mix_spans(text)
    assert [span.text for span in spans] == ["light"]
    assert has_code_mix("నమస్కారం") is False
    _raw, norm = normalize_text("Hello WORLD")
    assert norm == "hello world"


def test_prepare_tts_text_reads_money_dates_times() -> None:
    assert prepare_tts_text("₹5") == "ఐదు రూపాయలు"
    assert prepare_tts_text("₹1.50") == "ఒక రూపాయి యాభై పైసలు"
    assert prepare_tts_text("21/11/2026") == "ఇరవై ఒకటి నవంబర్ రెండు వేలు ఇరవై ఆరు"
    assert prepare_tts_text("10:30") == "పది గంటలు ముప్పై నిమిషాలు"
    assert prepare_tts_text("50%") == "యాభై శాతం"
    assert prepare_tts_text("1వ") == "మొదటి"
    # Latin words stay Latin; they are not transliterated.
    assert "light" in prepare_tts_text("light వెలిగింది")


def test_contains_telugu_and_tokens() -> None:
    assert contains_telugu("నమస్కారం") is True
    assert contains_telugu("hello") is False
    assert tokens("  నమస్కారం   ఎలా  ") == ["నమస్కారం", "ఎలా"]
    assert tokens("") == []
