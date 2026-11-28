from telugu_speech.text.g2p import graphemes_to_aksharas
from telugu_speech.text.normalization import (
    NormalizationConfig,
    find_code_mix_spans,
    normalize_text,
    prepare_tts_text,
    speak_integer,
)
from telugu_speech.text.tokenizer import CharTokenizer
from telugu_speech.asr.decode import greedy_ctc


def test_spoken_twenty_five_keeps_punctuation_in_raw() -> None:
    raw, norm = normalize_text("25.", NormalizationConfig(digit_policy="spoken"))
    assert raw == "25."
    assert norm == "ఇరవై ఐదు"


def test_telugu_digits_match_ascii_digits() -> None:
    _, ascii_norm = normalize_text("౨౫", NormalizationConfig(digit_policy="spoken"))
    _, digit_norm = normalize_text("25", NormalizationConfig(digit_policy="spoken"))
    assert ascii_norm == digit_norm == speak_integer(25)


def test_tts_text_expands_rupees_and_percent() -> None:
    spoken = prepare_tts_text("₹500 మరియు 25%")
    assert spoken == "ఐదు వందలు రూపాయలు మరియు ఇరవై ఐదు శాతం"


def test_complete_telugu_reads_amounts_dates_and_times() -> None:
    assert speak_integer(1_25_000) == "లక్ష ఇరవై ఐదు వేలు"
    assert speak_integer(1_00_00_000) == "కోటి"
    assert speak_integer(10**9) == "ఒకటి సున్నా సున్నా సున్నా సున్నా సున్నా సున్నా సున్నా సున్నా సున్నా"
    spoken = prepare_tts_text("౨౦/౧౧/౨౦౨౬న 10:30కి ₹1.50, 3.5% మరియు 1వ బహుమతి. free coffee.")
    assert spoken == (
        "ఇరవై నవంబర్ రెండు వేలు ఇరవై ఆరు న పది గంటలు ముప్పై నిమిషాలు కి "
        "ఒక రూపాయి యాభై పైసలు, మూడు పాయింట్ ఐదు శాతం మరియు మొదటి బహుమతి. free coffee."
    )
    assert prepare_tts_text("1,25,000 మంది") == "లక్ష ఇరవై ఐదు వేలు మంది"
    assert prepare_tts_text("ఒక కప్పు టీ ఇవ్వండి") == "ఒక కప్పు టియీ ఇవ్వండి"


def test_code_mix_is_tagged_not_transliterated() -> None:
    raw, norm = normalize_text("నమస్కారం Hello!", NormalizationConfig())
    assert "Hello" in raw
    spans = find_code_mix_spans(norm)
    assert spans[0].text == "hello"
    assert "హలో" not in norm


def test_akshara_split_and_tokenizer_roundtrip() -> None:
    units = graphemes_to_aksharas("నమ")
    assert units
    tokenizer = CharTokenizer.from_texts(["నమస్కారం"])
    encoded = tokenizer.encode("నమ")
    assert tokenizer.decode(encoded) == "నమ"


def test_greedy_ctc_collapses_repeats_and_blanks() -> None:
    assert greedy_ctc([1, 1, 0, 2, 2], blank_id=0) == [1, 2]
