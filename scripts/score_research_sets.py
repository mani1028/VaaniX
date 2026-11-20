"""Score research clips and diagnose Telugu-English mixing. Does not change the recognizer.

The Andhra files come from SujithPulikodan/Vaani-sample-data, CC-BY-4.0 via
Project Vaani. They are prompted speech from Annamaya, Andhra Pradesh, with
background-noise tags and English words in braces. Sixteen clips are a
diagnosis, not a score for all of Andhra Pradesh. This is not a listener test.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telugu_speech.asr.infer import FasterWhisperBackend
from telugu_speech.evaluation.asr_metrics import character_error_rate, word_error_rate
from telugu_speech.evaluation.research_text import (
    english_hits,
    marked_english_words,
    research_reference,
    telugu_only,
    whole_token_present,
)
from telugu_speech.text.normalization import NormalizationConfig, normalize_text

SETTINGS = (
    ("production", "te", True),
    ("vad_off", "te", False),
    ("automatic_language", None, True),
)


def main() -> None:
    manifest = ROOT / "data/manifests/andhra_sample.jsonl"
    if not manifest.is_file():
        print("not measured. Andhra sample is not on disk.")
        return
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    backend = FasterWhisperBackend(str(ROOT / "models/telugu-asr-micro"))
    model = backend._load()
    decoded = {name: [_transcribe(model, row, language, vad) for row in rows] for name, language, vad in SETTINGS}
    clips = [_clip_report(row, decoded["production"][index]) for index, row in enumerate(rows)]
    report = {
        "clips": len(rows),
        "speakers": len({row.get("speaker_id") for row in rows}),
        "note": "Diagnostic on 16 prompted Andhra clips. Not a statewide score. Production decoder was not changed.",
        "production": _aggregate(rows, decoded["production"]),
        "settings": {name: _aggregate(rows, decoded[name]) for name, _language, _vad in SETTINGS},
        "vocabulary": _vocabulary(rows),
        "per_clip": clips,
    }
    destination = ROOT / "experiments" / "benchmarks" / "suite" / "codemix_diagnostic.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    production = report["production"]
    print(
        json.dumps(
            {
                "clips": report["clips"],
                "mixed_wer": production["mixed_wer"],
                "mixed_cer": production["mixed_cer"],
                "telugu_wer": production["telugu_wer"],
                "telugu_cer": production["telugu_cer"],
                "english_recall": production["english_recall"],
                "english_words": production["english_words"],
                "substitutions": production["substitutions"],
                "deletions": production["deletions"],
                "insertions": production["insertions"],
                "settings": {name: {"mixed_wer": item["mixed_wer"], "english_recall": item["english_recall"]} for name, item in report["settings"].items()},
                "report": str(destination),
            },
            ensure_ascii=False,
        )
    )


def _transcribe(model: object, row: dict[str, object], language: str | None, vad: bool) -> str:
    segments, _info = model.transcribe(  # type: ignore[attr-defined]
        str(row["audio"]),
        language=language,
        beam_size=5,
        temperature=0.0,
        vad_filter=vad,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        without_timestamps=True,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def _clip_report(row: dict[str, object], hypothesis: str) -> dict[str, object]:
    config = NormalizationConfig()
    english = marked_english_words(str(row["transcript"]))
    _raw, normalized = normalize_text(hypothesis, config)
    _reference_raw, reference = normalize_text(research_reference(str(row["transcript"])), config)
    words = word_error_rate(reference, normalized)
    chars = character_error_rate(reference, normalized)
    telugu_reference = telugu_only(reference)
    telugu_hypothesis = telugu_only(normalized)
    telugu_words = word_error_rate(telugu_reference, telugu_hypothesis)
    telugu_chars = character_error_rate(telugu_reference, telugu_hypothesis)
    hits = english_hits(english, normalized)
    return {
        "utt_id": row.get("utt_id"),
        "mixed_wer": _rate(words.substitutions + words.deletions + words.insertions, words.reference_length),
        "mixed_cer": _rate(chars.substitutions + chars.deletions + chars.insertions, chars.reference_length),
        "telugu_wer": _rate(telugu_words.substitutions + telugu_words.deletions + telugu_words.insertions, telugu_words.reference_length),
        "telugu_cer": _rate(telugu_chars.substitutions + telugu_chars.deletions + telugu_chars.insertions, telugu_chars.reference_length),
        "english_words": english,
        "english_hits": hits,
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
        "reference": reference,
        "hypothesis": normalized,
    }


def _aggregate(rows: list[dict[str, object]], hypotheses: list[str]) -> dict[str, object]:
    clips = [_clip_report(row, hypothesis) for row, hypothesis in zip(rows, hypotheses)]
    mixed_errors = mixed_refs = telugu_errors = telugu_refs = 0
    char_errors = char_refs = telugu_char_errors = telugu_char_refs = 0
    substitutions = deletions = insertions = hits = english_words = 0
    mixed_sentences = 0
    for clip in clips:
        if clip["english_words"]:
            mixed_sentences += 1
        substitutions += int(clip["substitutions"])
        deletions += int(clip["deletions"])
        insertions += int(clip["insertions"])
        hits += int(clip["english_hits"])
        english_words += len(clip["english_words"])
    # Recompute character and Telugu totals from the same clip fields that are rates.
    # Counts are recovered by scoring again so the totals are not averages of rates.
    config = NormalizationConfig()
    for row, hypothesis in zip(rows, hypotheses):
        english = marked_english_words(str(row["transcript"]))
        _raw, normalized = normalize_text(hypothesis, config)
        _reference_raw, reference = normalize_text(research_reference(str(row["transcript"])), config)
        words = word_error_rate(reference, normalized)
        chars = character_error_rate(reference, normalized)
        telugu_words = word_error_rate(telugu_only(reference), telugu_only(normalized))
        telugu_chars = character_error_rate(telugu_only(reference), telugu_only(normalized))
        mixed_errors += words.substitutions + words.deletions + words.insertions
        mixed_refs += words.reference_length
        char_errors += chars.substitutions + chars.deletions + chars.insertions
        char_refs += chars.reference_length
        telugu_errors += telugu_words.substitutions + telugu_words.deletions + telugu_words.insertions
        telugu_refs += telugu_words.reference_length
        telugu_char_errors += telugu_chars.substitutions + telugu_chars.deletions + telugu_chars.insertions
        telugu_char_refs += telugu_chars.reference_length
        del english
    return {
        "mixed_wer": _rate(mixed_errors, mixed_refs),
        "mixed_cer": _rate(char_errors, char_refs),
        "telugu_wer": _rate(telugu_errors, telugu_refs),
        "telugu_cer": _rate(telugu_char_errors, telugu_char_refs),
        "english_recall": _rate(hits, english_words),
        "english_hits": hits,
        "english_words": english_words,
        "mixed_sentences": mixed_sentences,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }


def _vocabulary(rows: list[dict[str, object]]) -> dict[str, object]:
    path = ROOT / "models/telugu-asr-micro/vocabulary.json"
    if not path.is_file():
        return {"status": "not_measured", "reason": "vocabulary.json is not on disk."}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    vocabulary = set(loaded) if isinstance(loaded, list) else set(loaded.keys())
    words = []
    for row in rows:
        words.extend(marked_english_words(str(row["transcript"])))
    unique = sorted(set(words))
    missing = [word for word in unique if not whole_token_present(word, vocabulary)]
    return {
        "unique_english_words": len(unique),
        "missing_as_whole_token": missing,
        "note": "A word missing as one token can still be built from smaller pieces. This does not prove an acoustic miss.",
    }


def _rate(errors: int, total: int) -> float | None:
    if total <= 0:
        return None
    return round(errors / total, 4)


if __name__ == "__main__":
    main()
