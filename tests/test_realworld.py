"""Real-world harness: manifest parsing, condition validation, aggregation.

Offline, no model, no network. The WAVs written here are empty fixtures
whose only job is to satisfy the manifest's audio-exists check — nothing
is transcribed or scored from them, and no score is asserted from fake
audio. Scores are tested through the metric functions on fixed strings
and fixed error counts only.
"""

from __future__ import annotations

import json
from pathlib import Path
import wave

import pytest

from telugu_speech.evaluation.asr_metrics import ErrorCounts
from telugu_speech.evaluation.realworld import (
    CONDITIONS,
    condition_stats,
    load_manifest,
    pooled_stats,
    render_condition_table,
)


def _silent_wav(path: Path, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * int(16000 * seconds))


def test_conditions_are_the_six_required() -> None:
    assert CONDITIONS == ("clean", "noise", "phone_8k", "code_mix", "fast", "multi_speaker")


def test_load_csv_manifest_resolves_paths_and_defaults(tmp_path: Path) -> None:
    clip_a = tmp_path / "a.wav"
    clip_b = tmp_path / "b.wav"
    _silent_wav(clip_a)
    _silent_wav(clip_b)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,reference_text,condition,speaker,region,duration_s\n"
        f"{clip_a},నమస్కారం ఎలా ఉన్నారు,clean,mani,telangana,1.0\n"
        f"{clip_b},light కూడా వెలుగుతూ ఉన్నాయి,code_mix,,,\n",
        encoding="utf-8",
    )
    rows = load_manifest(manifest)
    assert len(rows) == 2
    assert rows[0].reference_text == "నమస్కారం ఎలా ఉన్నారు"
    assert rows[0].condition == "clean"
    assert rows[0].speaker == "mani"
    assert rows[0].region == "telangana"
    assert rows[0].duration_s == pytest.approx(1.0)
    assert rows[1].condition == "code_mix"
    assert rows[1].speaker == "unknown"  # default when the column is blank
    assert rows[1].region == "unspecified"
    assert rows[1].duration_s is None


def test_load_jsonl_manifest(tmp_path: Path) -> None:
    clip = tmp_path / "clip.wav"
    _silent_wav(clip)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"path": str(clip), "reference_text": "నమస్కారం", "condition": "fast", "speaker": "sp1"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = load_manifest(manifest)
    assert len(rows) == 1
    assert rows[0].condition == "fast"
    assert rows[0].utt_id == "clip"


def test_load_manifest_accepts_audio_alias_and_relative_path(tmp_path: Path) -> None:
    clip = tmp_path / "clip.wav"
    _silent_wav(clip)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"audio": "clip.wav", "raw_text": "నమస్కారం", "condition": "noise", "speaker": "sp1"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = load_manifest(manifest)
    assert rows[0].audio.endswith("clip.wav")
    assert rows[0].reference_text == "నమస్కారం"


def test_unknown_condition_is_rejected_naming_the_value(tmp_path: Path) -> None:
    clip = tmp_path / "clip.wav"
    _silent_wav(clip)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,reference_text,condition,speaker\n" f"{clip},నమస్కారం,studio,mani\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="studio"):
        load_manifest(manifest)


def test_missing_audio_is_named_and_nothing_is_scored(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,reference_text,condition,speaker\n" "no-such-clip.wav,నమస్కారం,clean,mani\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="no-such-clip.wav"):
        load_manifest(manifest)


def test_empty_reference_is_rejected(tmp_path: Path) -> None:
    clip = tmp_path / "clip.wav"
    _silent_wav(clip)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,reference_text,condition,speaker\n" f"{clip},,clean,mani\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reference_text"):
        load_manifest(manifest)


def _row(condition: str, speaker: str, wer_errors: int, ref_words: int = 10) -> dict[str, object]:
    counts = ErrorCounts(wer_errors, 0, 0, ref_words)
    return {
        "condition": condition,
        "speaker": speaker,
        "duration_s": 2.0,
        "elapsed_s": 0.2,
        "wer": counts,
        "cer": counts,
        "aer": counts,
        "numerals": None,
        "latin": None,
    }


def test_condition_stats_micro_average_is_reference_weighted() -> None:
    rows = [_row("clean", "sp1", 1), _row("clean", "sp2", 3), _row("noise", "sp1", 0)]
    stats = {row["condition"]: row for row in condition_stats(rows)}
    # clean: (1+3) errors over (10+10) reference words = 0.2; noise: 0.0.
    assert stats["clean"]["wer"] == pytest.approx(0.2)
    assert stats["clean"]["clips"] == 2
    assert stats["clean"]["speakers"] == 2
    assert stats["noise"]["wer"] == pytest.approx(0.0)
    assert stats["noise"]["rtf"] == pytest.approx(0.1)
    # Conditions come out in the canonical order, and only present ones appear.
    assert [row["condition"] for row in condition_stats(rows)] == ["clean", "noise"]


def test_condition_stats_empty_is_empty() -> None:
    assert condition_stats([]) == []


def test_pooled_stats_covers_all_conditions() -> None:
    rows = [_row("clean", "sp1", 1), _row("noise", "sp2", 1)]
    pooled = pooled_stats(rows)
    assert pooled["condition"] == "pooled"
    assert pooled["clips"] == 2
    assert pooled["speakers"] == 2
    assert pooled["wer"] == pytest.approx(0.1)


def test_render_table_lists_conditions_before_pooled_and_warns_small_n() -> None:
    rows = [_row("clean", "sp1", 1), _row("noise", "sp2", 1)]
    stats = condition_stats(rows)
    text = render_condition_table(stats, pooled_stats(rows), "telugu-asr-micro")
    assert "clean" in text and "noise" in text
    assert text.index("clean") < text.index("pooled")
    assert "Do not quote it alone" in text
    assert "Small-n warning" in text  # 1 clip per condition is not quotable


def test_render_table_marks_small_n_per_condition() -> None:
    rows = [_row("clean", f"sp{i}", 0) for i in range(12)]
    text = render_condition_table(condition_stats(rows), pooled_stats(rows), "telugu-asr-micro")
    assert "Small-n warning" not in text  # 12 clips, 12 speakers: no warning
