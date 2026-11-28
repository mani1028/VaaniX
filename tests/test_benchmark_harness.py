"""Benchmark harness: AER, slices, JSON results, and the bench CLI.

No model weights, no datasets, no network. The fake backend returns a
fixed transcript; the CLI tests assert it fails clearly when a dataset
manifest is absent — datasets are not in this repo and are never invented.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from telugu_speech.evaluation.asr_metrics import (
    akshara_error_rate,
    akshara_units,
    character_error_rate,
    micro_average,
    segment_scores,
    word_error_rate,
)
from telugu_speech.evaluation.benchmark import (
    default_result_path,
    slice_stats,
    validate_benchmark_set,
)


def test_akshara_units_keep_conjuncts_together() -> None:
    # A virama conjunct is one akshara, not one unit per code point.
    assert len(akshara_units("క్ష")) == 1
    assert akshara_units("నమస్కారం") == ["న", "మ", "స్కా", "రం"]
    assert akshara_units("hello") == list("hello")
    assert akshara_units("") == []


def test_aer_is_lower_than_wer_for_a_matra_error() -> None:
    # One wrong ending voids the whole word for WER (1.0) but costs two
    # of four aksharas (0.5). Same transcript, very different headline.
    ref, hyp = "నమస్కారం", "నమస్కారము"
    wer = word_error_rate(ref, hyp).rate
    aer = akshara_error_rate(ref, hyp).rate
    assert wer == pytest.approx(1.0)
    assert aer == pytest.approx(0.5)
    assert aer < wer


def test_identical_text_scores_zero_on_all_three() -> None:
    text = "నమస్కారం ఎలా ఉన్నారు"
    assert word_error_rate(text, text).rate == 0.0
    assert character_error_rate(text, text).rate == 0.0
    assert akshara_error_rate(text, text).rate == 0.0


def test_segment_scores_split_numerals_and_latin() -> None:
    ref = "నాకు 5 పుస్తకాలు light కావాలి"
    hyp = "నాకు 6 పుస్తకాలు కావాలి"
    scores = segment_scores(ref, hyp)
    assert scores.words.rate == pytest.approx(2 / 5)
    assert scores.numerals is not None and scores.numerals.rate == 1.0
    assert scores.latin is not None and scores.latin.rate == 1.0
    plain = segment_scores("నమస్కారం", "నమస్కారం")
    assert plain.numerals is None and plain.latin is None


def test_micro_average_is_reference_weighted() -> None:
    from telugu_speech.evaluation.asr_metrics import ErrorCounts

    assert micro_average([]) is None
    assert micro_average([ErrorCounts(1, 0, 0, 10), ErrorCounts(1, 0, 0, 10)]) == pytest.approx(0.1)


def _row(region: str, speaker: str, wer: float) -> dict[str, object]:
    from telugu_speech.evaluation.asr_metrics import ErrorCounts

    n = 10
    errors = int(round(wer * n))
    counts = ErrorCounts(errors, 0, 0, n)
    return {
        "region": region,
        "condition": "clean",
        "code_mix": False,
        "speaker": speaker,
        "wer": counts,
        "cer": counts,
        "aer": counts,
        "wer_raw": counts,
        "cer_raw": counts,
        "aer_raw": counts,
        "numerals": None,
        "latin": None,
    }


def test_slice_stats_break_down_by_region_and_count_speakers() -> None:
    rows = [_row("telangana", "sp1", 0.2), _row("telangana", "sp2", 0.4), _row("coastal_andhra", "sp3", 0.0)]
    slices = {(row["kind"], row["name"]): row for row in slice_stats(rows)}
    assert slices[("region", "telangana")]["wer"] == pytest.approx(0.3)
    assert slices[("region", "telangana")]["speakers"] == 2
    assert slices[("region", "coastal_andhra")]["wer"] == pytest.approx(0.0)
    assert slices[("overall", "all")]["clips"] == 3
    assert slices[("overall", "all")]["speakers"] == 3


def test_validate_benchmark_set_rejects_ten_clips() -> None:
    from telugu_speech.asr.dataset import Utterance

    rows = [
        Utterance(f"u{i}", f"clip{i}.wav", "నమస్కారం", "నమస్కారం", f"sp{i % 3}", "telangana", "clean")
        for i in range(10)
    ]
    with pytest.raises(ValueError, match="10 clips"):
        validate_benchmark_set(rows)


def test_default_result_path_is_dated_and_namespaced(tmp_path: Path) -> None:
    path = default_result_path("telugu-asr-micro", tmp_path)
    assert path.parent.name == "telugu-asr-micro"
    assert path.suffix == ".json"
    assert path.parent.parent == tmp_path


def test_bench_cli_fails_clearly_without_a_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from telugu_speech.bench import main

    fake_model = tmp_path / "fake-model"
    fake_model.mkdir()
    with pytest.raises(SystemExit) as exc:
        main(["--asr-model", str(fake_model), "--sets", "fleurs-te", "--data-root", str(tmp_path)])
    captured = capsys.readouterr()
    assert "manifest.jsonl" in str(exc.value)
    assert "fleurs-te" in str(exc.value)
    assert str(tmp_path) in str(exc.value)
    assert captured.out == ""  # nothing was scored, so nothing was printed


def test_bench_cli_fails_clearly_without_a_model(tmp_path: Path) -> None:
    from telugu_speech.bench import main

    with pytest.raises(SystemExit, match="asr model not found"):
        main(["--asr-model", "no-such-model", "--sets", "fleurs-te", "--data-root", str(tmp_path)])


def test_run_benchmark_writes_json_with_wer_cer_aer(tmp_path: Path) -> None:
    from telugu_speech.asr.dataset import Utterance, write_manifest
    from telugu_speech.evaluation.benchmark import run_benchmark

    import io
    import wave

    manifest = tmp_path / "manifest.jsonl"
    audio = tmp_path / "clip.wav"
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)
    audio.write_bytes(buffer.getvalue())
    write_manifest(
        manifest,
        [Utterance("u1", str(audio), "నమస్కారం", "నమస్కారం", "sp1", "telangana", "clean", duration_s=1.0)],
    )

    class FakeBackend:
        def transcribe(self, path):
            return "నమస్కారం"

    report = tmp_path / "report.md"
    json_path = tmp_path / "result.json"
    # The fake backend returns the reference text, so a clean run scores
    # 0.0 on all three metrics. If audio preparation ever fails, the run
    # records an error instead — either way the JSON must exist and must
    # not invent a WER.
    text = run_benchmark(manifest, report, [("fake-model", FakeBackend())], max_clips=10, json_path=json_path)
    assert json_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    model = payload["models"][0]
    assert model["model_id"] == "fake-model"
    assert model["model_revision_sha"] is None
    if model["error"] is None:
        assert model["wer"] == 0.0 and model["cer"] == 0.0 and model["aer"] == 0.0
        assert "WER" in text and "AER" in text
    else:
        assert model["wer"] is None
