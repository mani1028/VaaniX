from pathlib import Path

from telugu_speech.evaluation.competitors import download_decision, locate_competitors
from telugu_speech.evaluation.dataset_gate import dataset_counts, release_decision
from telugu_speech.evaluation.research_text import (
    english_hits,
    marked_english_words,
    research_reference,
    telugu_only,
    whole_token_present,
)
from telugu_speech.evaluation.suite import latency_summary, model_catalog, percentile, render_html, silence_stats


def test_percentile_is_empty_without_samples() -> None:
    assert percentile([], 95) is None
    summary = latency_summary([0.2, 0.4, 0.6, 0.8])
    assert summary["count"] == 4
    assert summary["p50_s"] == 0.5


def test_missing_models_are_not_scored(tmp_path: Path) -> None:
    rows = {row["model"]: row for row in model_catalog(tmp_path)}
    assert rows["chatterbox"]["status"] == "not_installed"
    assert rows["whisper.cpp"]["status"] == "not_installed"
    assert rows["hima-int8"]["bytes"] is None


def test_report_does_not_invent_a_listening_score() -> None:
    html = render_html(
        {
            "hardware": {"platform": "test", "cpu_count": 1, "python": "3.11", "threads": 2},
            "models": [{"role": "tts", "model": "chatterbox", "status": "not_installed", "bytes": None, "note": "absent"}],
            "stt": [],
            "tts": [{"voice": "telugu_srinivas", "threads": 2, "rtf": 0.1, "mos": None}],
            "concurrency": [],
        }
    )
    assert "MOS was not collected" in html
    assert "4.5" not in html


def test_competitors_stay_unscored_until_files_exist(tmp_path: Path) -> None:
    rows = {row["model"]: row for row in locate_competitors(tmp_path)}
    assert rows["chatterbox-telugu"]["status"] == "not_installed"
    assert rows["laya-telugu"]["measured"] is False
    assert download_decision("laya-telugu", 20 * 1024**3)["allowed"] is False
    assert download_decision("chatterbox-telugu", 20 * 1024**3, allow_large=True)["allowed"] is False
    assert download_decision("sherpa-onnx", 2 * 1024**3)["allowed"] is False


def test_release_keeps_production_without_a_shared_measurement(tmp_path: Path) -> None:
    decision = release_decision({"status": "measured", "clips": 10, "wer": 0.4, "rtf": 0.1, "peak_rss_mb": 300}, {"status": "not_measured"})
    assert decision["replace_production"] is False
    counts = dataset_counts(tmp_path / "absent.jsonl")
    assert all(row["clips"] == 0 and row["target"] == 100 for row in counts)


def test_research_tags_keep_the_spoken_english_word() -> None:
    text = "<noise> లైట్లు {light} [breathing] సోఫా {sofa}-- </noise>"
    assert research_reference(text) == "లైట్లు light సోఫా sofa"
    assert marked_english_words(text) == ["light", "sofa"]
    assert english_hits(["light", "sofa"], "లైట్లు light కూడా") == 1
    assert telugu_only("లైట్లు light సోఫా sofa") == "లైట్లు సోఫా"
    assert whole_token_present("light", {"Ġlight"}) is True
    assert whole_token_present("sofa", {"light"}) is False


def test_one_clip_can_count_toward_region_and_noise(tmp_path: Path) -> None:
    manifest = tmp_path / "clips.jsonl"
    manifest.write_text('{"category":"andhra","code_mix":true,"noisy":true}\n', encoding="utf-8")
    counts = {row["category"]: row["clips"] for row in dataset_counts(manifest)}
    assert counts["andhra"] == 1
    assert counts["code_mix"] == 1
    assert counts["noisy"] == 1
    assert counts["clean_conversational"] == 0


def test_silence_stats_count_a_gap() -> None:
    stats = silence_stats([0.0, 0.0, 0.5], 2)
    assert stats["silence_fraction"] == 2 / 3
    assert stats["longest_silence_s"] == 1.0
