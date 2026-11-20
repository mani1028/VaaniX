from pathlib import Path

import numpy as np

from telugu_speech.asr.dataset import Utterance, assert_speaker_disjoint, assign_splits, write_manifest, read_manifest
from telugu_speech.audio.augmentation import add_noise, telephone_bandwidth
from telugu_speech.audio.preprocessing import prepare_utterance
from telugu_speech.audio.resampling import tighten_pauses, write_wav
from telugu_speech.audio.vad import energy_vad


def _row(speaker: str, utt: str) -> Utterance:
    return Utterance(
        utt_id=utt,
        audio=f"{utt}.wav",
        raw_text="నమస్కారం",
        norm_text="నమస్కారం",
        speaker_id=speaker,
        region="telangana",
        condition="clean",
    )


def test_speaker_overlap_is_rejected() -> None:
    try:
        assert_speaker_disjoint({"train": [_row("s1", "a")], "test": [_row("s1", "b")]})
    except ValueError as exc:
        assert "s1" in str(exc)
    else:
        raise AssertionError("overlapping speaker was accepted")


def test_assign_splits_keeps_speakers_apart() -> None:
    rows = [_row(f"speaker-{index}", f"u{index}") for index in range(30)]
    splits = assign_splits(rows)
    assert_speaker_disjoint(splits)
    assert sum(len(part) for part in splits.values()) == 30


def test_manifest_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [_row("s", "u")])
    loaded = read_manifest(path)
    assert loaded[0].speaker_id == "s"
    assert loaded[0].code_mix is False


def test_prepare_resamples_to_16k(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    tone = (0.2 * np.sin(2 * np.pi * 220 * np.arange(8000) / 8000)).astype(np.float32)
    write_wav(source, tone, 8000)
    audio, spans = prepare_utterance(source)
    assert audio.shape[0] > 14000
    assert spans
    silent = np.zeros(16000, dtype=np.float32)
    assert energy_vad(silent, 16000) == []
    noisy = add_noise(audio, snr_db=10, rng=np.random.default_rng(0))
    assert noisy.shape == audio.shape
    phone = telephone_bandwidth(audio, 16000)
    assert phone.shape[0] > 0


def test_long_dead_air_is_shortened() -> None:
    rate = 24000
    tone = (0.2 * np.sin(2 * np.pi * 180 * np.arange(rate) / rate)).astype(np.float32)
    gap = np.zeros(rate, dtype=np.float32)
    audio = np.concatenate([tone, gap, tone])
    tightened = tighten_pauses(audio, rate)
    assert tightened.shape[0] < audio.shape[0] - int(0.6 * rate)
    assert tightened.shape[0] > tone.shape[0] * 2
