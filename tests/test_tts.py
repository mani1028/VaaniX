from pathlib import Path

import pytest

from telugu_speech.asr.train import TrainingDeferred, load_training_config, train
from telugu_speech.compression.export_onnx import export_onnx
from telugu_speech.tts.chunks import speech_pieces
from telugu_speech.tts.runtime import _POOLS, borrow_engine, clear_shared_engines, speech_slots
from telugu_speech.tts.train import load_tts_config
from telugu_speech.tts.vocoder import select_vocoder


def test_finetune_config_is_valid_and_does_not_train() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_training_config(root / "configs/asr/finetune.yaml")
    assert config["speaker_independent"] is True
    with pytest.raises(TrainingDeferred):
        train(root / "configs/asr/finetune.yaml")


def test_tts_config_and_vocoder() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = load_tts_config(root / "configs/tts/fastpitch.yaml")
    assert payload["speakers"] == "single"
    assert select_vocoder("vocos") == "vocos"


def test_speech_slots_default_and_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TELUGU_SPEECH_SLOTS", raising=False)
    assert speech_slots() == 8
    monkeypatch.setenv("TELUGU_SPEECH_SLOTS", "32")
    assert speech_slots() == 16
    monkeypatch.setenv("TELUGU_SPEECH_SLOTS", "0")
    assert speech_slots() == 1


def test_extra_engines_are_dropped_when_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELUGU_SPEECH_SLOTS", "4")
    clear_shared_engines()
    key = ("scale-pack", "idle")
    made: list[object] = []

    def load() -> object:
        engine = object()
        made.append(engine)
        return engine

    first = borrow_engine(key, load)
    second = borrow_engine(key, load)
    with first, second:
        assert len(made) == 2
        assert len(_POOLS[key].engines) == 2
    assert len(_POOLS[key].engines) == 1
    clear_shared_engines()


def test_long_telugu_is_split_under_the_voice_limit() -> None:
    pieces = speech_pieces("న" * 2000)
    assert pieces
    assert all(len(piece) <= 180 for piece in pieces)
    assert "".join(pieces) == "న" * 2000
    sentence = "నమస్కారం. " * 4
    packed = speech_pieces(sentence)
    assert len(packed) == 1


def test_onnx_export_refuses_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        export_onnx(tmp_path / "missing.pt", tmp_path / "model.onnx")
