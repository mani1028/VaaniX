import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from telugu_speech.asr.infer import IndicConformerBackend, ModelUnavailable
from telugu_speech.asr.train import TrainingDeferred, assert_eval_not_in_train, train
from telugu_speech.audio.resampling import write_wav
from telugu_speech.detection.lid import classify_probabilities, detect_path, load_detection_config
from telugu_speech.evaluation.benchmark import run_benchmark
from telugu_speech.evaluation.detection_report import write_detection_report
from telugu_speech.text.styles import apply_output_style


def _config():
    return load_detection_config(Path("configs/detection.yaml"))


def _tone(path: Path, seconds: float = 1.0) -> None:
    rate = 16000
    samples = np.arange(int(rate * seconds))
    audio = (0.2 * np.sin(2 * np.pi * 220 * samples / rate)).astype(np.float32)
    write_wav(path, audio, rate)


def test_silence_is_uncertain_and_does_not_call_the_model(tmp_path: Path) -> None:
    path = tmp_path / "silence.wav"
    write_wav(path, np.zeros(16000, dtype=np.float32), 16000)
    called = {"count": 0}

    def explode(_path: Path) -> dict[str, float]:
        called["count"] += 1
        return {"te": 0.99}

    result = detect_path(path, _config(), explode, "stub")
    assert result.label == "uncertain"
    assert result.code_mix_audio == "unsupported"
    assert called["count"] == 0


def test_tone_with_english_probabilities_is_not_telugu(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    _tone(path)
    result = detect_path(path, _config(), lambda _path: {"en": 0.91, "te": 0.02}, "stub")
    assert result.label == "other"
    assert result.label != "te-IN"


def test_missing_lid_weights_do_not_label_a_tone_as_telugu(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    _tone(path)
    missing = _config().__class__(**{**_config().__dict__, "model_path": str(tmp_path / "absent-lid")})
    with pytest.raises(ModelUnavailable):
        detect_path(path, missing)


def test_detector_has_no_text_argument() -> None:
    parameters = inspect.signature(detect_path).parameters
    assert "text" not in parameters
    assert "transcript" not in parameters


def test_low_confidence_telugu_is_uncertain() -> None:
    config = _config()
    result = classify_probabilities({"te": 0.35, "si": 0.34}, 1.0, config, "stub")
    assert result.label == "uncertain"
    assert result.code_mix_audio == "unsupported"
    assert result.closest[0] == ("te", 0.35)
    assert result.closest[1][0] == "si"


def test_styles_do_not_invent_fillers() -> None:
    verbatim, raw, norm = apply_output_style("నమస్కారం hello!", "verbatim")
    assert verbatim == "నమస్కారం hello!"
    telugu, _raw, _norm = apply_output_style("నమస్కారం hello!", "telugu")
    assert telugu == norm
    assert "hello" in telugu
    codemix, raw_text, _norm = apply_output_style("నమస్కారం hello!", "codemix")
    assert codemix == raw
    assert codemix == raw_text


def test_detection_gate_is_not_evaluable(tmp_path: Path) -> None:
    manifest = Path("data/manifests/eval.jsonl")
    text = write_detection_report(manifest, tmp_path / "status.md")
    assert "not evaluable" in text
    assert "not measured" in text or "not measured" in text.lower() or "are not measured" in text


def test_indicconformer_without_nemo_is_not_measured(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    write_wav(audio, np.zeros(1600, dtype=np.float32), 16000)
    manifest = tmp_path / "eval.jsonl"
    row = {
        "utt_id": "local-1",
        "audio": str(audio),
        "raw_text": "నమస్కారం",
        "norm_text": "నమస్కారం",
        "speaker_id": "s",
        "region": "unspecified",
        "condition": "clean",
        "code_mix": False,
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    text = run_benchmark(
        manifest,
        tmp_path / "report.md",
        [("indicconformer_te_ctc", IndicConformerBackend("ai4bharat/indicconformer_stt_te_hybrid_ctc_rnnt_large", "ctc"))],
        max_clips=1,
    )
    assert "not measured" in text


def test_train_rejects_fleurs_overlap_and_still_refuses(tmp_path: Path) -> None:
    leaked = tmp_path / "train.jsonl"
    leaked.write_text(Path("data/manifests/eval.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    config = tmp_path / "finetune.yaml"
    config.write_text(
        "\n".join(
            [
                "optimizer: adamw",
                "learning_rate: 0.0001",
                "decoder: ctc",
                "speaker_independent: true",
                "datasets: [kathbath_te]",
                "held_out: [fleurs_te]",
                "held_out_manifest: data/manifests/eval.jsonl",
                f'train_manifest: "{leaked}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="held-out"):
        train(config)
    payload = {
        "held_out": ["fleurs_te"],
        "datasets": ["fleurs_te"],
    }
    with pytest.raises(ValueError, match="evaluation-only"):
        assert_eval_not_in_train(payload)
    with pytest.raises(TrainingDeferred):
        train(Path("configs/asr/finetune.yaml"))


def test_optional_manifest_fields_still_load(tmp_path: Path) -> None:
    from telugu_speech.asr.dataset import read_manifest

    path = tmp_path / "row.jsonl"
    path.write_text(
        json.dumps(
            {
                "utt_id": "opt-1",
                "audio": "opt.wav",
                "raw_text": "నమస్కారం",
                "norm_text": "నమస్కారం",
                "speaker_id": "s",
                "region": "telangana",
                "condition": "clean",
                "lid_label": "te-IN",
                "license": "CC0-1.0",
                "consent_status": "granted",
                "transcript_romanized": "namaskaram",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = read_manifest(path)
    assert rows[0].utt_id == "opt-1"
    assert rows[0].code_mix is False
