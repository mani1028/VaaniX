from pathlib import Path

import numpy as np

from telugu_speech.asr.dataset import Utterance, write_manifest
from telugu_speech.asr.infer import accepted_transcript
from telugu_speech.audio.resampling import write_wav
from telugu_speech.evaluation.asr_metrics import word_error_rate
from telugu_speech.evaluation.benchmark import run_benchmark


class EchoBackend:
    name = "echo"

    def transcribe(self, audio_path: Path) -> str:
        return "నమస్కారం"


class MissingBackend:
    name = "missing"

    def transcribe(self, audio_path: Path) -> str:
        raise FileNotFoundError("weights absent")


class _Piece:
    def __init__(self, text: str, logprob: float = -0.2, no_speech: float = 0.1, compression: float = 1.4) -> None:
        self.text = text
        self.avg_logprob = logprob
        self.no_speech_prob = no_speech
        self.compression_ratio = compression


class _Word:
    def __init__(self, word: str, start: float, end: float, probability: float = 0.9) -> None:
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


def test_verifier_starts_each_sentence_on_a_new_line() -> None:
    timed = _Piece("హలో నా పేరు వస్తుంది నీళ్ళు నీళ్ళు")
    timed.words = [
        _Word("హలో", 0.0, 0.3),
        _Word("నా", 0.3, 0.5),
        _Word("పేరు", 0.5, 0.8),
        _Word("వస్తుంది", 0.8, 1.1, 0.4),
        _Word("నీళ్ళు", 2.4, 2.8),
        _Word("నీళ్ళు", 2.8, 3.1),
    ]
    assert accepted_transcript([timed]) == "హలో నా పేరు\nనీళ్ళు"


def test_accepted_transcript_drops_silence_and_repeats() -> None:
    phrase = "ఒక పెద్ద వాక్యం ఇక్కడ"
    text = accepted_transcript(
        [
            _Piece("నమస్కారం"),
            _Piece("నమస్కారం"),
            _Piece("ఊహించిన వాక్యం", logprob=-1.4),
            _Piece("నిశ్శబ్దం", logprob=-0.8, no_speech=0.9),
            _Piece(f"{phrase} {phrase}"),
        ]
    )
    assert text == f"నమస్కారం\n{phrase}"


def test_wer_substitution_rate() -> None:
    counts = word_error_rate("a b", "a c")
    assert (counts.substitutions, counts.deletions, counts.insertions, counts.reference_length) == (1, 0, 0, 2)
    assert counts.rate == 0.5


def test_benchmark_reports_slices(tmp_path: Path) -> None:
    audio = tmp_path / "utt.wav"
    write_wav(audio, np.zeros(1600, dtype=np.float32), 16000)
    manifest = tmp_path / "eval.jsonl"
    write_manifest(
        manifest,
        [
            Utterance(
                utt_id="u1",
                audio=str(audio),
                raw_text="నమస్కారం",
                norm_text="నమస్కారం",
                speaker_id="s1",
                region="rayalaseema",
                condition="phone_8k",
                duration_s=0.1,
            )
        ],
    )
    report = tmp_path / "report.md"
    text = run_benchmark(manifest, report, [("echo", EchoBackend())], max_clips=20, model_bytes={"echo": 12})
    assert "rayalaseema" in text
    assert "phone_8k" in text
    assert "0.0000" in text
    failed = run_benchmark(manifest, report, [("missing", MissingBackend())], max_clips=20)
    assert "unavailable" in failed
