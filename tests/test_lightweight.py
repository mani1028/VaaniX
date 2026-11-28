"""Lightweight guarantees: lazy loading + INT8 defaults. No mocks, no weights.

Every assertion here constructs the real backend/config objects and checks
they hold no loaded model. A weight is loaded only by _load() /
synthesize() / transcribe() — never by construction. If a constructor ever
starts loading weights, these tests fail and docs/LIGHTWEIGHT.md must be
updated with the new numbers.

No decoding parameter, beam/VAD setting, or model default is asserted
beyond compute_type == "int8": those belong to the accuracy worker.
"""

from __future__ import annotations

from pathlib import Path

from telugu_speech.asr.decode import DecodingConfig
from telugu_speech.asr.infer import FasterWhisperBackend, IndicConformerBackend
from telugu_speech.core.config import load_config
from telugu_speech.tts.infer import build_tts_backend
from telugu_speech.tts.voices import backend_for_voice, pack_ready

ROOT = Path(__file__).resolve().parents[1]


def test_asr_default_is_int8_tiny() -> None:
    cfg = load_config(env={})
    assert list(cfg.asr_chain) == ["telugu-asr-micro"]
    assert cfg.asr_model == "models/telugu-asr-micro"
    assert cfg.asr_backend == "faster-whisper"
    assert DecodingConfig().compute_type == "int8"
    assert DecodingConfig().language == "te"


def test_tts_default_chain_is_hima_int8() -> None:
    cfg = load_config(env={})
    assert list(cfg.tts_chain) == ["hima-nano"]
    assert cfg.default_voice == "telugu_srinivas"
    assert cfg.default_engine == "hima-nano"
    assert pack_ready(ROOT)


def test_constructing_backends_loads_no_weights() -> None:
    asr = FasterWhisperBackend("models/telugu-asr-micro")
    assert asr._model is None
    conformer = IndicConformerBackend(
        "ai4bharat/indicconformer_stt_te_hybrid_ctc_rnnt_large", "ctc"
    )
    assert conformer._model is None
    micro = build_tts_backend("mms-micro", "models/telugu-tts-micro")
    assert micro.name == "mms-tts-tel-micro"
    assert micro._model is None and micro._tokenizer is None
    full = build_tts_backend("mms", "models/telugu-tts")
    assert full.name == "mms-tts-tel"
    assert full._model is None and full._tokenizer is None


def test_hima_voice_backend_is_lazy() -> None:
    backend = backend_for_voice("telugu_srinivas", ROOT)
    assert backend is not None
    assert backend._engine is None
    assert backend.sample_rate_hz == 24000


def test_configure_loads_no_weights() -> None:
    from telugu_speech.api.auth import Settings
    from telugu_speech.api.main import app, configure

    configure(
        Settings(
            "required",
            frozenset({"lightweight-test-key"}),
            20000000,
            "faster_whisper",
            "ctc",
            "models/telugu-asr-micro",
            "mms-micro",
        )
    )
    assert app.state.asr_backend._model is None
    assert app.state.tts_backend._model is None
