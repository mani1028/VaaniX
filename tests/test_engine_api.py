import io
import json
import math
import struct
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from telugu_speech.api.auth import Settings
from telugu_speech.api.main import app, configure
from telugu_speech.asr.infer import ModelUnavailable
from telugu_speech.detection.lid import DetectionConfig, detect_path
from telugu_speech.tts.infer import IndicF5Backend


class FakeAsr:
    name = "fake"

    def transcribe(self, audio_path) -> str:
        return "నమస్కారం hello!"


def _wav_bytes(samples: int = 160, amplitude: float = 0.0) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        if amplitude == 0.0:
            frames = b"\x00\x00" * samples
        else:
            frames = b"".join(
                struct.pack("<h", int(32767 * amplitude * math.sin(2 * math.pi * 220 * index / 16000)))
                for index in range(samples)
            )
        handle.writeframes(frames)
    return buffer.getvalue()


def test_detect_silence_is_uncertain_and_protected() -> None:
    configure(Settings("required", frozenset(), 20000000, "indicconformer", "ctc", "small", "indicf5"))
    client = TestClient(app)
    denied = client.post("/api/v1/detect", files={"audio": ("a.wav", _wav_bytes(), "audio/wav")})
    assert denied.status_code == 503
    configure(Settings("local_open", frozenset(), 20000000, "indicconformer", "ctc", "small", "indicf5"))
    client = TestClient(app)
    response = client.post("/api/v1/detect", files={"audio": ("a.wav", _wav_bytes(16000), "audio/wav")})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "uncertain"
    assert body["code_mix_audio"] == "unsupported"
    # A 1 s, 0.2-amplitude 220 Hz tone is above the energy gate, so the route
    # loads the language scorer. With faster-whisper installed it must return
    # 200 and must not call a sine tone Telugu. Without that library the route
    # returns 503 model_unavailable; that is the installed-runtime contract,
    # not a skipped classification check.
    tone = client.post("/api/v1/detect", files={"audio": ("a.wav", _wav_bytes(16000, 0.2), "audio/wav")})
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        faster_whisper = None
    if faster_whisper is None:
        assert tone.status_code == 503
        assert tone.json()["detail"]["code"] == "model_unavailable"
    else:
        assert tone.status_code == 200
        assert tone.json()["label"] != "te-IN"
        assert tone.json()["code_mix_audio"] == "unsupported"


def test_detect_raises_model_unavailable_without_lid_weights(tmp_path: Path) -> None:
    """The 503 contract for /api/v1/detect, tested at the function level.

    routes_detect.py maps ModelUnavailable -> 503 model_unavailable.
    detect_path raises it only when the LID weights are absent, so this
    test points at a path that does not exist instead of depending on
    the repo's real weights being absent (they are present).
    """
    clip = tmp_path / "tone.wav"
    clip.write_bytes(_wav_bytes(16000, 0.2))
    config = DetectionConfig(
        sample_rate_hz=16000,
        min_speech_seconds=0.4,
        te_confidence_min=0.6,
        other_confidence_min=0.5,
        backend="faster_whisper_multilingual",
        model_path="models/whisper-multilingual-tiny-missing",
        code_mix_audio="unsupported",
    )
    with pytest.raises(ModelUnavailable):
        detect_path(clip, config)


def test_stt_styles_and_stream_use_the_batch_backend() -> None:
    configure(Settings("local_open", frozenset(), 20000000, "indicconformer", "ctc", "small", "indicf5"))
    app.state.asr_backend = FakeAsr()
    client = TestClient(app)
    verbatim = client.post(
        "/api/v1/stt?style=verbatim",
        files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
    )
    assert verbatim.status_code == 200
    assert verbatim.json()["text"] == "నమస్కారం hello!"
    telugu = client.post(
        "/api/v1/stt?style=telugu",
        files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
    )
    assert "hello" in telugu.json()["text"]
    assert "!" not in telugu.json()["text"]
    with client.websocket_connect("/api/v1/stt/stream") as socket:
        socket.send_bytes(_wav_bytes())
        socket.send_text(json.dumps({"event": "end", "style": "verbatim"}))
        message = socket.receive_json()
    assert message["streaming"] is False
    assert message["text"] == "నమస్కారం hello!"
    assert "timestamps" not in message


def test_stream_denies_without_keys() -> None:
    configure(Settings("required", frozenset(), 20000000, "indicconformer", "ctc", "small", "indicf5"))
    client = TestClient(app)
    with client.websocket_connect("/api/v1/stt/stream") as socket:
        message = socket.receive_json()
    assert message["code"] == "api_keys_not_configured"
    assert message["streaming"] is False


def test_voices_are_real_and_custom_requires_consent() -> None:
    configure(Settings("local_open", frozenset(), 20000000, "indicconformer", "ctc", "small", "indicf5"))
    client = TestClient(app)
    catalog = client.get("/api/v1/tts/voices")
    ids = [item["id"] for item in catalog.json()["voices"]]
    for blocked in ("kavitha_te_conversation", "pooja_te_conversation", "tarun_te_narration"):
        assert blocked not in ids
    missing = client.post(
        "/api/v1/tts/voices/custom",
        data={"consent": "false"},
        files={"reference": ("ref.wav", _wav_bytes(), "audio/wav")},
    )
    assert missing.status_code == 400
    assert missing.json()["detail"]["code"] == "consent_required"
    app.state.tts_backend = IndicF5Backend()
    unavailable = client.post(
        "/api/v1/tts/voices/custom",
        data={"consent": "true"},
        files={"reference": ("ref.wav", _wav_bytes(), "audio/wav")},
    )
    assert unavailable.status_code == 503
    assert unavailable.json()["detail"]["code"] == "model_unavailable"
    unknown = client.post("/api/v1/tts/voices/preview", json={"text": "నమస్కారం", "voice": "kavitha_te_conversation"})
    assert unknown.status_code == 404
    sample = client.get("/api/v1/tts/voices/samples/srinivas.wav")
    assert sample.status_code in {200, 404}
    if sample.status_code == 200:
        assert sample.headers["content-type"].startswith("audio/wav")
        assert sample.headers["content-disposition"].startswith("inline")
    blocked = client.get("/api/v1/tts/voices/samples/../secrets.wav")
    assert blocked.status_code == 404
    catalog = client.get("/api/v1/tts/voices").json()
    ids = [item["id"] for item in catalog["voices"]]
    for voice_id in ("telugu_srinivas", "telugu_sravani"):
        assert voice_id in ids
    assert "te_female_01" not in ids and "te_male_01" not in ids
    assert catalog["styles"] == ["conversational"]
    assert "telugu_bhavana" not in ids and "telugu_karthik" not in ids
    assert "telugu_pro_sravani" not in ids


def test_old_asr_route_still_returns_the_same_fields() -> None:
    configure(Settings("local_open", frozenset(), 20000000, "indicconformer", "ctc", "small", "indicf5"))
    app.state.asr_backend = FakeAsr()
    client = TestClient(app)
    response = client.post("/api/v1/asr/transcribe", files={"audio": ("a.wav", _wav_bytes(), "audio/wav")})
    assert response.status_code == 200
    assert set(response.json()) == {"text", "raw_text", "norm_text", "backend", "code_mix"}
