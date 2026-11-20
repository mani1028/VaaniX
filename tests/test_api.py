from fastapi.testclient import TestClient

from telugu_speech.api.auth import Settings
from telugu_speech.api.main import app, configure


class FakeAsr:
    name = "fake"

    def transcribe(self, audio_path) -> str:
        return "నమస్కారం hello"


def _wav_bytes() -> bytes:
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)
    return buffer.getvalue()


def test_generated_audio_requires_the_same_key(tmp_path) -> None:
    configure(Settings("required", frozenset({"app-key"}), 20000000, "indicconformer", "ctc", "small", "indicf5"))
    app.state.output_dir = tmp_path
    name = "a" * 32 + ".wav"
    (tmp_path / name).write_bytes(_wav_bytes())
    client = TestClient(app)
    denied = client.get(f"/api/v1/tts/audio/{name}")
    assert denied.status_code == 401
    ready = client.get(f"/api/v1/tts/audio/{name}", headers={"Authorization": "Bearer app-key"})
    assert ready.status_code == 200
    assert ready.headers["content-type"].startswith("audio/wav")
    blocked = client.get("/api/v1/tts/audio/latest.wav", headers={"Authorization": "Bearer app-key"})
    assert blocked.status_code == 404


def test_dashboard_is_served() -> None:
    configure(Settings("required", frozenset(), 20000000, "indicconformer", "ctc", "small", "indicf5"))
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    assert "Telugu Speech Engine" in page.text
    assert "telugu-speech" in page.text
    assert 'pip install -e ".[host]"' in page.text
    assert "lite/models/hima-tts-nano-te" in page.text
    assert "Run checks" not in page.text


def test_health_is_public_and_protected_routes_deny_without_keys() -> None:
    configure(Settings("required", frozenset(), 20000000, "indicconformer", "ctc", "small", "indicf5"))
    client = TestClient(app)
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["auth_mode"] == "required"
    denied = client.post("/api/v1/asr/transcribe", files={"audio": ("a.wav", _wav_bytes(), "audio/wav")})
    assert denied.status_code == 503
    assert denied.json()["detail"]["code"] == "api_keys_not_configured"


def test_invalid_key_is_rejected() -> None:
    configure(Settings("required", frozenset({"secret-token"}), 20000000, "indicconformer", "ctc", "small", "indicf5"))
    client = TestClient(app)
    response = client.post(
        "/api/v1/asr/transcribe",
        headers={"Authorization": "Bearer wrong"},
        files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 401


def test_local_open_transcribes_with_injected_backend() -> None:
    configure(Settings("local_open", frozenset(), 20000000, "indicconformer", "ctc", "small", "indicf5"))
    app.state.asr_backend = FakeAsr()
    client = TestClient(app)
    response = client.post(
        "/api/v1/asr/transcribe",
        files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["code_mix"] is True
    assert body["norm_text"] == "నమస్కారం hello"
    rejected = client.post(
        "/api/v1/asr/transcribe",
        files={"audio": ("notes.txt", b"hello", "text/plain")},
    )
    assert rejected.status_code == 400
    models = client.get("/api/v1/models")
    assert models.status_code == 200
    tasks = {item["task"] for item in models.json()}
    assert {"asr", "tts", "detection"} <= tasks
    by_id = {item["id"]: item for item in models.json()}
    assert by_id["models/telugu-tts-micro"]["available"] is False
    empty = client.post("/api/v1/tts/generate", json={"text": ""})
    assert empty.status_code == 422
    assert empty.json()["detail"]["code"] == "invalid_text"
    too_long = client.post("/api/v1/tts/generate", json={"text": "న" * 2001})
    assert too_long.status_code == 422
    assert too_long.json()["detail"] == {"code": "invalid_text", "message": "Text must be at most 2000 characters."}
    blank_audio = client.post("/api/v1/asr/transcribe", files={"audio": ("empty.wav", b"", "audio/wav")})
    assert blank_audio.status_code == 400
    assert blank_audio.json()["detail"]["code"] == "invalid_audio"


def test_tts_rejects_non_telugu_and_unknown_voices() -> None:
    configure(Settings("local_open", frozenset(), 20000000, "indicconformer", "ctc", "small", "indicf5"))
    client = TestClient(app)
    missing_script = client.post("/api/v1/tts/generate", json={"text": "hello", "language": "te"})
    assert missing_script.status_code == 400
    unknown = client.post("/api/v1/tts/generate", json={"text": "నమస్కారం", "language": "te", "voice": "telugu_lab"})
    assert unknown.status_code == 404
    assert unknown.json()["detail"]["code"] == "unknown_voice"
    english = client.post("/api/v1/tts/stream", json={"text": "hello", "language": "te"})
    assert english.status_code == 400
    expressive = client.post(
        "/api/v1/tts/generate",
        json={"text": "నమస్కారం", "language": "te", "voice": "telugu_srinivas", "style": "excited"},
    )
    assert expressive.status_code == 400
    assert expressive.json()["detail"]["code"] == "unsupported_style"
    voices = client.get("/api/v1/tts/voices")
    assert voices.status_code == 200
    languages = client.get("/api/v1/asr/languages")
    assert languages.json()["languages"] == ["te"]
    assert languages.json()["code_mix_audio"] == "unsupported"
    garbage = client.post("/api/v1/asr/transcribe", files={"audio": ("notes.wav", b"not a wav", "audio/wav")})
    assert garbage.status_code == 400
    assert garbage.json()["detail"]["code"] == "invalid_audio"
    detected = client.post("/api/v1/detect", files={"audio": ("notes.wav", b"not a wav", "audio/wav")})
    assert detected.status_code == 400
    assert detected.json()["detail"]["message"] == "Upload a readable WAV file."
