"""OpenAI-compatible surface: POST /v1/audio/speech + transcriptions.

These tests pin the *contract*, not the weights:

- routes that do not exist yet are skipped, never faked into passing;
- errors are OpenAI-shaped: ``{"error": {"message", "type", "code"}}``;
- a missing model reports 503 with that shape, never a traceback;
- auth is the same bearer key as /api/v1 (``TELUGU_API_KEYS``).

Part 2 owns the router (src/telugu_speech/api/routes_openai.py). If a test
below fails after Part 2 lands, the router broke the contract — do not
weaken the test to match it.
"""

from __future__ import annotations

import importlib.util
import io
import wave

import pytest
from fastapi.testclient import TestClient

from telugu_speech.api.auth import Settings
from telugu_speech.api.main import app, configure

SPEECH_PATH = "/v1/audio/speech"
TRANSCRIBE_PATH = "/v1/audio/transcriptions"
MODELS_PATH = "/v1/models"


class FakeAsr:
    name = "fake"

    def transcribe(self, audio_path) -> str:
        return "నమస్కారం hello"


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)
    return buffer.getvalue()


def _routes() -> set[str]:
    return {getattr(route, "path", "") for route in app.routes}


def _require_v1() -> None:
    if importlib.util.find_spec("telugu_speech.api.routes_openai") is None:
        pytest.skip("openai compat router not landed yet (Part 2)")
    missing = {SPEECH_PATH, TRANSCRIBE_PATH} - _routes()
    if missing:
        pytest.skip(f"openai compat routes not wired yet: {sorted(missing)}")


def _client() -> TestClient:
    configure(Settings("local_open", frozenset(), 20000000, "indicconformer", "ctc", "small", "indicf5"))
    app.state.asr_backend = FakeAsr()
    return TestClient(app)


def _assert_openai_error(body: object, status: int) -> None:
    assert status >= 400, f"expected an error status, got {status}: {body!r}"
    assert isinstance(body, dict) and isinstance(body.get("error"), dict)
    assert body["error"].get("message"), "error.message must not be empty"


def test_speech_rejects_unknown_voice_with_openai_error() -> None:
    _require_v1()
    client = _client()
    response = client.post(
        SPEECH_PATH,
        json={"model": "vaani-tts-piper", "input": "నమస్కారం", "voice": "telugu_lab"},
    )
    assert response.status_code in {400, 404}
    _assert_openai_error(response.json(), response.status_code)


def test_speech_rejects_empty_input() -> None:
    _require_v1()
    client = _client()
    response = client.post(SPEECH_PATH, json={"model": "vaani-tts-piper", "input": "", "voice": "telugu_srinivas"})
    assert response.status_code in {400, 422}
    _assert_openai_error(response.json(), response.status_code)


def test_transcriptions_accepts_fake_backend_or_reports_503() -> None:
    _require_v1()
    client = _client()
    response = client.post(
        TRANSCRIBE_PATH,
        files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"model": "vaani-asr-small", "language": "te"},
    )
    # With the fake ASR backend injected the route should answer 200.
    # If the route instead binds the real weights, 503 (not measured) is
    # the only acceptable non-200: weights are not in this checkout.
    assert response.status_code in {200, 503}
    if response.status_code == 200:
        assert "text" in response.json()
    else:
        _assert_openai_error(response.json(), response.status_code)


def test_transcriptions_rejects_missing_file() -> None:
    _require_v1()
    client = _client()
    response = client.post(TRANSCRIBE_PATH, data={"model": "vaani-asr-small"})
    assert response.status_code in {400, 422}
    _assert_openai_error(response.json(), response.status_code)


def test_v1_models_lists_openai_shaped_models() -> None:
    _require_v1()
    if MODELS_PATH not in _routes():
        pytest.skip("/v1/models not wired yet")
    client = _client()
    response = client.get(MODELS_PATH)
    assert response.status_code == 200
    body = response.json()
    models = body.get("data", body) if isinstance(body, dict) else body
    assert isinstance(models, list) and models
    assert all("id" in item for item in models)


def test_v1_requires_a_key_when_auth_is_required() -> None:
    _require_v1()
    configure(Settings("required", frozenset({"app-key"}), 20000000, "indicconformer", "ctc", "small", "indicf5"))
    client = TestClient(app)
    denied = client.post(
        SPEECH_PATH,
        json={"model": "vaani-tts-piper", "input": "నమస్కారం", "voice": "telugu_srinivas"},
    )
    assert denied.status_code == 401
    _assert_openai_error(denied.json(), denied.status_code)
