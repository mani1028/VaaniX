"""Voice registry: what is installed, what is a voice, what is not.

No model weights are loaded. A voice pack is a folder shape in tmp_path:
``lite/models/hima-tts-nano-te/onnx/voices/step_int8.onnx`` plus
``voices/voices.json``. A missing marker means ``available`` is False —
never a synthesized result.

The OpenAI-compatible ``/v1/audio/voices`` route is Part 2 work
(src/telugu_speech/api/routes_openai.py). The test below skips (does not
fail) if that router is not wired into the app.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from telugu_speech.tts.voices import (
    VOICES,
    backend_for_voice,
    is_known_voice,
    list_installed_voices,
    pack_ready,
    preferred_voice,
    selected_voices,
)

PACK_REL = Path("lite/models/hima-tts-nano-te")


def _make_pack(root: Path) -> Path:
    pack = root / PACK_REL
    (pack / "onnx" / "voices" / "step_int8.onnx").parent.mkdir(parents=True)
    (pack / "onnx" / "voices" / "step_int8.onnx").write_bytes(b"fake-onnx")
    (pack / "voices").mkdir(parents=True, exist_ok=True)
    (pack / "voices" / "voices.json").write_text("{}", encoding="utf-8")
    return pack


def test_only_two_shipped_voices_exist() -> None:
    assert set(VOICES) == {"telugu_srinivas", "telugu_sravani"}
    assert VOICES["telugu_srinivas"]["gender"] == "male"
    assert VOICES["telugu_sravani"]["gender"] == "female"


def test_pack_is_not_ready_without_markers(tmp_path: Path) -> None:
    assert pack_ready(tmp_path) is False
    voices = list_installed_voices(tmp_path)
    assert {voice["id"] for voice in voices} == {"telugu_srinivas", "telugu_sravani"}
    assert all(voice["available"] is False for voice in voices)
    assert all(voice["language"] == "te" for voice in voices)


def test_tmp_voice_pack_reports_available(tmp_path: Path) -> None:
    _make_pack(tmp_path)
    assert pack_ready(tmp_path) is True
    voices = {voice["id"]: voice for voice in list_installed_voices(tmp_path)}
    assert voices["telugu_srinivas"]["available"] is True
    assert voices["telugu_srinivas"]["backend"] == "hima-onnx-int8"
    # Both default voices share the one int8 pack; they are not two engines.
    assert voices["telugu_srinivas"]["shared_weights"] is True
    assert voices["telugu_sravani"]["shared_weights"] is True


def test_voice_pack_marker_restricts_voices(tmp_path: Path) -> None:
    _make_pack(tmp_path)
    (tmp_path / "voice_pack.json").write_text(
        json.dumps({"voices": ["telugu_sravani"]}), encoding="utf-8"
    )
    assert selected_voices(tmp_path) == frozenset({"telugu_sravani"})
    assert is_known_voice("telugu_sravani", tmp_path) is True
    assert is_known_voice("telugu_srinivas", tmp_path) is False
    assert preferred_voice(tmp_path) == "telugu_sravani"
    assert [voice["id"] for voice in list_installed_voices(tmp_path)] == ["telugu_sravani"]


def test_unknown_voice_has_no_backend(tmp_path: Path) -> None:
    assert backend_for_voice("telugu_lab", tmp_path) is None
    assert backend_for_voice("grok-voice", tmp_path) is None
    assert is_known_voice("telugu_lab", tmp_path) is False


def test_backend_for_known_voice_is_lazy(tmp_path: Path) -> None:
    _make_pack(tmp_path)
    backend = backend_for_voice("telugu_srinivas", tmp_path)
    assert backend is not None
    assert backend.name.startswith("hima-onnx-int8:")
    assert backend.sample_rate_hz == 24000
    # Synthesizing needs the real ONNX engine, which is not in tmp_path:
    # it must raise ModelUnavailable, not crash or return fake audio.
    from telugu_speech.asr.infer import ModelUnavailable

    with pytest.raises(ModelUnavailable):
        backend.synthesize("నమస్కారం", None, "", tmp_path / "out.wav")


def _v1_routes_present() -> bool:
    if importlib.util.find_spec("telugu_speech.api.routes_openai") is None:
        return False
    from telugu_speech.api.main import app

    return any(getattr(route, "path", None) == "/v1/audio/voices" for route in app.routes)


def test_v1_voices_lists_both_shipped_voices(tmp_path: Path) -> None:
    if not _v1_routes_present():
        pytest.skip("openai compat router not wired yet (Part 2)")
    from fastapi.testclient import TestClient

    from telugu_speech.api.auth import Settings
    from telugu_speech.api.main import app, configure

    _make_pack(tmp_path)
    configure(Settings("local_open", frozenset(), 20000000, "indicconformer", "ctc", "small", "indicf5"))
    client = TestClient(app)
    response = client.get("/v1/audio/voices")
    assert response.status_code == 200
    ids = {voice["id"] for voice in response.json()}
    assert {"telugu_srinivas", "telugu_sravani"} <= ids
