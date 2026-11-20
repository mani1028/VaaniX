import io
import json
from pathlib import Path
import zipfile

from fastapi.testclient import TestClient

from telugu_speech.api.auth import Settings
from telugu_speech.api.main import app, configure
from telugu_speech.distribution.bundle import bundle_files, write_voice_bundle
from telugu_speech.tts.voices import backend_for_voice, list_installed_voices, preferred_voice


def _touch(path: Path, payload: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _fixture(tmp_path: Path) -> None:
    _touch(tmp_path / "src" / "telugu_speech" / "api" / "main.py", b"print('core')\n")
    _touch(tmp_path / "pyproject.toml", b"[project]\nname='telugu-speech'\n")
    _touch(tmp_path / ".env", b"TELUGU_API_KEYS=secret-value\n")
    _touch(tmp_path / ".venv" / "pyvenv.cfg", b"home = hidden\n")
    _touch(tmp_path / "lite" / "models" / "hima-tts-nano-te" / "onnx" / "voices" / "step_int8.onnx", b"hima")
    _touch(tmp_path / "lite" / "models" / "hima-tts-nano-te" / "voices" / "voices.json", b"{}")
    _touch(tmp_path / "lite" / "voices" / "samples" / "srinivas.wav", b"srinivas")
    _touch(tmp_path / "lite" / "voices" / "samples" / "sravani.wav", b"sravani")
    _touch(tmp_path / "models" / "telugu-asr-micro" / "model.bin", b"asr")


def test_bundle_keeps_one_pack_and_drops_secrets(tmp_path: Path) -> None:
    _fixture(tmp_path)
    names = {name for _path, name in bundle_files(tmp_path, "telugu_srinivas")}
    assert "src/telugu_speech/api/main.py" in names
    assert "lite/models/hima-tts-nano-te/onnx/voices/step_int8.onnx" in names
    assert "lite/voices/samples/srinivas.wav" in names
    assert "models/telugu-asr-micro/model.bin" not in names
    assert ".env" not in names
    assert ".venv/pyvenv.cfg" not in names
    buffer = io.BytesIO()
    write_voice_bundle(tmp_path, "telugu_sravani", buffer)
    with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
        marker = json.loads(archive.read("voice_pack.json"))
        assert marker["voice"] == "telugu_sravani"
        assert marker["pack"] == "hima-tts-nano-te"
        assert "HOST.txt" in archive.namelist()


def test_download_route_returns_only_the_selected_pack(tmp_path: Path) -> None:
    _fixture(tmp_path)
    configure(Settings("local_open", frozenset(), 20000000, "indicconformer", "ctc", "small", "indicf5"))
    app.state.bundle_root = tmp_path
    client = TestClient(app)
    missing = client.get("/api/v1/tts/voices/download/telugu_lab")
    assert missing.status_code == 404
    escaped = client.get("/api/v1/tts/voices/download/..%2F.env")
    assert escaped.status_code == 404
    removed = client.get("/api/v1/tts/voices/download/te_female_01")
    assert removed.status_code == 404
    ready = client.get("/api/v1/tts/voices/download/telugu_sravani")
    assert ready.status_code == 200
    assert ready.headers["content-type"].startswith("application/zip")
    with zipfile.ZipFile(io.BytesIO(ready.content)) as archive:
        names = set(archive.namelist())
        assert json.loads(archive.read("voice_pack.json"))["voice"] == "telugu_sravani"
        assert "lite/models/hima-tts-nano-te/onnx/voices/step_int8.onnx" in names
        assert ".env" not in names
    app.state.bundle_root = Path(".")


def test_selected_voices_include_the_hima_pack_once(tmp_path: Path) -> None:
    _fixture(tmp_path)
    buffer = io.BytesIO()
    write_voice_bundle(tmp_path, ["telugu_srinivas", "telugu_sravani", "telugu_srinivas"], buffer)
    with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
        names = set(archive.namelist())
        marker = json.loads(archive.read("voice_pack.json"))
    assert marker["voice"] == "telugu_srinivas"
    assert marker["voices"] == ["telugu_srinivas", "telugu_sravani"]
    assert marker["packs"] == ["hima-tts-nano-te"]
    assert "lite/models/hima-tts-nano-te/onnx/voices/step_int8.onnx" in names
    assert "lite/voices/samples/srinivas.wav" in names
    assert "lite/voices/samples/sravani.wav" in names
    assert "models/telugu-asr-micro/model.bin" not in names
    configure(Settings("local_open", frozenset(), 20000000, "indicconformer", "ctc", "small", "indicf5"))
    app.state.bundle_root = tmp_path
    client = TestClient(app)
    ready = client.get("/api/v1/tts/voices/download?voices=telugu_srinivas,telugu_sravani")
    assert ready.status_code == 200
    disposition = ready.headers["content-disposition"]
    assert "telugu_srinivas" in disposition and "telugu_sravani" in disposition
    unknown = client.get("/api/v1/tts/voices/download?voices=telugu_srinivas,te_female_01")
    assert unknown.status_code == 404
    app.state.bundle_root = Path(".")


def test_downloaded_marker_limits_the_hosted_voices(tmp_path: Path) -> None:
    _touch(tmp_path / "lite" / "models" / "hima-tts-nano-te" / "onnx" / "voices" / "step_int8.onnx")
    _touch(tmp_path / "lite" / "models" / "hima-tts-nano-te" / "voices" / "voices.json", b"{}")
    (tmp_path / "voice_pack.json").write_text(
        json.dumps({"voice": "telugu_srinivas", "voices": ["telugu_srinivas"], "packs": ["hima-tts-nano-te"]})
    )
    assert preferred_voice(tmp_path) == "telugu_srinivas"
    listed = [item["id"] for item in list_installed_voices(tmp_path)]
    assert listed == ["telugu_srinivas"]
    default = backend_for_voice("telugu_default", tmp_path)
    assert default is not None and default.speaker == "Srinivas"
    assert backend_for_voice("telugu_sravani", tmp_path) is None
    assert backend_for_voice("te_female_01", tmp_path) is None


def test_downloaded_marker_selects_that_voice(tmp_path: Path) -> None:
    _touch(tmp_path / "lite" / "models" / "hima-tts-nano-te" / "onnx" / "voices" / "step_int8.onnx")
    _touch(tmp_path / "lite" / "models" / "hima-tts-nano-te" / "voices" / "voices.json", b"{}")
    (tmp_path / "voice_pack.json").write_text(json.dumps({"voice": "telugu_sravani", "pack": "hima-tts-nano-te"}))
    assert preferred_voice(tmp_path) == "telugu_sravani"
    backend = backend_for_voice("telugu_default", tmp_path)
    assert backend is not None
    assert backend.speaker == "Sravani"
