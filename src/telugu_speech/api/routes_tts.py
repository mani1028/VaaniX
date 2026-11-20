"""TTS HTTP route."""

from __future__ import annotations

from pathlib import Path
import re
import tempfile
import threading
from uuid import uuid4

from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask

from telugu_speech.api.auth import authorize
from telugu_speech.api.schemas import TtsRequest, TtsResponse
from telugu_speech.api.uploads import read_wav_upload
from telugu_speech.asr.infer import ModelUnavailable
from telugu_speech.text.normalization import contains_telugu, prepare_tts_text
from telugu_speech.tts.voices import (
    SAMPLE_DIR,
    SUPPORTED_STYLES,
    is_known_voice,
    list_installed_voices,
    sample_names,
)

router = APIRouter(prefix="/api/v1/tts", tags=["tts"])
_STORED_WAV = re.compile(r"^[0-9a-f]{32}\.wav$")
_KEEP_WAVS = 32
_WAV_LOCK = threading.Lock()


def _checked_voice(body: TtsRequest, request: Request, authorization: str | None):
    try:
        authorize(request.app.state.settings, authorization)
    except PermissionError as exc:
        code = str(exc)
        status = 503 if code == "api_keys_not_configured" else 401
        raise HTTPException(status_code=status, detail={"code": code, "message": "Authentication failed."}) from exc
    if body.language != "te":
        raise HTTPException(status_code=400, detail={"code": "unsupported_language", "message": "Only Telugu (te) is supported."})
    if body.format != "wav":
        raise HTTPException(status_code=400, detail={"code": "unsupported_format", "message": "Only wav is supported."})
    if not contains_telugu(body.text):
        raise HTTPException(status_code=400, detail={"code": "invalid_text", "message": "Text must include Telugu letters."})
    if body.style not in SUPPORTED_STYLES:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "unsupported_style",
                "message": "Only conversational is supported. The installed models have no emotion control.",
            },
        )
    from telugu_speech.tts.voices import backend_for_voice

    voice_id = "telugu_srinivas" if body.voice == "telugu_default" else body.voice
    if not is_known_voice(voice_id):
        raise HTTPException(status_code=404, detail={"code": "unknown_voice", "message": "Voice is not installed."})
    try:
        backend = backend_for_voice(voice_id)
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": str(exc)}) from exc
    if backend is None:
        raise HTTPException(status_code=404, detail={"code": "unknown_voice", "message": "Voice is not installed."})
    return backend, prepare_tts_text(body.text)


@router.post("/generate", response_model=TtsResponse)
def generate(
    body: TtsRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> TtsResponse:
    backend, spoken = _checked_voice(body, request, authorization)
    output_dir = Path(request.app.state.output_dir)
    output = _new_wav_path(output_dir)
    try:
        written = backend.synthesize(spoken, None, body.reference_text or "", output)
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": str(exc)}) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail={"code": "reference_missing", "message": str(exc)}) from exc
    sample_rate = int(getattr(backend, "sample_rate_hz", 16000))
    return TtsResponse(
        audio_path=written.name,
        audio_url=f"/api/v1/tts/audio/{written.name}",
        backend=backend.name,
        sample_rate_hz=sample_rate,
    )


@router.post("/stream")
def stream_speech(
    body: TtsRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> StreamingResponse:
    """Send PCM as each sentence is ready. Sample rate is in the response headers."""
    backend, spoken = _checked_voice(body, request, authorization)
    try:
        pieces = iter(backend.iter_pcm(spoken))
        first = next(pieces)
    except StopIteration:
        first = None
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": str(exc)}) from exc

    def chunks():
        import numpy as np

        def encode(audio: object) -> bytes:
            samples = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
            return (samples * 32767.0).astype(np.int16).tobytes()

        if first is not None:
            yield encode(first)
        for audio in pieces:
            yield encode(audio)

    return StreamingResponse(
        chunks(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(int(getattr(backend, "sample_rate_hz", 24000))),
            "X-Channels": "1",
            "X-Sample-Format": "s16le",
        },
    )


@router.get("/voices")
def voices() -> dict[str, object]:
    installed = list_installed_voices()
    return {
        "voices": installed,
        "download_id": "hima-tts-nano-te",
        "shared_weights": True,
        "custom_voice": "IndicF5 is not enabled. It needs an authorized reference recording.",
        "styles": sorted(SUPPORTED_STYLES),
        "providers": [
            {
                "id": "hima-onnx-int8",
                "license": "other",
                "role": "low-memory voices shipped with the code",
            },
            {
                "id": "indicf5",
                "license": "MIT model card; cloning requires speaker permission",
                "role": "not installed; reference-conditioned experiments only",
            },
        ],
    }


@router.get("/audio/{name}")
def generated_audio(
    name: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> FileResponse:
    settings = request.app.state.settings
    try:
        authorize(settings, authorization)
    except PermissionError as exc:
        code = str(exc)
        status = 503 if code == "api_keys_not_configured" else 401
        raise HTTPException(status_code=status, detail={"code": code, "message": "Authentication failed."}) from exc
    if not _STORED_WAV.match(name):
        raise HTTPException(status_code=404, detail={"code": "unknown_audio", "message": "Audio was not found."})
    root = Path(request.app.state.output_dir).resolve()
    path = (root / name).resolve()
    if path.parent != root or not path.is_file():
        raise HTTPException(status_code=404, detail={"code": "unknown_audio", "message": "Audio was not found."})
    return FileResponse(path, media_type="audio/wav", filename=name)


def _new_wav_path(output_dir: Path) -> Path:
    with _WAV_LOCK:
        output_dir.mkdir(parents=True, exist_ok=True)
        _prune_stored_wavs(output_dir)
        return output_dir / f"{uuid4().hex}.wav"


def _prune_stored_wavs(output_dir: Path) -> None:
    stored = [path for path in output_dir.glob("*.wav") if _STORED_WAV.match(path.name)]
    stored.sort(key=lambda path: path.stat().st_mtime)
    overflow = stored[: max(0, len(stored) - (_KEEP_WAVS - 1))]
    for path in overflow:
        path.unlink(missing_ok=True)


@router.get("/voices/download")
def download_selected_voices(voices: str, request: Request) -> FileResponse:
    from telugu_speech.distribution.bundle import normalize_voice_ids, write_voice_bundle

    try:
        voice_ids = normalize_voice_ids(voices)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail={"code": "unknown_voice", "message": "Voice is not installed."}) from exc
    root = Path(getattr(request.app.state, "bundle_root", "."))
    handle = tempfile.NamedTemporaryFile(prefix="telugu-voice-", suffix=".zip", delete=False)
    archive_path = Path(handle.name)
    try:
        write_voice_bundle(root, voice_ids, handle)
        handle.close()
    except FileNotFoundError as exc:
        handle.close()
        archive_path.unlink(missing_ok=True)
        raise HTTPException(status_code=404, detail={"code": "pack_missing", "message": str(exc)}) from exc
    except Exception:
        handle.close()
        archive_path.unlink(missing_ok=True)
        raise
    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename="telugu-speech-" + "+".join(voice_ids) + ".zip",
        background=BackgroundTask(archive_path.unlink, missing_ok=True),
    )


@router.get("/voices/download/{voice_id}")
def download_voice_code(voice_id: str, request: Request) -> FileResponse:
    if not is_known_voice(voice_id):
        raise HTTPException(status_code=404, detail={"code": "unknown_voice", "message": "Voice is not installed."})
    root = Path(getattr(request.app.state, "bundle_root", "."))
    handle = tempfile.NamedTemporaryFile(prefix="telugu-voice-", suffix=".zip", delete=False)
    archive_path = Path(handle.name)
    try:
        from telugu_speech.distribution.bundle import write_voice_bundle

        write_voice_bundle(root, voice_id, handle)
        handle.close()
    except FileNotFoundError as exc:
        handle.close()
        archive_path.unlink(missing_ok=True)
        raise HTTPException(status_code=404, detail={"code": "pack_missing", "message": str(exc)}) from exc
    except Exception:
        handle.close()
        archive_path.unlink(missing_ok=True)
        raise
    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename=f"telugu-speech-{voice_id}.zip",
        background=BackgroundTask(archive_path.unlink, missing_ok=True),
    )


@router.get("/voices/samples/{name}")
def voice_sample(name: str) -> FileResponse:
    safe = Path(name).name
    if safe != name or safe not in sample_names():
        raise HTTPException(status_code=404, detail={"code": "unknown_sample", "message": "Sample was not found."})
    path = (SAMPLE_DIR / safe).resolve()
    if path.parent != SAMPLE_DIR.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail={"code": "unknown_sample", "message": "Sample was not found."})
    return FileResponse(path, media_type="audio/wav", filename=safe, content_disposition_type="inline")


@router.post("/voices/preview", response_model=TtsResponse)
def preview_voice(
    body: TtsRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> TtsResponse:
    settings = request.app.state.settings
    try:
        authorize(settings, authorization)
    except PermissionError as exc:
        code = str(exc)
        status = 503 if code == "api_keys_not_configured" else 401
        raise HTTPException(status_code=status, detail={"code": code, "message": "Authentication failed."}) from exc
    catalog = {str(item["id"]): item for item in list_installed_voices()}
    if body.voice not in catalog:
        raise HTTPException(status_code=404, detail={"code": "unknown_voice", "message": "Voice is not installed."})
    if not catalog[body.voice].get("available", False):
        raise HTTPException(
            status_code=503,
            detail={"code": "model_unavailable", "message": str(catalog[body.voice].get("note"))},
        )
    return generate(body, request, authorization)


@router.post("/voices/custom")
async def custom_voice(
    request: Request,
    reference: UploadFile = File(...),
    consent: str = Form(default="false"),
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    settings = request.app.state.settings
    try:
        authorize(settings, authorization)
    except PermissionError as exc:
        code = str(exc)
        status = 503 if code == "api_keys_not_configured" else 401
        raise HTTPException(status_code=status, detail={"code": code, "message": "Authentication failed."}) from exc
    if consent.strip().lower() not in {"true", "1", "yes"}:
        raise HTTPException(
            status_code=400,
            detail={"code": "consent_required", "message": "Custom voice requires consent=true for the reference speaker."},
        )
    backend = request.app.state.tts_backend
    if getattr(backend, "name", "") != "indicf5":
        raise HTTPException(
            status_code=503,
            detail={"code": "model_unavailable", "message": "IndicF5 is not the active TTS backend."},
        )
    payload = await read_wav_upload(reference, settings.max_upload_bytes)
    try:
        backend._load()
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": str(exc)}) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": str(exc)}) from exc
    destination = Path(request.app.state.output_dir) / "custom_reference.wav"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    request.app.state.reference_audio = destination
    return {
        "voice_id": None,
        "persistent": False,
        "status": "reference_accepted_for_session",
        "license": "caller-asserted consent; IndicF5 MIT plus model-card voice terms",
    }
