"""Language detection route. Audio only."""

from __future__ import annotations

from pathlib import Path
import tempfile

import anyio

from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile

from telugu_speech.api.auth import authorize
from telugu_speech.api.uploads import UnreadableWav, read_wav_upload, require_readable_wav
from telugu_speech.asr.infer import ModelUnavailable
from telugu_speech.detection.lid import detect_path, load_detection_config

router = APIRouter(prefix="/api/v1", tags=["detection"])
CONFIG = Path("configs/detection.yaml")


@router.post("/detect")
async def detect(
    request: Request,
    audio: UploadFile = File(...),
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    settings = request.app.state.settings
    try:
        host = request.client.host if request.client else None
        authorize(settings, authorization, host)
    except PermissionError as exc:
        code = str(exc)
        status = 503 if code == "api_keys_not_configured" else 401
        raise HTTPException(status_code=status, detail={"code": code, "message": "Authentication failed."}) from exc
    payload = await read_wav_upload(audio, settings.max_upload_bytes)
    if not CONFIG.is_file():
        raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": "Detection config is missing."})
    config = load_detection_config(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "audio.wav"
        path.write_bytes(payload)
        try:
            require_readable_wav(path)
            result = await anyio.to_thread.run_sync(detect_path, path, config)
        except UnreadableWav as exc:
            raise HTTPException(status_code=400, detail={"code": "invalid_audio", "message": str(exc)}) from exc
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": str(exc)}) from exc
    return {
        "label": result.label,
        "confidence": result.confidence,
        "backend": result.backend,
        "code_mix_audio": result.code_mix_audio,
        "speech_seconds": result.speech_seconds,
        "closest": [{"language": name, "probability": score} for name, score in result.closest],
    }
