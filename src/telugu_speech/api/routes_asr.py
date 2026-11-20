"""ASR HTTP route."""

from __future__ import annotations

from pathlib import Path
import tempfile

import anyio

from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile

from telugu_speech.api.auth import authorize
from telugu_speech.api.schemas import AsrResponse
from telugu_speech.api.uploads import UnreadableWav, read_wav_upload, require_readable_wav
from telugu_speech.asr.infer import ModelUnavailable
from telugu_speech.text.normalization import NormalizationConfig, has_code_mix, normalize_text

router = APIRouter(prefix="/api/v1/asr", tags=["asr"])


@router.post("/transcribe", response_model=AsrResponse)
async def transcribe(
    request: Request,
    audio: UploadFile = File(...),
    authorization: str | None = Header(default=None),
) -> AsrResponse:
    settings = request.app.state.settings
    try:
        authorize(settings, authorization)
    except PermissionError as exc:
        code = str(exc)
        status = 503 if code == "api_keys_not_configured" else 401
        raise HTTPException(status_code=status, detail={"code": code, "message": "Authentication failed."}) from exc
    payload = await read_wav_upload(audio, settings.max_upload_bytes)
    filename = Path(audio.filename or "audio.wav").name
    backend = request.app.state.asr_backend
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / filename
        path.write_bytes(payload)
        try:
            require_readable_wav(path)
            hypothesis = await anyio.to_thread.run_sync(backend.transcribe, path)
        except UnreadableWav as exc:
            raise HTTPException(status_code=400, detail={"code": "invalid_audio", "message": str(exc)}) from exc
        except ModelUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "model_unavailable", "message": str(exc)},
            ) from exc
    raw_text, norm_text = normalize_text(hypothesis, NormalizationConfig())
    return AsrResponse(
        text=hypothesis,
        raw_text=raw_text,
        norm_text=norm_text,
        backend=getattr(backend, "name", settings.asr_backend),
        code_mix=has_code_mix(norm_text),
    )


@router.get("/languages")
def languages() -> dict[str, object]:
    return {
        "languages": ["te"],
        "modes": ["transcribe"],
        "code_mix_text": "flagged",
        "code_mix_audio": "unsupported",
    }
