"""Platform HTTP API — Sarvam-style surface on open models."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

from fastapi import APIRouter, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "platform" / "src"))

from bhasha_platform import __version__
from bhasha_platform.asr import transcribe
from bhasha_platform.cascade import speech_to_speech
from bhasha_platform.catalog import load_registry
from bhasha_platform.translate import translate
from bhasha_platform.tts import synthesize
from telugu_speech.api.auth import Settings, authorize
from telugu_speech.asr.infer import ModelUnavailable
from telugu_speech.text.normalization import contains_telugu

router = APIRouter(prefix="/platform/v1", tags=["bhasha-platform"])
DASHBOARD = REPO / "platform" / "apps" / "index.html"
OUTPUT = REPO / "experiments" / "platform"
OUTPUT.mkdir(parents=True, exist_ok=True)


class TranslateBody(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    source: str = "te"
    target: str = "en"


class TtsBody(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    language: str = "te"
    backend: str = "telugu_micro"


def _auth(authorization: str | None) -> None:
    settings = Settings.from_env()
    try:
        authorize(settings, authorization)
    except PermissionError as exc:
        code = str(exc)
        status = 503 if code == "api_keys_not_configured" else 401
        raise HTTPException(status_code=status, detail={"code": code, "message": "Authentication failed."}) from exc


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "product": "Bhasha Platform", "version": __version__, "mode": "open_weights"}


@router.get("/catalog")
def catalog() -> dict:
    return load_registry()


@router.post("/asr/transcribe")
async def asr_route(
    audio: UploadFile = File(...),
    language: str = "te",
    authorization: str | None = Header(default=None),
) -> dict:
    _auth(authorization)
    payload = await audio.read()
    if not payload or not (audio.filename or "").lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail={"code": "invalid_audio"})
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / Path(audio.filename or "audio.wav").name
        path.write_bytes(payload)
        try:
            return transcribe(path, language=language)
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": str(exc)}) from exc


@router.post("/tts/synthesize")
def tts_route(body: TtsBody, authorization: str | None = Header(default=None)) -> dict:
    _auth(authorization)
    if body.language == "te" and not contains_telugu(body.text):
        raise HTTPException(status_code=400, detail={"code": "invalid_text"})
    out = OUTPUT / "tts.wav"
    try:
        return synthesize(body.text, out, backend=body.backend)
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": str(exc)}) from exc


@router.post("/translate")
def translate_route(body: TranslateBody, authorization: str | None = Header(default=None)) -> dict:
    _auth(authorization)
    try:
        return translate(body.text, source=body.source, target=body.target)
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": str(exc)}) from exc


@router.post("/speech-to-speech")
async def s2s_route(
    audio: UploadFile = File(...),
    source_language: str = "te",
    target_language: str = "en",
    authorization: str | None = Header(default=None),
) -> dict:
    _auth(authorization)
    payload = await audio.read()
    if not payload:
        raise HTTPException(status_code=400, detail={"code": "invalid_audio"})
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "in.wav"
        path.write_bytes(payload)
        out = OUTPUT / "s2s.wav"
        try:
            return speech_to_speech(path, out, source_language, target_language)
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": str(exc)}) from exc


@router.get("/audio/{name}")
def audio_file(name: str) -> FileResponse:
    safe = Path(name).name
    if safe != name or not safe.endswith(".wav"):
        raise HTTPException(status_code=400, detail={"code": "invalid_name"})
    path = (OUTPUT / safe).resolve()
    if path.parent != OUTPUT.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail={"code": "missing_audio"})
    return FileResponse(path, media_type="audio/wav", filename=safe)


def attach(app: FastAPI) -> FastAPI:
    app.include_router(router)

    @app.get("/platform/", include_in_schema=False)
    def platform_home() -> FileResponse:
        if not DASHBOARD.is_file():
            raise HTTPException(status_code=404, detail={"code": "missing_dashboard"})
        return FileResponse(DASHBOARD)

    return app


def main() -> None:
    import uvicorn
    from telugu_speech.api.main import app as lab_app

    attach(lab_app)
    uvicorn.run(lab_app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
