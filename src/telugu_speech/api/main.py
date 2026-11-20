"""FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
import sys

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from telugu_speech import __version__
from telugu_speech.api.auth import Settings
from telugu_speech.api.routes_asr import router as asr_router
from telugu_speech.api.routes_detect import router as detect_router
from telugu_speech.api.routes_stt import router as stt_router
from telugu_speech.api.routes_tts import router as tts_router
from telugu_speech.api.schemas import HealthResponse, ModelInfo
from telugu_speech.asr.infer import FasterWhisperBackend, IndicConformerBackend
from telugu_speech.tts.infer import build_tts_backend

def _warm_default_voice() -> None:
    import tempfile

    from telugu_speech.tts.voices import backend_for_voice

    backend = backend_for_voice("telugu_srinivas")
    if backend is None:
        return
    with tempfile.TemporaryDirectory() as directory:
        backend.synthesize("నమస్కారం", None, "", Path(directory) / "warm.wav")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    if os.environ.get("TELUGU_WARMUP") == "1":
        import anyio

        await anyio.to_thread.run_sync(_warm_default_voice)
    yield


app = FastAPI(
    title="Telugu Speech API",
    version=__version__,
    description="Self-hosted Telugu speech-to-text and text-to-speech. Low-memory int8 models. Not a measured Sarvam quality match.",
    lifespan=_lifespan,
)
_cors_origins = [item.strip() for item in os.environ.get("TELUGU_CORS_ORIGINS", "").split(",") if item.strip() and item.strip() != "*"]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
app.include_router(asr_router)
app.include_router(detect_router)
app.include_router(stt_router)
app.include_router(tts_router)


def _validation_error(exc: RequestValidationError) -> tuple[str, str]:
    errors = exc.errors()
    first = errors[0] if errors else {}
    kind = str(first.get("type", ""))
    loc = [str(part) for part in first.get("loc", []) if part not in {"body", "query", "path"}]
    field = loc[-1] if loc else "request"
    if kind in {"missing", "string_too_short"}:
        if field == "text":
            return "invalid_text", "Text is required."
        return "invalid_request", f"{field} is required."
    if kind == "string_too_long":
        if field in {"text", "reference_text"}:
            return "invalid_text", "Text must be at most 2000 characters."
        return "invalid_request", f"{field} is too long."
    return "invalid_request", "The request is not valid."


@app.exception_handler(RequestValidationError)
async def reject_invalid_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
    code, message = _validation_error(exc)
    return JSONResponse(status_code=422, content={"detail": {"code": code, "message": message}})


class UnavailableTts:
    name = "tts-unavailable"
    requires_reference = False
    sample_rate_hz = 16000

    def synthesize(self, text: str, reference_audio: Path | None, reference_text: str, output_path: Path) -> Path:
        from telugu_speech.asr.infer import ModelUnavailable

        raise ModelUnavailable("No local TTS weights under models/telugu-tts. Run scripts/fetch_mms_tts.py once.")


def _model_bytes(path: Path) -> int | None:
    if not path.is_dir():
        return None
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def configure(settings: Settings | None = None) -> FastAPI:
    chosen = settings or Settings.from_env()
    app.state.settings = chosen
    if chosen.asr_backend == "faster_whisper":
        app.state.asr_backend = FasterWhisperBackend(chosen.whisper_model)
    else:
        app.state.asr_backend = IndicConformerBackend(
            "ai4bharat/indicconformer_stt_te_hybrid_ctc_rnnt_large",
            chosen.asr_decoding,
        )
    micro = Path("models/telugu-tts-micro")
    full = Path("models/telugu-tts")
    try:
        if chosen.tts_backend in {"mms-micro", "mms_tts_micro", "compact"} or (
            chosen.tts_backend in {"mms", "mms-tts-tel", "mms_tts"} and micro.is_dir()
        ):
            app.state.tts_backend = build_tts_backend("mms-micro", str(micro if micro.is_dir() else full))
        elif full.is_dir():
            app.state.tts_backend = build_tts_backend("mms", str(full))
        else:
            app.state.tts_backend = UnavailableTts()
    except Exception:
        app.state.tts_backend = UnavailableTts()
    app.state.reference_audio = None
    app.state.output_dir = Path("experiments/tts")
    stt_path = Path(chosen.whisper_model if chosen.asr_backend == "faster_whisper" else "models/telugu-asr-micro")
    app.state.stt_bytes = _model_bytes(stt_path)
    app.state.tts_bytes = _model_bytes(micro if micro.is_dir() else full)
    return app


@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = getattr(app.state, "settings", None)
    auth_mode = settings.auth_mode if settings else "required"
    return HealthResponse(status="ok", auth_mode=auth_mode, version=__version__)


@app.get("/api/v1/models", response_model=list[ModelInfo])
def models() -> list[ModelInfo]:
    stt_path = Path("models/telugu-asr-micro")
    tts_path = Path("models/telugu-tts-micro")
    stt_bytes = _model_bytes(stt_path)
    tts_bytes = _model_bytes(tts_path)
    return [
        ModelInfo(
            id="models/telugu-asr-micro",
            task="asr",
            available=stt_path.is_dir(),
            note=(
                f"Compact Telugu STT int8. Size {stt_bytes} bytes."
                if stt_bytes is not None
                else "Run scripts/compress_models.py"
            ),
        ),
        ModelInfo(
            id="lite/models/hima-tts-nano-te",
            task="tts",
            available=(Path("lite/models/hima-tts-nano-te/onnx/voices/step_int8.onnx")).is_file(),
            note="Low-memory Telugu TTS. Trained voices telugu_srinivas and telugu_sravani. License: other.",
        ),
        ModelInfo(
            id="models/telugu-tts-micro",
            task="tts",
            available=False,
            note=(
                "Older MMS Telugu files are on disk. CC-BY-NC-4.0, so they are not a hosted voice."
                if tts_bytes is not None
                else "Older MMS pack is absent. Hosted voices are Srinivas and Sravani."
            ),
        ),
        ModelInfo(
            id="models/whisper-multilingual-tiny",
            task="detection",
            available=(Path("models/whisper-multilingual-tiny/model.bin")).is_file(),
            note="Multilingual tiny, int8, about 75 MB. The heavier base model is not loaded. te-en is unsupported. No detection F1 is claimed.",
        ),
    ]


DASHBOARD = Path(__file__).resolve().parents[3] / "index.html"


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    if not DASHBOARD.is_file():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail={"code": "dashboard_missing", "message": "Dashboard file is missing."})
    return FileResponse(DASHBOARD, media_type="text/html")


configure()

if os.environ.get("BHASHA_ENABLE_PLATFORM", "1") == "1":
    repo = Path(__file__).resolve().parents[3]
    platform_src = repo / "platform" / "src"
    if platform_src.is_dir():
        sys.path.insert(0, str(platform_src))
        try:
            from bhasha_platform.api import attach

            attach(app)
        except Exception:
            # Platform is optional; lab routes still work.
            pass


def main() -> None:
    import uvicorn

    host = os.environ.get("TELUGU_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get("TELUGU_PORT", "8000"))
    if port < 1 or port > 65535:
        raise ValueError("TELUGU_PORT must be between 1 and 65535")
    uvicorn.run(
        "telugu_speech.api.main:app",
        host=host,
        port=port,
        workers=1,
        backlog=1024,
        timeout_keep_alive=30,
        limit_concurrency=int(os.environ.get("TELUGU_LIMIT_CONCURRENCY", "256")),
    )
