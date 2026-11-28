"""FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import sys

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from telugu_speech import __version__
from telugu_speech.api.auth import Settings, authorize_any
from telugu_speech.core.config import ConfigError, get_config, reload_config
from telugu_speech.core.fallback import (
    TTS_MODEL_ALIASES,
    VOICE_ALIASES,
    clear_used,
    component_status,
    get_used,
)
from telugu_speech.core.registry import get_registry, reload_registry
from telugu_speech.api.routes_asr import router as asr_router
from telugu_speech.api.routes_detect import router as detect_router
from telugu_speech.api.routes_openai import router as openai_router
from telugu_speech.api.routes_stt import router as stt_router
from telugu_speech.api.routes_tts import router as tts_router
from telugu_speech.api.schemas import ModelInfo
from telugu_speech.asr.infer import FasterWhisperBackend, IndicConformerBackend, decoding_from_env
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
# v2.1: CORS comes from the validated effective config (configs/vaani.yaml
# + VAANI_CORS_ORIGINS / TELUGU_CORS_ORIGINS). A bad value fails startup
# here with a ConfigError naming the key — it is never silently ignored.
# CORS changes need a restart: middleware is fixed at app creation.
_vaani_config = get_config()
_cors_origins = [origin for origin in _vaani_config.cors_origins if origin != "*"]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
app.include_router(asr_router)
app.include_router(detect_router)
app.include_router(openai_router)
app.include_router(stt_router)
app.include_router(tts_router)


@app.middleware("http")
async def _vaani_used_headers(request: Request, call_next):
    """Stamp X-Vaani-Model / X-Vaani-Voice on every response that ran a model.

    Source 1: the core pool's contextvar (set by EnginePool.synthesize /
    transcribe) — the authoritative record of what actually ran, including
    which fallback-chain entry was used.
    Source 2: for POST /v1/audio/speech, the request JSON itself (the
    /v1 routes still use the legacy backends directly, not the pool, so
    the contextvar is empty for them until those routes migrate to the
    pool — see docs/CONFIGURATION.md). The header then reports the
    REQUESTED model/voice, mapped to canonical ids; it never invents a
    different engine. Transcriptions carry no X-Vaani-Model header yet:
    that route will get it from the pool (model + attempted) once it
    migrates to EnginePool.transcribe — see docs/CONFIGURATION.md §6.

    NOTE (VM-only fix, 2026-11-21 verification): the request body MUST be
    read BEFORE call_next. Reading it after raises RuntimeError("Stream
    consumed") in this Starlette version — the endpoint already consumed
    the stream — and the old code swallowed that in a bare except, so
    X-Vaani-* headers were NEVER stamped on /v1/audio/speech responses.
    Reading before call_next is safe: Starlette caches the body and the
    endpoint reads from the cache.
    """
    import json as _json

    body_json = None
    if request.url.path == "/v1/audio/speech" and request.method == "POST":
        try:
            body_json = _json.loads(await request.body())
        except Exception:
            body_json = None
    clear_used()
    response = await call_next(request)
    used = get_used()
    model = used.get("model")
    voice = used.get("voice")
    if model is None and body_json is not None:
        model = TTS_MODEL_ALIASES.get(str(body_json.get("model", "")).strip())
        voice = VOICE_ALIASES.get(str(body_json.get("voice", "")).strip().lower())
    if model:
        response.headers["X-Vaani-Model"] = str(model)
    if voice:
        response.headers["X-Vaani-Voice"] = str(voice)
    return response


def _validation_error(exc: RequestValidationError) -> tuple[str, str]:
    errors = exc.errors()
    first = errors[0] if errors else {}
    kind = str(first.get("type", ""))
    loc = [str(part) for part in first.get("loc", []) if part not in {"body", "query", "path"}]
    field = loc[-1] if loc else "request"
    if kind in {"missing", "string_too_short"}:
        if field == "text":
            return "invalid_text", "Text is required."
        if field == "input":
            return "invalid_text", "Input is required."
        return "invalid_request", f"{field} is required."
    if kind == "string_too_long":
        if field in {"text", "reference_text"}:
            return "invalid_text", "Text must be at most 2000 characters."
        if field == "input":
            return "invalid_text", "Input must be at most 4096 characters."
        return "invalid_request", f"{field} is too long."
    if field == "speed":
        return "invalid_request", "speed must be between 0.25 and 4.0."
    return "invalid_request", "The request is not valid."


@app.exception_handler(RequestValidationError)
async def reject_invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    code, message = _validation_error(exc)
    if request.url.path.startswith("/v1/"):
        # OpenAI-compatible routes render validation failures in the
        # OpenAI error shape with status 400. /api/v1/* is untouched.
        return JSONResponse(
            status_code=400,
            content={"error": {"message": message, "type": "invalid_request_error", "code": code}},
        )
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
    # v2.1: validate the effective config NOW, at startup. A bad key fails
    # startup with a ConfigError naming the key/value — never a silent
    # default. The validated config is stored for /v1/config and /v1/reload.
    vaani_cfg = get_config()
    app.state.vaani_config = vaani_cfg
    logging.getLogger("vaani").setLevel(vaani_cfg.log_level)
    # Apply the configured rate limit + upload cap to the /v1 routes
    # (their module-level defaults are 60/min and 25 MB; the config file
    # is the declared value, this makes the running server match it).
    from telugu_speech.api import routes_openai as _openai_routes

    _openai_routes._RATE_LIMITER._limit = vaani_cfg.rate_limit_per_minute
    _openai_routes._OPENAI_UPLOAD_MAX_BYTES = vaani_cfg.max_audio_bytes
    if chosen.asr_backend == "faster_whisper":
        # decoding_from_env() applies VAANI_ASR_BEAM / TEMPERATURE / VAD /
        # NO_SPEECH_THRESHOLD when set; otherwise the measured defaults
        # (beam 5, temp 0.0, VAD on, language locked te). Previously the
        # defaults were used unconditionally and the env vars were dead.
        app.state.asr_backend = FasterWhisperBackend(chosen.whisper_model, decoding=decoding_from_env())
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


@app.get("/api/v1/health")
def health() -> dict:
    """Liveness + per-component status (v2.1).

    Keeps the old fields (status/auth_mode/version) and adds
    ``components`` from core/fallback.component_status(): asr chain,
    tts voices, registry rejections, config. Status is "ok" only when
    every component is ok; "degraded" when something is missing but the
    default voice + at least one ASR backend work; "unavailable" when
    nothing works. Health probes never load a model. NOTE: with the
    shipped zero-sha256 manifests the registry rejects both packs, so
    this honestly reports "degraded" (legacy fallback still serves both
    voices) until Mani fills the real hash — see configs/voices/README.md.
    """
    settings = getattr(app.state, "settings", None)
    auth_mode = settings.auth_mode if settings else "required"
    cfg = getattr(app.state, "vaani_config", None)
    if cfg is None:
        try:
            cfg = get_config()
        except ConfigError:
            cfg = None
    legacy: list[dict] = []
    try:
        from telugu_speech.tts.voices import list_installed_voices

        legacy = list_installed_voices()
    except Exception:
        legacy = []
    registry = None
    pool = None
    try:
        registry = get_registry()
    except Exception:
        registry = None
    try:
        from telugu_speech.core.engine import get_pool

        pool = get_pool()
    except Exception:
        pool = None
    components = component_status(pool=pool, registry=registry, config=cfg, legacy_voices=legacy)
    return {
        "status": components["overall"],
        "auth_mode": auth_mode,
        "version": __version__,
        "components": components,
    }


@app.get("/health")
def liveness(request: Request) -> dict:
    """Liveness alias, no auth — same shape docs/API_V2.md promises.

    The handler lives in routes_openai.py, but that router carries the
    "/v1" prefix, so its @router.get("/health") mounts at /v1/health,
    not /health — that is why GET /health 404'd. This route delegates
    to that one handler instead of duplicating its logic.
    """
    from telugu_speech.api.routes_openai import health as _openai_health

    return _openai_health(request)


def _v1_auth(request: Request, authorization: str | None, x_api_key: str | None) -> str | JSONResponse:
    """Auth for the /v1 admin endpoints. Returns the key, or a JSONResponse error."""
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "Server is not configured yet.", "type": "server_error", "code": "not_configured"}},
        )
    try:
        return authorize_any(settings, authorization, x_api_key)
    except PermissionError as exc:
        if str(exc) == "api_keys_not_configured":
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "No API keys are configured on this server.", "type": "server_error", "code": "api_keys_not_configured"}},
            )
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Incorrect API key provided.", "type": "authentication_error", "code": "invalid_api_key"}},
        )


@app.post("/v1/reload")
def reload_server(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> JSONResponse:
    """Hot reload without restarting the server (v2.1, auth required).

    Re-reads configs/vaani.yaml + env, re-discovers voice packs, applies
    the config to the live engine pool, and clears the /v1 routes'
    cached voice list so new voices appear immediately. A BAD reload
    never kills the server: the last good config keeps serving and the
    response carries reloaded=false + the error. Loaded engines are kept
    (lazy-load on next use); voices that disappeared from the registry
    are unloaded. Slot-count changes apply only when no slot is in use
    (see EnginePool.apply_config) and are reported in ``warnings``.
    """
    key = _v1_auth(request, authorization, x_api_key)
    if isinstance(key, JSONResponse):
        return key
    errors: list[str] = []
    warnings: list[str] = []
    try:
        cfg = reload_config()
    except ConfigError as exc:
        current = get_config()
        return JSONResponse(
            status_code=400,
            content={
                "reloaded": False,
                "errors": [str(exc)],
                "warnings": [],
                "voices": [],
                "models": [],
                "config": current.redacted(),
            },
        )
    try:
        reg_status = reload_registry(extra_dirs=cfg.voices_dirs)
    except Exception as exc:  # noqa: BLE001 — reload must report, not crash
        reg_status = {"voices": [], "rejected": [{"pack_dir": "registry", "reason": str(exc)}]}
        errors.append(f"registry reload failed: {exc}")
    # Clear the /v1 routes' cached voice list + apply the new limits.
    try:
        from telugu_speech.api import routes_openai as _openai_routes

        _openai_routes._REGISTRY_STATE.update({"tried": False, "ok": False})
        _openai_routes._REGISTRY_STATE.pop("packs", None)
        _openai_routes._RATE_LIMITER._limit = cfg.rate_limit_per_minute
        _openai_routes._OPENAI_UPLOAD_MAX_BYTES = cfg.max_audio_bytes
    except Exception as exc:  # noqa: BLE001
        errors.append(f"could not refresh /v1 route state: {exc}")
    # Apply the config to the live pool; unload voices that are gone.
    try:
        from telugu_speech.core.engine import get_pool

        pool = get_pool()
        warnings.extend(pool.apply_config(cfg))
        kept = {str(v.get("id")) for v in reg_status.get("voices", [])}
        for voice_id in pool.loaded_voices():
            if voice_id not in kept:
                pool.unload_voice(voice_id)
                warnings.append(f"unloaded voice {voice_id!r}: no longer registered")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"could not apply config to engine pool: {exc}")
    request.app.state.vaani_config = cfg
    voices = [str(v.get("id")) for v in reg_status.get("voices", [])]
    return JSONResponse(
        content={
            "reloaded": True,
            "errors": errors,
            "warnings": warnings,
            "voices": voices,
            "models": [f"vaani-tts-{e}" for e in cfg.tts_chain] + [f"vaani-asr-{m}" for m in cfg.asr_chain],
            "rejected": reg_status.get("rejected", []),
            "config": cfg.redacted(),
        },
    )


@app.get("/v1/config")
def effective_config(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> JSONResponse:
    """The effective config, secrets redacted (v2.1, auth required).

    Each key also reports its source: env:VAANI_X, file, or default —
    so a developer can see exactly why a value is what it is.
    """
    key = _v1_auth(request, authorization, x_api_key)
    if isinstance(key, JSONResponse):
        return key
    try:
        cfg = get_config()
    except ConfigError as exc:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "invalid_request_error", "code": "config_invalid"}})
    return JSONResponse(content=cfg.redacted())


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

    # v2.1: host/port/concurrency come from the validated effective config
    # (configs/vaani.yaml + VAANI_*/TELUGU_* env). A bad value already
    # failed at import/configure time with a ConfigError naming the key.
    cfg = get_config()
    uvicorn.run(
        "telugu_speech.api.main:app",
        host=cfg.host,
        port=cfg.port,
        workers=1,
        backlog=1024,
        timeout_keep_alive=30,
        limit_concurrency=cfg.limit_concurrency,
    )
