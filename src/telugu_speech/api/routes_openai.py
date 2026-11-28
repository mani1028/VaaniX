"""OpenAI-compatible audio routes (Vaani X API v2, ``/v1/*``).

Drop-in replacement surface for OpenAI's audio API, backed by the local
Telugu engines — no cloud, no new model downloads:

* ``POST /v1/audio/speech`` — TTS, audio bytes out (OpenAI shape).
* ``POST /v1/audio/transcriptions`` — ASR, multipart upload (OpenAI shape).
* ``GET /v1/models`` / ``GET /v1/audio/voices`` — model and voice catalogues.
* ``GET /health`` — no key; liveness plus loaded voices/models/slots.

Honest limits, all documented in ``docs/API_V2.md``:

* All ``vaani-tts-*`` model names map to the one installed local engine
  (Hima ONNX int8, ``lite/models/hima-tts-nano-te``). Piper and IndicF5
  are future tiers, not installed — the names are aliases, not engines.
* ``speed`` is accepted (0.25–4.0) and currently a no-op: the local
  engine speaks at its natural rate.
* ``mp3``/``opus``/``flac`` output needs ``ffmpeg`` on the host.
  Without it only ``wav`` works and other formats return 400.
* ASR is VAD-segmented **batch** decode. There is no fake streaming:
  the old ``/api/v1/stt/stream`` websocket is unchanged and still
  answers a whole chunk with ``streaming: false``.
* Segment timestamps in ``verbose_json`` come from faster-whisper when
  it is the active ASR backend. Other backends return one segment
  covering the whole clip, with duration read from the WAV header.

Errors on ``/v1/*`` use the OpenAI shape::

    {"error": {"message": ..., "type": ..., "code": ...}}

Existing ``/api/v1/*`` routes keep their own error shape untouched.
"""

from __future__ import annotations

import base64
from collections import OrderedDict, deque
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import wave

import anyio
import numpy as np
from fastapi import APIRouter, File, Form, Header, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from telugu_speech import __version__
from telugu_speech.api.auth import authorize_any
from telugu_speech.api.schemas import OpenAISpeechRequest
from telugu_speech.asr.infer import FasterWhisperBackend, ModelUnavailable, accepted_transcript
from telugu_speech.text.normalization import contains_telugu, prepare_tts_text
from telugu_speech.tts.chunks import speech_pieces
from telugu_speech.tts.voices import backend_for_voice, is_known_voice, list_installed_voices

router = APIRouter(prefix="/v1", tags=["openai-compatible"])

_OPENAI_UPLOAD_MAX_BYTES = 25_000_000  # OpenAI's documented upload cap.
_RATE_LIMIT_PER_MINUTE = 2000
_RATE_WINDOW_SECONDS = 60.0
_STARTED_AT = int(time.time())

_TTS_MODELS = ("vaani-tts-piper", "vaani-tts-hima", "vaani-tts")
_ASR_MODELS = ("vaani-asr-small", "vaani-asr")
_AUDIO_FORMATS = ("mp3", "wav", "opus", "flac")
_AUDIO_MIME = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/opus",
    "flac": "audio/flac",
}
_TRANSCRIPTION_FORMATS = ("json", "verbose_json", "srt", "vtt", "text")
_UPLOAD_EXTENSIONS = {".wav", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".webm", ".flac", ".ogg"}

# Short voice aliases, so OpenAI-style voice names work. The canonical
# ids are the installed pack ids (telugu_srinivas / telugu_sravani).
_VOICE_ALIASES = {
    "srinivas": "telugu_srinivas",
    "telugu_srinivas": "telugu_srinivas",
    "sravani": "telugu_sravani",
    "telugu_sravani": "telugu_sravani",
    "telugu_default": "telugu_srinivas",
}


class _ApiError(Exception):
    """An error that must be rendered in the OpenAI shape."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        err_type: str = "invalid_request_error",
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.err_type = err_type
        self.headers = headers or {}


def _error_response(exc: _ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status,
        content={"error": {"message": exc.message, "type": exc.err_type, "code": exc.code}},
        headers=exc.headers,
    )


# ---------------------------------------------------------------------------
# Auth + rate limiting (per-key, in-memory).
# Production must still put TLS and real quotas on a proxy: this process
# does not terminate TLS and an in-memory limiter does not survive a
# restart or scale past one process. See docs/API_V2.md.
# ---------------------------------------------------------------------------


def _authenticate(
    request: Request,
    authorization: str | None,
    x_api_key: str | None,
) -> str:
    """Return the accepted key (rate-limit identity) or raise _ApiError."""
    try:
        host = request.client.host if request.client else None
        return authorize_any(request.app.state.settings, authorization, x_api_key, host)
    except PermissionError as exc:
        code = str(exc)
        if code == "api_keys_not_configured":
            raise _ApiError(
                503,
                "api_keys_not_configured",
                "No API keys are configured on this server.",
                "server_error",
            ) from exc
        raise _ApiError(
            401,
            "invalid_api_key",
            "Incorrect API key provided. Send Authorization: Bearer <key> or X-API-Key: <key>.",
            "authentication_error",
        ) from exc


class _RateLimiter:
    def __init__(self, limit: int, window_seconds: float) -> None:
        self._limit = limit
        self._window = window_seconds
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def retry_after(self, key: str) -> float | None:
        """None when the call is allowed; seconds to wait when limited."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= now - self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                return max(0.0, hits[0] + self._window - now)
            hits.append(now)
            return None


_RATE_LIMITER = _RateLimiter(_RATE_LIMIT_PER_MINUTE, _RATE_WINDOW_SECONDS)


class _SpeechGate:
    """Two synthesis slots and a fixed waiting line. A full line is refused."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self._waiting = 0
        self._cv = threading.Condition(self._lock)

    def enter(self, slots: int, queue: int) -> None:
        with self._cv:
            if self._active + self._waiting >= slots + queue:
                raise _ApiError(
                    503,
                    "engine_busy",
                    "Speech queue is full. Retry after 5 seconds.",
                    "server_error",
                    headers={"Retry-After": "5"},
                )
            self._waiting += 1
            try:
                while self._active >= slots:
                    if not self._cv.wait(timeout=30):
                        raise _ApiError(
                            503,
                            "engine_busy",
                            "Speech queue timed out. Retry after 5 seconds.",
                            "server_error",
                            headers={"Retry-After": "5"},
                        )
            except BaseException:
                self._waiting -= 1
                self._cv.notify()
                raise
            self._waiting -= 1
            self._active += 1

    def leave(self) -> None:
        with self._cv:
            self._active = max(0, self._active - 1)
            self._cv.notify()


class _SpeechCache:
    def __init__(self) -> None:
        self._items: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            audio = self._items.get(key)
            if audio is not None:
                self._items.move_to_end(key)
            return audio

    def put(self, key: str, audio: bytes, entries: int, max_bytes: int) -> None:
        if entries <= 0 or max_bytes <= 0:
            return
        with self._lock:
            self._items[key] = audio
            self._items.move_to_end(key)
            while len(self._items) > entries or sum(len(item) for item in self._items.values()) > max_bytes:
                self._items.popitem(last=False)


_SPEECH_GATE = _SpeechGate()
_SPEECH_CACHE = _SpeechCache()


def _check_rate_limit(key: str) -> None:
    retry_after = _RATE_LIMITER.retry_after(key)
    if retry_after is not None:
        raise _ApiError(
            429,
            "rate_limit_exceeded",
            f"Rate limit exceeded: {_RATE_LIMITER._limit} requests per minute per key. Retry after {retry_after:.0f} seconds.",
            "rate_limit_error",
            headers={"Retry-After": str(max(1, int(retry_after + 0.5)))},
        )


# ---------------------------------------------------------------------------
# Voice registry (Part 1 contract) with a fallback to the installed pack.
#
# Part 1 builds telugu_speech.core.registry.VoiceRegistry. If it is not
# there yet, the server still starts and voices come from tts/voices.py.
# The registry outcome is cached after the first call so /health stays
# cheap; a server restart picks up a registry that lands later.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _VoicePack:
    id: str
    name: str
    gender: str
    language: str
    engine: str
    sample_rate: int
    licence: str


_REGISTRY_STATE: dict[str, object] = {"tried": False, "ok": False}


def _registry_packs() -> list[_VoicePack] | None:
    if not _REGISTRY_STATE["tried"]:
        _REGISTRY_STATE["tried"] = True
        try:
            from telugu_speech.core.registry import VoiceRegistry  # type: ignore[import-not-found]

            packs = [
                _VoicePack(
                    id=str(pack.id),
                    name=str(pack.name),
                    gender=str(pack.gender),
                    language=str(pack.language),
                    engine=str(pack.engine),
                    sample_rate=int(pack.sample_rate),
                    licence=str(pack.licence),
                )
                for pack in VoiceRegistry().list()
            ]
            _REGISTRY_STATE["packs"] = packs
            _REGISTRY_STATE["ok"] = True
        except Exception:
            _REGISTRY_STATE["ok"] = False
    if _REGISTRY_STATE.get("ok"):
        return list(_REGISTRY_STATE["packs"])  # type: ignore[arg-type]
    return None


def _list_voice_packs() -> list[_VoicePack]:
    packs = _registry_packs()
    if packs is not None:
        return packs
    installed = list_installed_voices()
    return [
        _VoicePack(
            id=str(item["id"]),
            name=str(item.get("name") or item["id"]),
            gender=str(item.get("gender", "unknown")),
            language="te",
            engine=str(item.get("backend", "hima-onnx-int8")),
            sample_rate=24000,
            licence=str(item.get("license", "other")),
        )
        for item in installed
    ]


# ---------------------------------------------------------------------------
# TTS: POST /v1/audio/speech
# ---------------------------------------------------------------------------


def _resolve_tts_voice(requested: str) -> str:
    canonical = _VOICE_ALIASES.get(requested.strip().lower())
    if canonical is None:
        raise _ApiError(
            400,
            "invalid_voice",
            f"Unknown voice: {requested!r}. Available voices: srinivas, sravani.",
        )
    if not is_known_voice(canonical):
        raise _ApiError(503, "model_unavailable", f"Voice {canonical} is not installed.", "server_error")
    return canonical


def _checked_tts_backend(body: OpenAISpeechRequest):
    if body.model not in _TTS_MODELS:
        raise _ApiError(
            404,
            "model_not_found",
            f"Unknown model: {body.model!r}. Available models: {', '.join(_TTS_MODELS)}.",
        )
    if body.response_format not in _AUDIO_FORMATS:
        raise _ApiError(
            400,
            "invalid_request",
            f"response_format must be one of {', '.join(_AUDIO_FORMATS)}.",
        )
    if not contains_telugu(body.input):
        raise _ApiError(400, "invalid_text", "Text must include Telugu letters.")
    voice_id = _resolve_tts_voice(body.voice)
    try:
        backend = backend_for_voice(voice_id)
    except ModelUnavailable as exc:
        raise _ApiError(503, "model_unavailable", str(exc), "server_error") from exc
    if backend is None:
        raise _ApiError(503, "model_unavailable", f"Voice {voice_id} is not installed.", "server_error")
    return backend, prepare_tts_text(body.input), voice_id


def _transcode_wav(wav_path: Path, fmt: str) -> bytes:
    """Convert a WAV file with ffmpeg. Raises _ApiError when unavailable."""
    if shutil.which("ffmpeg") is None:
        raise _ApiError(
            400,
            "unsupported_format",
            f"{fmt} output needs ffmpeg installed on the host. Install ffmpeg or request response_format=wav.",
        )
    codecs = {"mp3": ["-c:a", "libmp3lame", "-q:a", "4"], "opus": ["-c:a", "libopus", "-b:a", "64k"], "flac": ["-c:a", "flac"]}
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / f"speech.{fmt}"
        try:
            completed = subprocess.run(
                ["ffmpeg", "-y", "-i", str(wav_path), "-vn", *codecs[fmt], str(target)],
                capture_output=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise _ApiError(500, "transcode_failed", f"Could not convert audio to {fmt}.", "server_error") from exc
        if completed.returncode != 0 or not target.is_file():
            raise _ApiError(500, "transcode_failed", f"Could not convert audio to {fmt}.", "server_error")
        return target.read_bytes()


def _pcm_to_wav_bytes(audio: object, sample_rate: int) -> bytes:
    samples = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16).tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def _speech_pieces_guarded(body: OpenAISpeechRequest):
    """Speak one piece at a time and free the slot between pieces.

    A minute of text is several ~180-character pieces. Holding the slot
    for the whole minute blocks every other site. Entering only for the
    piece that is actually synthesizing lets a short request run in the gap.
    """
    from telugu_speech.core.config import get_config

    backend, spoken, voice_id = _checked_tts_backend(body)
    cfg = get_config()
    sample_rate = int(getattr(backend, "sample_rate_hz", 24000))
    pieces = speech_pieces(spoken, cfg.max_chunk_chars)
    for index, piece in enumerate(pieces):
        _SPEECH_GATE.enter(cfg.tts_slots, cfg.tts_queue)
        try:
            try:
                parts = [np.asarray(part, dtype=np.float32).reshape(-1) for part in backend.iter_pcm(piece)]
                audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
            except ModelUnavailable as exc:
                raise _ApiError(503, "model_unavailable", str(exc), "server_error") from exc
        finally:
            _SPEECH_GATE.leave()
        yield index, piece, audio, voice_id, sample_rate


def _speech_bytes(body: OpenAISpeechRequest) -> tuple[bytes, str, int]:
    """Synthesize the whole input. Returns (audio bytes, mime, sample rate)."""
    from telugu_speech.tts.chunks import write_chunked_speech

    chunks: list[object] = []
    sample_rate = 24000
    for _index, _piece, audio, _voice_id, sample_rate in _speech_pieces_guarded(body):
        chunks.append(audio)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "speech.wav"
        write_chunked_speech(chunks, sample_rate, output)
        if body.response_format == "wav":
            return output.read_bytes(), _AUDIO_MIME["wav"], sample_rate
        return _transcode_wav(output, body.response_format), _AUDIO_MIME[body.response_format], sample_rate


def _speech_sse_chunks(body: OpenAISpeechRequest):
    """Yield one SSE event per sentence piece (Vaani extension).

    OpenAI's speech endpoint has no SSE standard; this is Vaani's own
    streaming shape, documented in docs/API_V2.md. Each event carries
    base64 WAV for one sentence; the stream ends with ``data: [DONE]``.
    The speech slot is held only while that piece is synthesized.
    """
    try:
        for index, text, audio, voice_id, sample_rate in _speech_pieces_guarded(body):
            payload = {
                "audio": base64.b64encode(_pcm_to_wav_bytes(audio, sample_rate)).decode("ascii"),
                "index": index,
                "text": text,
                "voice": voice_id,
                "format": "wav",
                "sample_rate": sample_rate,
            }
            yield f"data: {json.dumps(payload)}\n\n"
    except _ApiError as exc:
        yield f"data: {json.dumps({'error': {'message': exc.message, 'type': exc.err_type, 'code': exc.code}})}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/audio/speech")
def speech(
    body: OpenAISpeechRequest,
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    try:
        key = _authenticate(request, authorization, x_api_key)
        _check_rate_limit(key)
        if body.stream and body.response_format != "wav":
            raise _ApiError(
                400,
                "invalid_request",
                "Streaming is WAV only. Use stream=false for mp3/opus/flac, or response_format=wav for streaming.",
            )
        if body.stream:
            _checked_tts_backend(body)
            return StreamingResponse(
                _speech_sse_chunks(body),
                media_type="text/event-stream",
                headers={
                    "X-Sample-Rate": "24000",
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        from telugu_speech.core.config import get_config

        cfg = get_config()
        cache_key = f"{body.voice}|{body.response_format}|{body.input}"
        cached = _SPEECH_CACHE.get(cache_key)
        if cached is not None:
            return Response(
                content=cached,
                media_type=_AUDIO_MIME.get(body.response_format, "application/octet-stream"),
                headers={"X-Sample-Rate": "24000", "X-Vaani-Cache": "HIT"},
            )
        audio_bytes, mime, sample_rate = _speech_bytes(body)
        _SPEECH_CACHE.put(cache_key, audio_bytes, cfg.audio_cache_entries, cfg.audio_cache_max_mb * 1024 * 1024)
        return Response(
            content=audio_bytes,
            media_type=mime,
            headers={"X-Sample-Rate": str(sample_rate), "X-Vaani-Cache": "MISS"},
        )
    except _ApiError as exc:
        return _error_response(exc)


# ---------------------------------------------------------------------------
# ASR: POST /v1/audio/transcriptions
# ---------------------------------------------------------------------------


def _wav_duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            return frames / rate if rate > 0 else 0.0
    except (wave.Error, EOFError, OSError):
        return 0.0


def _transcribe_file(path: Path, backend) -> tuple[str, float, list[dict], list[dict]]:
    """Decode one file. Returns (text, duration, segments, words).

    With faster-whisper, segments and word timestamps come from the
    model itself (VAD-filtered). The decode settings come from the
    backend's DecodingConfig — the single source of truth, shared with
    asr/infer.py — not from literals copied here (the old literals had
    already drifted: they omitted no_speech/logprob thresholds, so the
    two live paths could decode differently). Language stays locked to
    "te" by DecodingConfig validation; word_timestamps stays on because
    accepted_transcript's word-probability filter needs it.
    Any other backend returns the plain transcript as one segment.
    """
    if isinstance(backend, FasterWhisperBackend):
        config = backend.decoding
        model = backend._load()
        segments, info = model.transcribe(
            str(path),
            language=config.language,
            beam_size=config.beam_size,
            temperature=config.temperature,
            vad_filter=config.vad_enabled,
            vad_parameters={
                "threshold": config.vad_threshold,
                "min_speech_duration_ms": config.vad_min_speech_duration_ms,
                "min_silence_duration_ms": config.vad_min_silence_duration_ms,
                "speech_pad_ms": config.vad_speech_pad_ms,
            },
            condition_on_previous_text=config.condition_on_previous_text,
            log_prob_threshold=config.logprob_threshold,
            no_speech_threshold=config.no_speech_threshold,
            compression_ratio_threshold=config.compression_ratio_threshold,
            word_timestamps=True,
            hallucination_silence_threshold=config.hallucination_silence_threshold,
        )
        segment_list = list(segments)
        text = accepted_transcript(segment_list)
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        if duration <= 0.0:
            duration = _wav_duration_seconds(path)
        out_segments: list[dict] = []
        out_words: list[dict] = []
        for segment in segment_list:
            out_segments.append(
                {
                    "id": int(getattr(segment, "id", len(out_segments))),
                    "seek": int(getattr(segment, "seek", 0)),
                    "start": float(getattr(segment, "start", 0.0)),
                    "end": float(getattr(segment, "end", 0.0)),
                    "text": str(getattr(segment, "text", "")),
                    "tokens": [int(token) for token in (getattr(segment, "tokens", None) or [])],
                    "temperature": float(getattr(segment, "temperature", 0.0)),
                    "avg_logprob": float(getattr(segment, "avg_logprob", 0.0)),
                    "compression_ratio": float(getattr(segment, "compression_ratio", 1.0)),
                    "no_speech_prob": float(getattr(segment, "no_speech_prob", 0.0)),
                }
            )
            for word in getattr(segment, "words", None) or []:
                out_words.append(
                    {
                        "word": str(getattr(word, "word", "")).strip(),
                        "start": float(getattr(word, "start", 0.0)),
                        "end": float(getattr(word, "end", 0.0)),
                    }
                )
        return text, duration, out_segments, out_words
    text = backend.transcribe(path)
    duration = _wav_duration_seconds(path)
    segment = {
        "id": 0,
        "seek": 0,
        "start": 0.0,
        "end": duration,
        "text": text,
        "tokens": [],
        "temperature": 0.0,
        "avg_logprob": 0.0,
        "compression_ratio": 1.0,
        "no_speech_prob": 0.0,
    }
    return text, duration, [segment], []


def _srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _vtt_timestamp(seconds: float) -> str:
    return _srt_timestamp(seconds).replace(",", ".")


def _segments_to_srt(segments: list[dict]) -> str:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        lines.append(str(index))
        lines.append(f"{_srt_timestamp(float(segment.get('start', 0.0)))} --> {_srt_timestamp(float(segment.get('end', 0.0)))}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _segments_to_vtt(segments: list[dict]) -> str:
    lines = ["WEBVTT", ""]
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        lines.append(f"{_vtt_timestamp(float(segment.get('start', 0.0)))} --> {_vtt_timestamp(float(segment.get('end', 0.0)))}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


@router.post("/audio/transcriptions")
async def transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(default="vaani-asr-small"),
    language: str = Form(default="te"),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    try:
        key = _authenticate(request, authorization, x_api_key)
        _check_rate_limit(key)
        if model not in _ASR_MODELS:
            raise _ApiError(
                404,
                "model_not_found",
                f"Unknown model: {model!r}. Available models: {', '.join(_ASR_MODELS)}.",
            )
        language = (language or "te").strip().lower()
        if language != "te":
            raise _ApiError(400, "unsupported_language", "Only Telugu (te) is supported.")
        if response_format not in _TRANSCRIPTION_FORMATS:
            raise _ApiError(
                400,
                "invalid_request",
                f"response_format must be one of {', '.join(_TRANSCRIPTION_FORMATS)}.",
            )
        if not 0.0 <= temperature <= 1.0:
            raise _ApiError(400, "invalid_request", "temperature must be between 0.0 and 1.0.")
        # temperature is accepted for OpenAI compatibility; the local
        # decoder always runs at temperature 0.0 (see docs/API_V2.md).
        payload = await file.read()
        if len(payload) == 0:
            raise _ApiError(400, "invalid_audio", "Upload a non-empty audio file.")
        if len(payload) > _OPENAI_UPLOAD_MAX_BYTES:
            raise _ApiError(
                413,
                "file_too_large",
                f"Audio exceeds the 25 MB upload limit ({len(payload)} bytes).",
            )
        filename = Path(file.filename or "audio.wav").name
        suffix = Path(filename).suffix.lower()
        if suffix not in _UPLOAD_EXTENSIONS:
            raise _ApiError(
                400,
                "invalid_audio",
                "Upload an audio file (wav, mp3, m4a, mp4, mpeg, mpga, webm, flac, ogg).",
            )
        backend = request.app.state.asr_backend
        if suffix != ".wav" and not isinstance(backend, FasterWhisperBackend):
            raise _ApiError(
                400,
                "invalid_audio",
                "Only WAV uploads are supported with the active ASR backend. faster-whisper is required for mp3/m4a and other formats.",
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / filename
            path.write_bytes(payload)
            if suffix == ".wav":
                from telugu_speech.api.uploads import UnreadableWav, require_readable_wav

                try:
                    require_readable_wav(path)
                except UnreadableWav as exc:
                    raise _ApiError(400, "invalid_audio", str(exc)) from exc
            try:
                text, duration, segments, words = await anyio.to_thread.run_sync(_transcribe_file, path, backend)
            except ModelUnavailable as exc:
                raise _ApiError(503, "model_unavailable", str(exc), "server_error") from exc
        if response_format == "json":
            return JSONResponse(content={"text": text})
        if response_format == "text":
            return PlainTextResponse(content=text)
        if response_format == "srt":
            return PlainTextResponse(content=_segments_to_srt(segments), media_type="text/plain; charset=utf-8")
        if response_format == "vtt":
            return PlainTextResponse(content=_segments_to_vtt(segments), media_type="text/plain; charset=utf-8")
        return JSONResponse(
            content={
                "task": "transcribe",
                "language": "te",
                "duration": duration,
                "text": text,
                "words": words,
                "segments": segments,
            }
        )
    except _ApiError as exc:
        return _error_response(exc)


# ---------------------------------------------------------------------------
# Catalogues + health
# ---------------------------------------------------------------------------


@router.get("/models")
def list_models(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    try:
        key = _authenticate(request, authorization, x_api_key)
        _check_rate_limit(key)
    except _ApiError as exc:
        return _error_response(exc)
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "created": _STARTED_AT, "owned_by": "vaani"}
            for model_id in (*_TTS_MODELS, *_ASR_MODELS)
        ],
    }


@router.get("/audio/voices")
def list_voices(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    # Authenticated, but not rate-limited: a cheap catalogue call.
    try:
        _authenticate(request, authorization, x_api_key)
    except _ApiError as exc:
        return _error_response(exc)
    return {
        "object": "list",
        "data": [
            {
                "id": pack.id,
                "name": pack.name,
                "gender": pack.gender,
                "language": pack.language,
                "engine": pack.engine,
                "sample_rate": pack.sample_rate,
                "licence": pack.licence,
            }
            for pack in _list_voice_packs()
        ],
    }


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    """No key. Liveness plus what is loaded. No latency or accuracy claims."""
    settings = getattr(request.app.state, "settings", None)
    slots = 2
    try:
        from telugu_speech.core.config import get_config

        slots = get_config().tts_slots
    except Exception:
        slots = 2
    return {
        "status": "ok",
        "version": __version__,
        "auth_mode": settings.auth_mode if settings else "required",
        "voices": [pack.id for pack in _list_voice_packs()],
        "models": [*_TTS_MODELS, *_ASR_MODELS],
        "slots": slots,
        "asr_backend": settings.asr_backend if settings else "faster_whisper",
    }
