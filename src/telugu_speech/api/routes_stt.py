"""Additive STT route and batch-chunk WebSocket.

The socket runs the same batch backend as /api/v1/asr/transcribe.
streaming is false until an RNNT streaming decoder is loaded.
"""

from __future__ import annotations

from pathlib import Path
import json
import tempfile
import wave

import anyio

from fastapi import APIRouter, File, Header, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect

from telugu_speech.api.auth import authorize
from telugu_speech.api.schemas import AsrResponse
from telugu_speech.api.uploads import UnreadableWav, read_wav_upload, require_readable_wav
from telugu_speech.asr.infer import ModelUnavailable
from telugu_speech.text.normalization import NormalizationConfig, has_code_mix, normalize_text
from telugu_speech.text.styles import apply_output_style

router = APIRouter(prefix="/api/v1", tags=["stt"])


def _auth_http(request: Request, authorization: str | None) -> None:
    try:
        host = request.client.host if request.client else None
        authorize(request.app.state.settings, authorization, host)
    except PermissionError as exc:
        code = str(exc)
        status = 503 if code == "api_keys_not_configured" else 401
        raise HTTPException(status_code=status, detail={"code": code, "message": "Authentication failed."}) from exc


def transcribe_wav(request: Request, payload: bytes, style: str) -> AsrResponse:
    backend = request.app.state.asr_backend
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "audio.wav"
        path.write_bytes(payload)
        try:
            require_readable_wav(path)
            hypothesis = backend.transcribe(path)
        except UnreadableWav as exc:
            raise HTTPException(status_code=400, detail={"code": "invalid_audio", "message": str(exc)}) from exc
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": str(exc)}) from exc
    try:
        text, raw_text, norm_text = apply_output_style(hypothesis, style)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"code": "invalid_style", "message": str(exc)}) from exc
    _raw, normalized = normalize_text(hypothesis, NormalizationConfig())
    if normalized != norm_text:
        norm_text = normalized
    return AsrResponse(
        text=text,
        raw_text=raw_text,
        norm_text=norm_text,
        backend=getattr(backend, "name", request.app.state.settings.asr_backend),
        code_mix=has_code_mix(norm_text),
    )


@router.post("/stt", response_model=AsrResponse)
async def stt(
    request: Request,
    audio: UploadFile = File(...),
    style: str = Query(default="codemix"),
    authorization: str | None = Header(default=None),
) -> AsrResponse:
    _auth_http(request, authorization)
    payload = await read_wav_upload(audio, request.app.state.settings.max_upload_bytes)
    return await anyio.to_thread.run_sync(transcribe_wav, request, payload, style)


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


@router.websocket("/stt/stream")
async def stt_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    settings = websocket.app.state.settings
    try:
        host = websocket.client.host if websocket.client else None
        authorize(settings, websocket.headers.get("authorization"), host)
    except PermissionError as exc:
        await websocket.send_json({"code": str(exc), "message": "Authentication failed.", "streaming": False})
        await websocket.close(code=1008)
        return
    chunks = bytearray()
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
            data = message.get("bytes")
            if data:
                if len(chunks) + len(data) > settings.max_upload_bytes:
                    await websocket.send_json({"code": "upload_rejected", "message": "Audio exceeds the upload limit.", "streaming": False})
                    await websocket.close(code=1009)
                    return
                chunks.extend(data)
            text = message.get("text")
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                await websocket.send_json({"code": "invalid_audio", "message": "Control messages must be JSON.", "streaming": False})
                continue
            if not isinstance(event, dict) or event.get("event") not in {"flush", "end"}:
                await websocket.send_json({"code": "invalid_audio", "message": "Send event flush or end.", "streaming": False})
                continue
            if not chunks:
                await websocket.send_json({"code": "invalid_audio", "message": "No audio chunks were received.", "streaming": False})
                continue
            payload = bytes(chunks)
            if not payload.startswith(b"RIFF"):
                sample_rate = int(event.get("sample_rate") or 16000)
                payload = _pcm_to_wav(payload, sample_rate)
            try:
                body = await anyio.to_thread.run_sync(
                    transcribe_wav, websocket, payload, str(event.get("style") or "codemix")
                )
            except HTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, dict) else {"code": "invalid_audio", "message": "Audio was rejected."}
                await websocket.send_json({**detail, "streaming": False})
                continue
            await websocket.send_json(
                {
                    "text": body.text,
                    "raw_text": body.raw_text,
                    "norm_text": body.norm_text,
                    "backend": body.backend,
                    "code_mix": body.code_mix,
                    "streaming": False,
                    "partial": event.get("event") == "flush",
                }
            )
            if event.get("event") == "end":
                await websocket.close()
                return
    except WebSocketDisconnect:
        return
