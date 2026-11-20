"""Shared WAV upload checks."""

from __future__ import annotations

from pathlib import Path
import wave

from fastapi import HTTPException, UploadFile

MAX_NAME_LENGTH = 200


class UnreadableWav(ValueError):
    """The upload is named like a WAV but the bytes are not a WAV."""


async def read_wav_upload(audio: UploadFile, max_bytes: int) -> bytes:
    payload = await audio.read()
    if len(payload) == 0:
        raise HTTPException(status_code=400, detail={"code": "invalid_audio", "message": "Upload a non-empty WAV file."})
    if len(payload) > max_bytes:
        raise HTTPException(status_code=413, detail={"code": "upload_rejected", "message": "Audio exceeds the upload limit."})
    filename = Path(audio.filename or "audio.wav").name
    if not filename or len(filename) > MAX_NAME_LENGTH or not filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail={"code": "invalid_audio", "message": "Upload a WAV file."})
    return payload


def require_readable_wav(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getnchannels() < 1 or handle.getframerate() <= 0:
                raise wave.Error("invalid wav")
    except (wave.Error, EOFError) as exc:
        raise UnreadableWav("Upload a readable WAV file.") from exc
