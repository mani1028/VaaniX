"""Runtime settings. Protected routes deny access unless explicitly opened for local use."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    auth_mode: str
    api_keys: frozenset[str]
    max_upload_bytes: int
    asr_backend: str
    asr_decoding: str
    whisper_model: str
    tts_backend: str

    @classmethod
    def from_env(cls) -> "Settings":
        auth_mode = os.environ.get("TELUGU_SPEECH_AUTH", "required").strip() or "required"
        if auth_mode not in {"required", "local_open"}:
            raise ValueError("TELUGU_SPEECH_AUTH must be required or local_open")
        raw_keys = os.environ.get("TELUGU_API_KEYS", "")
        keys = frozenset(part.strip() for part in raw_keys.split(",") if part.strip())
        max_upload = int(os.environ.get("TELUGU_MAX_UPLOAD_BYTES", "20000000"))
        if max_upload < 1 or max_upload > 50_000_000:
            raise ValueError("TELUGU_MAX_UPLOAD_BYTES must be between 1 and 50000000")
        return cls(
            auth_mode=auth_mode,
            api_keys=keys,
            max_upload_bytes=max_upload,
            asr_backend=os.environ.get("TELUGU_ASR_BACKEND", "faster_whisper"),
            asr_decoding=os.environ.get("TELUGU_ASR_DECODING", "ctc"),
            whisper_model=os.environ.get("TELUGU_WHISPER_MODEL", "models/telugu-asr-micro"),
            tts_backend=os.environ.get("TELUGU_TTS_BACKEND", "mms-micro"),
        )


def authorize(settings: Settings, authorization_header: str | None) -> None:
    if settings.auth_mode == "local_open":
        return
    if not settings.api_keys:
        raise PermissionError("api_keys_not_configured")
    if not authorization_header or not authorization_header.startswith("Bearer "):
        raise PermissionError("missing_credentials")
    token = authorization_header.removeprefix("Bearer ").strip()
    if token not in settings.api_keys:
        raise PermissionError("invalid_credentials")
