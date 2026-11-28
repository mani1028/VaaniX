"""Vaani X core errors — OpenAI-shaped, never a bare traceback to a client.

Every error raised by the core carries:
  * ``code`` — a stable machine-readable code (e.g. ``voice_not_found``)
  * ``message`` — a human-readable message naming the missing field/file
  * ``http_status`` — the HTTP status the API layer should return
  * ``to_openai()`` — ``{"error": {"message": ..., "type": ..., "code": ...}}``

The API layer (not this package) maps these to HTTP responses. The core
itself never imports FastAPI.
"""

from __future__ import annotations


class VaaniError(Exception):
    """Base error for everything Vaani X core raises."""

    code = "vaani_error"
    http_status = 500

    def __init__(self, message: str, *, code: str | None = None,
                 http_status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if http_status is not None:
            self.http_status = http_status

    def to_openai(self) -> dict:
        return {
            "error": {
                "message": self.message,
                "type": self.code,
                "code": self.code,
            }
        }


class VoicePackError(VaaniError):
    """A voice pack is malformed, incomplete, or fails checksum validation.

    The message MUST name the pack directory and the missing field/file,
    e.g. ``voice pack 'lite/models/hima-tts-nano-te': manifest.yaml is
    missing required field 'sha256'``. A bad pack is rejected; it never
    crashes the server — the registry logs it and continues with the
    remaining packs.
    """

    code = "voice_pack_invalid"
    http_status = 400


class VoiceNotFoundError(VaaniError):
    """Requested voice id is not registered."""

    code = "voice_not_found"
    http_status = 404


class EngineNotLoadedError(VaaniError):
    """A backend was requested before its engine was loaded."""

    code = "engine_not_loaded"
    http_status = 503


class TextValidationError(VaaniError):
    """Input text failed validation (empty, oversize, unusable)."""

    code = "invalid_text"
    http_status = 400


class AudioValidationError(VaaniError):
    """Input audio failed validation (missing, empty, oversize, bad type)."""

    code = "invalid_audio"
    http_status = 400


class EngineBusyError(VaaniError):
    """All engine slots are in use (semaphore exhausted)."""

    code = "engine_busy"
    http_status = 429


class UnsupportedEngineError(VaaniError):
    """A manifest names an engine the core has no provider for."""

    code = "unsupported_engine"
    http_status = 400


# Backwards-compatible alias: older code may import ``VaaniXError``.
VaaniXError = VaaniError
