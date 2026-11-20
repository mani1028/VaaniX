"""Request and response contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AsrResponse(BaseModel):
    text: str
    raw_text: str
    norm_text: str
    backend: str
    code_mix: bool


class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    language: str = "te"
    voice: str = "telugu_default"
    style: str = "conversational"
    format: str = "wav"
    reference_text: str | None = Field(default=None, max_length=2000)


class TtsResponse(BaseModel):
    audio_path: str
    audio_url: str
    backend: str
    sample_rate_hz: int


class ModelInfo(BaseModel):
    id: str
    task: str
    available: bool
    note: str


class HealthResponse(BaseModel):
    status: str
    auth_mode: str
    version: str


class ErrorBody(BaseModel):
    code: str
    message: str
