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


# ---------------------------------------------------------------------------
# OpenAI-compatible shapes (Vaani X API v2, /v1/*).
#
# These mirror the OpenAI audio API so existing OpenAI SDK code works with
# only a base_url change. Fields the local engine cannot honour yet
# (for example TTS ``speed``) are accepted and documented as no-ops in
# docs/API_V2.md instead of being silently dropped or faked.
# ---------------------------------------------------------------------------


class OpenAISpeechRequest(BaseModel):
    """POST /v1/audio/speech JSON body. Mirrors OpenAI's speech request."""

    model: str = "vaani-tts-hima"
    input: str = Field(min_length=1, max_length=4096)
    voice: str = "srinivas"
    response_format: str = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    stream: bool = False


class OpenAIErrorDetail(BaseModel):
    message: str
    type: str
    code: str


class OpenAIErrorResponse(BaseModel):
    error: OpenAIErrorDetail


class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "vaani"


class OpenAIModelList(BaseModel):
    object: str = "list"
    data: list[OpenAIModel]


class OpenAIVoice(BaseModel):
    id: str
    name: str
    gender: str
    language: str = "te"
    engine: str
    sample_rate: int
    licence: str


class OpenAIVoiceList(BaseModel):
    object: str = "list"
    data: list[OpenAIVoice]


class OpenAITranscriptionSegment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int] = []
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0
    no_speech_prob: float = 0.0


class OpenAIVerboseTranscription(BaseModel):
    text: str
    language: str = "te"
    duration: float
    segments: list[OpenAITranscriptionSegment] = []
