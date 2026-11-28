"""Vaani X core — the engine port. Voices are USB connectors; this is the port.

Importing ``telugu_speech.core`` loads NO model, NO voice engine, and NO
FastAPI app. It only exposes the registry, the engine pool, the text
pipeline, the settings, and the errors. First model load happens lazily,
on the first request for that voice.

Public surface (everything other layers should need):
  * get_registry() / VoiceRegistry / VoicePack — voice discovery
  * get_pool() / EnginePool — TTS + ASR execution, slots, cache
  * prepare_text() / validate_text() / split_sentences() / expand_numbers()
  * settings — environment-driven tunables
  * VaaniError + subclasses — OpenAI-shaped errors
"""

from __future__ import annotations

from .config import settings
from .engine import EnginePool, get_pool, reset_pool
from .errors import (
    AudioValidationError,
    EngineBusyError,
    EngineNotLoadedError,
    TextValidationError,
    UnsupportedEngineError,
    VaaniError,
    VoiceNotFoundError,
    VoicePackError,
)
from .registry import VoicePack, VoiceRegistry, get_registry, reset_registry
from .text import (
    chunk_text,
    contains_latin,
    contains_telugu,
    expand_numbers,
    normalize_text,
    number_to_telugu,
    prepare_text,
    split_sentences,
    validate_text,
)

__all__ = [
    "settings",
    "EnginePool", "get_pool", "reset_pool",
    "VoicePack", "VoiceRegistry", "get_registry", "reset_registry",
    "prepare_text", "validate_text", "split_sentences", "expand_numbers",
    "normalize_text", "number_to_telugu", "chunk_text",
    "contains_telugu", "contains_latin",
    "VaaniError", "VoicePackError", "VoiceNotFoundError",
    "EngineNotLoadedError", "EngineBusyError", "UnsupportedEngineError",
    "TextValidationError", "AudioValidationError",
]

__version__ = "2.0.0"
