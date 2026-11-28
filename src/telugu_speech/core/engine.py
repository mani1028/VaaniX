"""Vaani X core engine pool — one shared ASR, one TTS engine per voice.

Rules that make this low-resource instead of merely small:
  * Engines are LAZY: nothing is loaded at import or at server start.
    The first request for a voice loads that voice's engine, once.
  * ONE TTS engine instance per loaded voice — never one per request.
    Re-loading a 260-470 MB engine per request would OOM the box and
    make latency dominated by model load.
  * ONE shared ASR instance for the whole process.
  * A slot semaphore (default 2, ``VAANI_TTS_SLOTS``) bounds concurrent
    syntheses. Sizing (Part 1 brief, UNMEASURED — TBD until bench):
    1 engine ~260-470 MB, 2 engines ~640-720 MB, so 8 concurrent
    engines would need ~1.5-1.8 GB. The old default of 8 was dangerous
    on a 1-2 GB VPS; 2 is the safe default. Raise only after measuring.
  * ONNX / OMP threads are pinned to ``TELUGU_CPU_THREADS`` (default 2,
    cap 4) BEFORE the first session is created — ORT reads them then.
  * Idle voices are unloaded after ``VAANI_VOICE_IDLE_SECS`` (default
    1800 s; 0 = never unload). ``unload_idle()`` is called on every
    synthesis so an idle voice frees its RAM without a restart.
  * Synthesised audio is cached in a bounded LRU (entries + MB caps).
    The cache key is (voice, style, normalised text hash) — never the
    raw text, so a 4 KB request does not become a 4 KB cache key.

The core imports NO voice engine at module import time. Backends are
resolved by engine name (``hima-nano`` / ``piper`` / ``onnx-vits``) and
imported lazily inside ``_load_backend()``. Adding a new engine means
adding one provider module + one line in ``_BACKENDS`` — the core, the
registry and the API do not change.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol

from .config import ConfigError, EffectiveConfig, apply_thread_env, get_config, settings
from .errors import (
    EngineBusyError,
    EngineNotLoadedError,
    TextValidationError,
    UnsupportedEngineError,
    VoicePackError,
)
from .fallback import (
    ChainResult,
    load_asr_backend,
    resolve_asr_model,
    resolve_tts_engine,
    set_used,
    try_chain,
)
from .registry import VoicePack, VoiceRegistry, get_registry
from .text import prepare_text, validate_text

log = logging.getLogger("vaani.core.engine")

# ---------------------------------------------------------------------------
# Backend protocols — the contract every provider must satisfy
# ---------------------------------------------------------------------------

class TTSBackend(Protocol):
    """A TTS engine. ``synthesize`` returns complete audio bytes (WAV)."""

    def synthesize(self, text: str, voice: VoicePack, style: str = "conversational") -> bytes: ...
    def synthesize_stream(self, text: str, voice: VoicePack,
                          style: str = "conversational") -> Iterator[bytes]: ...
    def close(self) -> None: ...


class ASRBackend(Protocol):
    """An ASR engine. The real backends return a plain ``str`` transcript
    (``asr/infer.py``: ``transcribe(Path) -> str``); ``EnginePool.transcribe``
    wraps that into {text, language, segments, duration}."""

    def transcribe(self, audio_path: str | Path) -> str: ...
    def close(self) -> None: ...


# engine name (manifest) -> "module:Class" — imported lazily, never at startup.
_BACKENDS: dict[str, str] = {
    "hima-nano": "telugu_speech.tts.providers.hima_nano:HimaNanoBackend",
    "piper": "telugu_speech.tts.providers.piper:PiperBackend",
    "onnx-vits": "telugu_speech.tts.providers.onnx_vits:OnnxVitsBackend",
}


def _load_backend(engine_name: str, pack: VoicePack):
    """Import and instantiate the backend for one voice pack (lazy)."""
    target = _BACKENDS.get(engine_name)
    if target is None:
        raise UnsupportedEngineError(
            f"voice '{pack.id}': engine '{engine_name}' has no provider; "
            f"supported engines: {', '.join(sorted(_BACKENDS))}"
        )
    module_name, _, class_name = target.partition(":")
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise EngineNotLoadedError(
            f"voice '{pack.id}': cannot load provider {target}: {exc} — "
            "the provider module is missing; check the install"
        )
    try:
        return cls(pack)
    except Exception as exc:
        raise EngineNotLoadedError(
            f"voice '{pack.id}': provider {target} failed to initialise: {exc}"
        )


# ---------------------------------------------------------------------------
# Loaded-voice bookkeeping + bounded LRU audio cache
# ---------------------------------------------------------------------------

@dataclass
class _LoadedVoice:
    backend: object
    last_used: float = field(default_factory=time.monotonic)


class _AudioCache:
    """Bounded LRU: max entries AND max bytes, whichever hits first."""

    def __init__(self, max_entries: int, max_bytes: int) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._data: OrderedDict[str, bytes] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    @staticmethod
    def key(voice_id: str, style: str, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"{voice_id}:{style}:{digest}"

    def get(self, key: str) -> bytes | None:
        if self.max_entries <= 0:
            return None
        with self._lock:
            value = self._data.get(key)
            if value is not None:
                self._data.move_to_end(key)
            return value

    def put(self, key: str, value: bytes) -> None:
        if self.max_entries <= 0 or not value:
            return
        with self._lock:
            if key in self._data:
                self._bytes -= len(self._data[key])
            self._data[key] = value
            self._bytes += len(value)
            while (len(self._data) > self.max_entries
                   or self._bytes > self.max_bytes) and self._data:
                _, evicted = self._data.popitem(last=False)
                self._bytes -= len(evicted)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._bytes = 0


# ---------------------------------------------------------------------------
# Engine pool
# ---------------------------------------------------------------------------

class EnginePool:
    """Owns the ASR singleton, the per-voice TTS engines, and the slots."""

    def __init__(self, registry: VoiceRegistry | None = None,
                 config: EffectiveConfig | None = None) -> None:
        self.registry = registry or get_registry()
        # v2.1: prefer the validated EffectiveConfig (file + env). If it
        # cannot be loaded (bad vaani.yaml), fall back to the legacy env
        # Settings and log loudly — the strict error is still raised at
        # server startup by api/main.py:configure(), so a broken config
        # file never starts a server silently.
        self.config: EffectiveConfig | None = config
        if self.config is None:
            try:
                self.config = get_config()
            except ConfigError:
                log.warning("effective config is invalid; using legacy env Settings", exc_info=True)
                self.config = None
        cfg = self.config
        self._tts_lock = threading.Lock()
        self._voices: dict[str, _LoadedVoice] = {}
        self._tts_slots_total = cfg.tts_slots if cfg else settings.tts_slots
        self._asr_slots_total = cfg.asr_slots if cfg else settings.asr_slots
        self._tts_slots = threading.Semaphore(self._tts_slots_total)
        self._asr_slots = threading.Semaphore(self._asr_slots_total)
        self._asr: object | None = None
        self._asr_by_model: dict[str, object] = {}
        self._asr_lock = threading.Lock()
        self._threads_applied = False
        self.cache = _AudioCache(
            cfg.audio_cache_entries if cfg else settings.audio_cache_entries,
            (cfg.audio_cache_max_mb if cfg else settings.audio_cache_max_mb) * 1024 * 1024,
        )

    # -- v2.1 config application -------------------------------------------
    def apply_config(self, cfg: EffectiveConfig) -> list[str]:
        """Apply a reloaded config to this live pool. Returns warnings.

        Cache limits, idle timeout, text/audio caps and the ASR chain
        apply immediately. Slot counts apply immediately ONLY when no
        slot is currently in use; otherwise the old counts are kept and
        a warning is returned (a semaphore cannot be resized safely
        mid-flight). Thread counts apply to engines loaded AFTER the
        reload — ONNX reads them at session creation. A changed ASR chain
        takes effect for the next transcription; already-loaded ASR
        backends are kept until the pool is reset.
        """
        warnings: list[str] = []
        old_config = self.config
        self.config = cfg
        # Audio cache: new limits mean a new (empty) cache.
        if (old_config is None
                or old_config.audio_cache_entries != cfg.audio_cache_entries
                or old_config.audio_cache_max_mb != cfg.audio_cache_max_mb):
            self.cache = _AudioCache(cfg.audio_cache_entries, cfg.audio_cache_max_mb * 1024 * 1024)
            warnings.append("audio cache was cleared by the config reload (new cache limits)")
        for label, attr, total_attr, wanted in (
            ("TTS", "_tts_slots", "_tts_slots_total", cfg.tts_slots),
            ("ASR", "_asr_slots", "_asr_slots_total", cfg.asr_slots),
        ):
            current = getattr(self, attr)
            total = getattr(self, total_attr)
            if wanted == total:
                continue
            # A semaphore is safe to replace only when fully free
            # (_value == its initial size). Otherwise keep the old one.
            try:
                free = bool(hasattr(current, "_value") and current._value == total)
            except Exception:
                free = False
            if free:
                setattr(self, attr, threading.Semaphore(wanted))
                setattr(self, total_attr, wanted)
            else:
                warnings.append(
                    f"{label} slots not changed to {wanted}: slots are in use; "
                    "the old count ({total}) stays until the server restarts"
                )
        return warnings

    # -- internals ---------------------------------------------------------
    def _apply_threads_once(self) -> None:
        if not self._threads_applied:
            applied = apply_thread_env()
            log.info("pinned inference threads: %s", applied)
            self._threads_applied = True

    def _get_tts(self, pack: VoicePack):
        with self._tts_lock:
            loaded = self._voices.get(pack.id)
            if loaded is None:
                self._apply_threads_once()
                backend = _load_backend(pack.engine, pack)
                loaded = _LoadedVoice(backend=backend)
                self._voices[pack.id] = loaded
                log.info("loaded TTS engine '%s' for voice '%s'", pack.engine, pack.id)
            loaded.last_used = time.monotonic()
            return loaded.backend

    def unload_idle(self, *, now: float | None = None) -> list[str]:
        """Unload voices idle longer than VAANI_VOICE_IDLE_SECS. Returns ids freed."""
        idle_secs = self.config.voice_idle_secs if self.config else settings.voice_idle_secs
        if idle_secs <= 0:
            return []
        now = time.monotonic() if now is None else now
        freed: list[str] = []
        with self._tts_lock:
            for voice_id, loaded in list(self._voices.items()):
                if now - loaded.last_used >= idle_secs:
                    try:
                        loaded.backend.close()  # type: ignore[attr-defined]
                    except Exception:
                        log.exception("error closing voice '%s'", voice_id)
                    del self._voices[voice_id]
                    freed.append(voice_id)
        if freed:
            log.info("unloaded idle voices: %s", ", ".join(freed))
        return freed

    def unload_voice(self, voice_id: str) -> bool:
        with self._tts_lock:
            loaded = self._voices.pop(voice_id, None)
        if loaded is None:
            return False
        try:
            loaded.backend.close()  # type: ignore[attr-defined]
        except Exception:
            log.exception("error closing voice '%s'", voice_id)
        return True

    def loaded_voices(self) -> list[str]:
        return sorted(self._voices)

    # -- TTS ----------------------------------------------------------------
    def synthesize(self, text: str, voice_id: str,
                   style: str = "conversational",
                   engine: str | None = None) -> bytes:
        """Synthesise full text to WAV bytes. Validates style + text first."""
        audio, _meta = self.synthesize_with_meta(text, voice_id, style=style, engine=engine)
        return audio

    def synthesize_with_meta(self, text: str, voice_id: str,
                             style: str = "conversational",
                             engine: str | None = None) -> tuple[bytes, dict]:
        """Like synthesize(), plus a meta dict {voice, model, engine, cached}.

        ``engine`` is a per-request override: it must match the voice
        pack's engine, else 400 — a request for a Piper voice never
        silently falls back to the Hima engine. The meta dict is what
        the API layer turns into the X-Vaani-Model / X-Vaani-Voice
        response headers, so the caller always knows what actually ran.
        """
        if style not in settings.allowed_styles:
            raise TextValidationError(
                f"style must be one of {', '.join(settings.allowed_styles)}, got {style!r}"
            )
        cleaned = validate_text(text)
        pack = self.registry.get(voice_id)
        if engine is not None:
            wanted = resolve_tts_engine(engine)
            if wanted != pack.engine:
                raise UnsupportedEngineError(
                    f"voice '{pack.id}' runs on engine '{pack.engine}', not '{wanted}' — "
                    "pick a voice that uses that engine, or omit the model override"
                )
        cache_key = self.cache.key(pack.id, style, cleaned)
        cached = self.cache.get(cache_key)
        if cached is not None:
            set_used(model=pack.engine, voice=pack.id)
            return cached, {"voice": pack.id, "model": pack.engine, "engine": pack.engine, "cached": True}
        self.unload_idle()
        if not self._tts_slots.acquire(timeout=30):
            raise EngineBusyError(
                "all TTS slots are busy; retry shortly (Retry-After: 5)"
            )
        try:
            backend = self._get_tts(pack)
            chunks = prepare_text(cleaned)
            parts: list[bytes] = []
            for chunk in chunks:
                # Re-use the streaming path per chunk so long text is
                # produced incrementally even in the non-streaming call.
                for pcm in backend.synthesize_stream(chunk, pack, style):  # type: ignore[attr-defined]
                    parts.append(pcm)
            audio = b"".join(parts)
            if not audio:
                raise EngineNotLoadedError(
                    f"voice '{pack.id}': backend returned no audio for the given text"
                )
            self.cache.put(cache_key, audio)
            set_used(model=pack.engine, voice=pack.id)
            return audio, {"voice": pack.id, "model": pack.engine, "engine": pack.engine, "cached": False}
        finally:
            self._tts_slots.release()

    def synthesize_stream(self, text: str, voice_id: str,
                          style: str = "conversational",
                          engine: str | None = None) -> Iterator[bytes]:
        """Yield PCM/WAV chunks as they are produced, chunk by chunk."""
        if style not in settings.allowed_styles:
            raise TextValidationError(
                f"style must be one of {', '.join(settings.allowed_styles)}, got {style!r}"
            )
        cleaned = validate_text(text)
        pack = self.registry.get(voice_id)
        if engine is not None:
            wanted = resolve_tts_engine(engine)
            if wanted != pack.engine:
                raise UnsupportedEngineError(
                    f"voice '{pack.id}' runs on engine '{pack.engine}', not '{wanted}' — "
                    "pick a voice that uses that engine, or omit the model override"
                )
        set_used(model=pack.engine, voice=pack.id)
        self.unload_idle()
        if not self._tts_slots.acquire(timeout=30):
            raise EngineBusyError("all TTS slots are busy; retry shortly (Retry-After: 5)")
        try:
            backend = self._get_tts(pack)
            for chunk in prepare_text(cleaned):
                yield from backend.synthesize_stream(chunk, pack, style)  # type: ignore[attr-defined]
        finally:
            self._tts_slots.release()

    # -- ASR ------------------------------------------------------------------
    def _get_asr(self, model: str | None = None) -> tuple[object, ChainResult | None]:
        """Single shared ASR instance per chain entry, loaded lazily.

        v2.1: the backend is resolved through the configured fallback
        chain (``asr.chain`` in configs/vaani.yaml, default
        ``[telugu-asr-micro]``). The first entry that constructs wins;
        a missing/unhealthy entry is skipped and logged by
        ``fallback.try_chain``. Returns (backend, chain_result).

        ``model`` is a per-request override: exactly one entry is
        constructed, with NO fallback to another entry — if that entry
        is unhealthy the call fails with 503 naming it, never silently
        substituting a different recogniser.

        The real backends (read from ``asr/infer.py`` 2026-11-21):
          * ``FasterWhisperBackend(model_size="models/telugu-asr-micro")``
            — the shipped recogniser: faster-whisper tiny INT8, CPU,
            ``language="te"``, beam 5, VAD on. ``transcribe(Path) -> str``.
          * ``IndicConformerBackend(checkpoint, decoding="ctc"|"rnnt")``
            — opt-in accuracy tier (NeMo, gated checkpoint, not installed
            in this checkout). ``transcribe(Path) -> str``.
        Both return a plain string, NOT a dict — ``transcribe()`` below
        wraps it into {text, language, segments, duration, model,
        attempted} so the API contract stays stable while the backends
        stay untouched.
        """
        cfg = self.config
        chain = list(cfg.asr_chain) if cfg else ["telugu-asr-micro"]
        decoding = cfg.asr_decoding if cfg else "ctc"
        with self._asr_lock:
            if model is not None:
                wanted = resolve_asr_model(model)
                if wanted in self._asr_by_model:
                    return self._asr_by_model[wanted], None
                self._apply_threads_once()
                backend = load_asr_backend(wanted, decoding=decoding)
                self._asr_by_model[wanted] = backend
                log.info("loaded ASR backend %r (per-request override)", wanted)
                return backend, None
            if self._asr is not None:
                return self._asr, None
            self._apply_threads_once()
            result = try_chain(
                chain,
                lambda name: load_asr_backend(name, decoding=decoding),
                task="ASR",
            )
            self._asr = result.backend
            self._asr_by_model[result.used] = result.backend
            log.info("loaded ASR backend %r from chain %s", result.used, chain)
            return result.backend, result

    def transcribe(self, audio_path: str | Path, model: str | None = None) -> dict:
        """Transcribe one audio file.

        Returns {text, language, segments, duration, model, attempted}.
        ``model`` is the chain entry that actually ran; ``attempted`` is
        the list of chain entries tried (names + outcome), so a fallback
        is visible to the caller instead of silent. ``model=<name>``
        selects one entry: unknown names raise 404, unhealthy entries
        raise 503 naming that entry.
        """
        from .errors import AudioValidationError

        max_mb = self.config.max_audio_mb if self.config else settings.max_audio_mb
        path = Path(audio_path)
        if not path.is_file():
            raise AudioValidationError(f"audio file not found: {path}")
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_mb:
            raise AudioValidationError(
                f"audio file is {size_mb:.1f} MB, over the limit of {max_mb} MB"
            )
        if path.stat().st_size == 0:
            raise AudioValidationError("audio file is empty")
        if not self._asr_slots.acquire(timeout=60):
            raise EngineBusyError("all ASR slots are busy; retry shortly (Retry-After: 5)")
        try:
            backend, chain_result = self._get_asr(model=model)
            used = chain_result.used if chain_result is not None else (resolve_asr_model(model) if model else None)
            if used is None:
                cfg = self.config
                used = list(cfg.asr_chain)[0] if cfg and cfg.asr_chain else "telugu-asr-micro"
            if callable(backend) and not hasattr(backend, "transcribe"):
                result = backend(str(path))
            else:
                result = backend.transcribe(str(path))  # type: ignore[attr-defined]
            if isinstance(result, str):
                result = {"text": result, "language": "te", "segments": [], "duration": None}
            result.setdefault("language", "te")
            result.setdefault("segments", [])
            result.setdefault("duration", None)
            result["model"] = used
            result["attempted"] = (
                [a.to_dict() for a in chain_result.attempted] if chain_result is not None else [{"name": used, "ok": True, "reason": "loaded"}]
            )
            set_used(model=used)
            return result
        finally:
            self._asr_slots.release()

    def close(self) -> None:
        """Unload every voice + ASR (server shutdown path)."""
        for voice_id in list(self._voices):
            self.unload_voice(voice_id)
        with self._asr_lock:
            asr, self._asr = self._asr, None
            self._asr_by_model.clear()
        if asr is not None and hasattr(asr, "close"):
            try:
                asr.close()  # type: ignore[attr-defined]
            except Exception:
                log.exception("error closing ASR backend")
        self.cache.clear()


# Module-level singleton, created lazily — importing core loads no model.
_pool: EnginePool | None = None


def get_pool() -> EnginePool:
    global _pool
    if _pool is None:
        _pool = EnginePool()
    return _pool


def reset_pool() -> None:
    """Test-only helper."""
    global _pool
    if _pool is not None:
        _pool.close()
    _pool = None
