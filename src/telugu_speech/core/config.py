"""Vaani X core configuration — one file + env overrides, validated strictly.

Two layers, kept separate on purpose:

1. ``Settings`` (legacy, env-only) — the v2 Part 1 object. It is kept
   byte-for-byte compatible so existing imports and tests do not break.
   Its integer helpers fall back silently on a bad value; that behaviour
   is preserved ONLY for this legacy object and is documented as a flaw,
   not a feature. New code must use layer 2.

2. ``EffectiveConfig`` (v2.1) — the single source of truth the server
   actually runs on. Sources, in precedence order:

       environment variable (``VAANI_*``)  >  ``configs/vaani.yaml``  >  built-in default

   Every value is VALIDATED. A bad value raises ``ConfigError`` naming
   the key, the value supplied, and what is allowed — at startup
   (``load_config()`` is called by ``api/main.py:configure()``) and on
   ``POST /v1/reload``. A bad value is NEVER silently replaced by a
   default, and a bad reload NEVER kills the running server: the last
   good config keeps serving and the reload response carries the error.

   ``configs/vaani.yaml`` is found via ``VAANI_CONFIG``, else
   ``<project>/configs/vaani.yaml`` if it exists, else built-in defaults
   alone. If ``VAANI_CONFIG`` names a file that does not exist, that is
   an error, not a silent fall-through.

No secrets live in this module or in the config file. API keys stay in
the environment (``TELUGU_API_KEYS``) and are never returned by
``GET /v1/config`` — ``EffectiveConfig.redacted()`` masks any key whose
name contains key/token/secret/password, should one ever appear.

Sizing note (from the Part 1 brief — UNMEASURED, mark as TBD until the
bench harness measures it on the target hardware):
  * one loaded TTS engine: ~260-470 MB
  * two loaded TTS engines: ~640-720 MB
  * eight concurrent engines would therefore need ~1.5-1.8 GB and would
    OOM a 1-2 GB VPS. That is why ``VAANI_TTS_SLOTS`` defaults to 2,
    not 8. Do not raise it without measuring peak RSS first.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import VaaniError


class ConfigError(VaaniError):
    """A config key is missing, malformed, or out of range.

    The message MUST name the key, the value supplied, and what is
    allowed, e.g. ``config key 'limits.tts_slots': value 9 is out of
    range 1-4``. Callers render this as HTTP 400 on reload and as a
    startup failure otherwise — never as a silent default.
    """

    code = "config_invalid"
    http_status = 400


def _int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    """Read an int env var, clamped to [minimum, maximum]; bad values fall back.

    LEGACY behaviour, kept for ``Settings`` only. Do not use for new keys.
    """
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def _str(name: str, default: str) -> str:
    value = os.environ.get(name, default)
    return value if value else default


@dataclass(frozen=True)
class Settings:
    """All core tunables. Frozen so a running server cannot drift mid-request."""

    # --- CPU / threading -------------------------------------------------
    # ONNX Runtime + ORT intra-op threads. Default 2, hard cap 4: more
    # threads on a 2-vCPU box only add contention, never throughput.
    cpu_threads: int = field(default_factory=lambda: _int("TELUGU_CPU_THREADS", 2, minimum=1, maximum=4))
    # Alias used by some deployments; TELUGU_CPU_THREADS wins if both set.
    onnx_threads: int = field(default_factory=lambda: _int("VAANI_ONNX_THREADS", 2, minimum=1, maximum=4))

    # --- Concurrency -----------------------------------------------------
    # Simultaneous TTS syntheses. Default 2 (see sizing note above).
    tts_slots: int = field(default_factory=lambda: _int("VAANI_TTS_SLOTS", 2, minimum=1, maximum=4))
    # Simultaneous ASR transcriptions. ASR is a single shared instance,
    # so this is a queue depth, not a copy count.
    asr_slots: int = field(default_factory=lambda: _int("VAANI_ASR_SLOTS", 2, minimum=1, maximum=4))

    # --- Voice discovery -------------------------------------------------
    # Extra directories to scan for voice packs, colon-separated.
    # Built-in search roots are always: <project>/voices, <project>/lite/models
    extra_voices_dirs: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            p for p in _str("VAANI_VOICES_DIR", "").split(":") if p
        )
    )

    # --- Lazy loading / idle unload --------------------------------------
    # Seconds a loaded voice may sit idle before it is unloaded to free
    # RAM. 0 disables idle unload (voice stays resident once loaded).
    voice_idle_secs: int = field(default_factory=lambda: _int("VAANI_VOICE_IDLE_SECS", 1800, minimum=0, maximum=86400))

    # --- Text limits ------------------------------------------------------
    # 2000 matches the API's validation ("Text must be at most 2000
    # characters", api/main.py). The core must not accept text the API
    # would then reject.
    max_text_chars: int = field(default_factory=lambda: _int("VAANI_MAX_TEXT_CHARS", 2000, minimum=1, maximum=20000))
    # Long text is split into chunks of at most this many characters,
    # always on a sentence boundary where possible. 100 keeps one Hima
    # call short enough that it is less likely to stop after the first
    # word. The ONNX embedding still allows much longer pieces.
    max_chunk_chars: int = field(default_factory=lambda: _int("VAANI_MAX_CHUNK_CHARS", 100, minimum=100, maximum=2000))

    # --- Audio cache ------------------------------------------------------
    # LRU cache of synthesised audio, keyed by (voice, style, text hash).
    # Bounded by entries AND bytes; whichever limit is hit first evicts.
    audio_cache_entries: int = field(default_factory=lambda: _int("VAANI_AUDIO_CACHE_ENTRIES", 128, minimum=0, maximum=1024))
    audio_cache_max_mb: int = field(default_factory=lambda: _int("VAANI_AUDIO_CACHE_MAX_MB", 256, minimum=0, maximum=2048))

    # --- ASR ---------------------------------------------------------------
    max_audio_mb: int = field(default_factory=lambda: _int("VAANI_MAX_AUDIO_MB", 25, minimum=1, maximum=100))

    # --- Voices --------------------------------------------------------------
    default_voice: str = field(default_factory=lambda: _str("VAANI_DEFAULT_VOICE", "telugu_srinivas"))
    # The only style shipped in v2 Part 1. Any other value is rejected
    # with a 400, not silently mapped.
    allowed_styles: tuple[str, ...] = ("conversational",)

    def effective_threads(self) -> int:
        """Threads actually applied to ONNX Runtime sessions."""
        return min(self.cpu_threads, self.onnx_threads) if (
            "VAANI_ONNX_THREADS" in os.environ
        ) else self.cpu_threads


# Module-level singleton; import it, do not re-instantiate per request.
settings = Settings()


def apply_thread_env() -> dict[str, str]:
    """Return the thread env vars the engine pool should set before loading ONNX.

    Callers (engine.py) apply these to ``os.environ`` before the first
    ONNX Runtime session is created — ORT reads them at session creation.
    Returns the dict actually applied, for logging.
    """
    threads = str(settings.effective_threads())
    applied = {
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "ONNX_NUM_THREADS": threads,
    }
    for key, value in applied.items():
        os.environ.setdefault(key, value)
    return applied


# ===========================================================================
# v2.1 — EffectiveConfig: one file + env overrides, strictly validated
# ===========================================================================

def _project_root() -> Path:
    """Project root from this file's location: .../<project>/src/telugu_speech/core/config.py."""
    return Path(__file__).resolve().parents[3]


# Flat key -> (section, key) in configs/vaani.yaml. Every key the file may
# carry is listed here; any other key in the file is a ConfigError.
_FILE_KEYS: dict[str, tuple[str, str]] = {
    "server.host": ("server", "host"),
    "server.port": ("server", "port"),
    "server.rate_limit_per_minute": ("server", "rate_limit_per_minute"),
    "server.cors_origins": ("server", "cors_origins"),
    "server.limit_concurrency": ("server", "limit_concurrency"),
    "asr.chain": ("asr", "chain"),
    "asr.model": ("asr", "model"),
    "asr.backend": ("asr", "backend"),
    "asr.decoding": ("asr", "decoding"),
    "tts.chain": ("tts", "chain"),
    "tts.default_voice": ("tts", "default_voice"),
    "tts.default_engine": ("tts", "default_engine"),
    "limits.tts_slots": ("limits", "tts_slots"),
    "limits.tts_queue": ("limits", "tts_queue"),
    "limits.asr_slots": ("limits", "asr_slots"),
    "limits.cpu_threads": ("limits", "cpu_threads"),
    "limits.max_text_chars": ("limits", "max_text_chars"),
    "limits.max_chunk_chars": ("limits", "max_chunk_chars"),
    "limits.max_upload_mb": ("limits", "max_upload_mb"),
    "limits.max_audio_mb": ("limits", "max_audio_mb"),
    "cache.dir": ("cache", "dir"),
    "cache.audio_cache_entries": ("cache", "audio_cache_entries"),
    "cache.audio_cache_max_mb": ("cache", "audio_cache_max_mb"),
    "voices.dirs": ("voices", "dirs"),
    "voices.idle_secs": ("voices", "idle_secs"),
    "offline": ("", "offline"),
    "log_level": ("", "log_level"),
}

# Flat key -> env var(s) that override it, in order (first set wins).
_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "server.host": ("VAANI_HOST", "TELUGU_HOST"),
    "server.port": ("VAANI_PORT", "TELUGU_PORT"),
    "server.rate_limit_per_minute": ("VAANI_RATE_LIMIT_PER_MINUTE",),
    "server.cors_origins": ("VAANI_CORS_ORIGINS", "TELUGU_CORS_ORIGINS"),
    "server.limit_concurrency": ("VAANI_LIMIT_CONCURRENCY", "TELUGU_LIMIT_CONCURRENCY"),
    "asr.chain": ("VAANI_ASR_CHAIN",),
    "asr.model": ("VAANI_ASR_MODEL", "TELUGU_WHISPER_MODEL"),
    "asr.backend": ("VAANI_ASR_BACKEND", "TELUGU_ASR_BACKEND"),
    "asr.decoding": ("VAANI_ASR_DECODING", "TELUGU_ASR_DECODING"),
    "tts.chain": ("VAANI_TTS_CHAIN",),
    "tts.default_voice": ("VAANI_DEFAULT_VOICE",),
    "tts.default_engine": ("VAANI_DEFAULT_ENGINE",),
    "limits.tts_slots": ("VAANI_TTS_SLOTS", "TELUGU_SPEECH_SLOTS"),
    "limits.tts_queue": ("VAANI_TTS_QUEUE",),
    "limits.asr_slots": ("VAANI_ASR_SLOTS",),
    "limits.cpu_threads": ("VAANI_CPU_THREADS", "TELUGU_CPU_THREADS"),
    "limits.max_text_chars": ("VAANI_MAX_TEXT_CHARS",),
    "limits.max_chunk_chars": ("VAANI_MAX_CHUNK_CHARS",),
    # max_upload_mb: VAANI_MAX_UPLOAD_MB is in MB; the legacy
    # TELUGU_MAX_UPLOAD_BYTES is in BYTES and is converted in load_config().
    "limits.max_upload_mb": ("VAANI_MAX_UPLOAD_MB", "TELUGU_MAX_UPLOAD_BYTES"),
    "limits.max_audio_mb": ("VAANI_MAX_AUDIO_MB",),
    "cache.dir": ("VAANI_CACHE_DIR",),
    "cache.audio_cache_entries": ("VAANI_AUDIO_CACHE_ENTRIES",),
    "cache.audio_cache_max_mb": ("VAANI_AUDIO_CACHE_MAX_MB",),
    "voices.dirs": ("VAANI_VOICES_DIR",),
    "voices.idle_secs": ("VAANI_VOICE_IDLE_SECS",),
    "offline": ("VAANI_OFFLINE",),
    "log_level": ("VAANI_LOG_LEVEL",),
}

_DEFAULTS: dict[str, Any] = {
    "server.host": "127.0.0.1",
    "server.port": 8000,
    "server.rate_limit_per_minute": 2000,
    "server.cors_origins": [],
    "server.limit_concurrency": 256,
    "asr.chain": ["telugu-asr-micro"],
    "asr.model": "models/telugu-asr-micro",
    "asr.backend": "faster-whisper",
    "asr.decoding": "ctc",
    "tts.chain": ["hima-nano"],
    "tts.default_voice": "telugu_srinivas",
    "tts.default_engine": "hima-nano",
    "limits.tts_slots": 2,
    "limits.tts_queue": 128,
    "limits.asr_slots": 1,
    "limits.cpu_threads": 2,
    "limits.max_text_chars": 2000,
    "limits.max_chunk_chars": 100,
    "limits.max_upload_mb": 25,
    "limits.max_audio_mb": 25,
    "cache.dir": "~/.cache/vaani",
    "cache.audio_cache_entries": 32,
    "cache.audio_cache_max_mb": 32,
    "voices.dirs": [],
    "voices.idle_secs": 1800,
    "offline": False,
    "log_level": "INFO",
}

# key -> (min, max) for strict integer validation. max_upload_mb is in MB
# here (1-50, matching the legacy TELUGU_MAX_UPLOAD_BYTES hard cap of 50 MB).
_INT_RANGES: dict[str, tuple[int, int]] = {
    "server.port": (1, 65535),
    "server.rate_limit_per_minute": (0, 100000),
    "server.limit_concurrency": (1, 10000),
    "limits.tts_slots": (1, 4),
    "limits.tts_queue": (0, 256),
    "limits.asr_slots": (1, 4),
    "limits.cpu_threads": (1, 4),
    "limits.max_text_chars": (1, 20000),
    "limits.max_chunk_chars": (100, 2000),
    "limits.max_upload_mb": (1, 50),
    "limits.max_audio_mb": (1, 100),
    "cache.audio_cache_entries": (0, 1024),
    "cache.audio_cache_max_mb": (0, 2048),
    "voices.idle_secs": (0, 86400),
}

_ENUMS: dict[str, tuple[str, ...]] = {
    "asr.backend": ("faster-whisper", "indicconformer"),
    "asr.decoding": ("ctc", "rnnt"),
    "log_level": ("DEBUG", "INFO", "WARNING", "ERROR"),
}

_LIST_KEYS = {"server.cors_origins", "asr.chain", "tts.chain", "voices.dirs"}
_BOOL_KEYS = {"offline"}
_SECRET_SUBSTRINGS = ("key", "token", "secret", "password")


def _coerce_int(key: str, raw: Any, source: str) -> int:
    """Strict int: bool is not an int, floats must be whole, strings must parse."""
    if isinstance(raw, bool):
        raise ConfigError(f"config key '{key}': value {raw!r} from {source} is not an integer")
    if isinstance(raw, float):
        if not raw.is_integer():
            raise ConfigError(f"config key '{key}': value {raw!r} from {source} is not an integer")
        raw = int(raw)
    elif isinstance(raw, str):
        try:
            raw = int(raw.strip())
        except ValueError:
            raise ConfigError(
                f"config key '{key}': value {raw!r} from {source} is not an integer"
            ) from None
    elif not isinstance(raw, int):
        raise ConfigError(f"config key '{key}': value {raw!r} from {source} is not an integer")
    lo, hi = _INT_RANGES[key]
    if not lo <= raw <= hi:
        raise ConfigError(
            f"config key '{key}': value {raw} from {source} is out of range {lo}-{hi}"
        )
    return raw


def _coerce_bool(key: str, raw: Any, source: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
    raise ConfigError(
        f"config key '{key}': value {raw!r} from {source} is not a boolean "
        "(use true/false, 1/0, yes/no, on/off)"
    )


def _coerce_list(key: str, raw: Any, source: str) -> list[str]:
    if isinstance(raw, str):
        items = [p.strip() for p in raw.split(",") if p.strip()]
    elif isinstance(raw, (list, tuple)):
        items = [str(p).strip() for p in raw if str(p).strip()]
    else:
        raise ConfigError(
            f"config key '{key}': value {raw!r} from {source} must be a list "
            "or a comma-separated string"
        )
    if key in ("asr.chain", "tts.chain") and not items:
        raise ConfigError(
            f"config key '{key}': value from {source} must be a non-empty list — "
            "a fallback chain with no backends can never serve a request"
        )
    return items


def _coerce_str(key: str, raw: Any, source: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"config key '{key}': value {raw!r} from {source} must be a non-empty string")
    return raw.strip()


def _validate_value(key: str, raw: Any, source: str) -> Any:
    """Coerce + validate one resolved value. Returns the typed value."""
    if key in _INT_RANGES:
        return _coerce_int(key, raw, source)
    if key in _BOOL_KEYS:
        return _coerce_bool(key, raw, source)
    if key in _LIST_KEYS:
        return _coerce_list(key, raw, source)
    if key in _ENUMS:
        value = _coerce_str(key, raw, source)
        allowed = _ENUMS[key]
        if key == "log_level":
            value = value.upper()
        if value not in allowed:
            raise ConfigError(
                f"config key '{key}': value {value!r} from {source} must be one of "
                f"{', '.join(allowed)}"
            )
        return value
    return _coerce_str(key, raw, source)


def _load_yaml_file(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        raise ConfigError(
            f"config file '{path}' cannot be read: PyYAML is required "
            "(declared dependency: pyyaml>=6) but is not installed"
        ) from None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError(f"config file '{path}' is not valid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file '{path}' must be a YAML mapping at the top level")
    flat: dict[str, Any] = {}
    for section, values in data.items():
        if section in ("offline", "log_level"):
            flat[section] = values
            continue
        if not isinstance(values, dict):
            raise ConfigError(
                f"config file '{path}': section '{section}' must be a mapping, got {values!r}"
            )
        for name, value in values.items():
            flat[f"{section}.{name}"] = value
    unknown = sorted(set(flat) - set(_FILE_KEYS))
    if unknown:
        raise ConfigError(
            f"config file '{path}': unknown key(s): {', '.join(unknown)} — "
            f"known keys are: {', '.join(sorted(_FILE_KEYS))}"
        )
    return flat


def _find_config_file(explicit: str | Path | None) -> Path | None:
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigError(f"config file '{path}' (from VAANI_CONFIG) does not exist")
        return path
    env_path = os.environ.get("VAANI_CONFIG", "").strip()
    if env_path:
        path = Path(env_path).expanduser()
        if not path.is_file():
            raise ConfigError(f"config file '{path}' (from VAANI_CONFIG) does not exist")
        return path
    candidate = _project_root() / "configs" / "vaani.yaml"
    return candidate if candidate.is_file() else None


@dataclass(frozen=True)
class EffectiveConfig:
    """The validated, merged configuration the server runs on."""

    host: str = "127.0.0.1"
    port: int = 8000
    rate_limit_per_minute: int = 2000
    cors_origins: tuple[str, ...] = ()
    limit_concurrency: int = 256
    asr_chain: tuple[str, ...] = ("telugu-asr-micro",)
    asr_model: str = "models/telugu-asr-micro"
    asr_backend: str = "faster-whisper"
    asr_decoding: str = "ctc"
    tts_chain: tuple[str, ...] = ("hima-nano",)
    default_voice: str = "telugu_srinivas"
    default_engine: str = "hima-nano"
    tts_slots: int = 2
    tts_queue: int = 128
    asr_slots: int = 1
    cpu_threads: int = 2
    max_text_chars: int = 2000
    max_chunk_chars: int = 100
    max_upload_mb: int = 25
    max_audio_mb: int = 25
    cache_dir: str = "~/.cache/vaani"
    audio_cache_entries: int = 32
    audio_cache_max_mb: int = 32
    voices_dirs: tuple[str, ...] = ()
    voice_idle_secs: int = 1800
    offline: bool = False
    log_level: str = "INFO"
    # Where each flat key came from: "env:VAANI_X" | "file" | "default".
    sources: dict[str, str] = field(default_factory=dict)
    config_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "server": {
                "host": self.host, "port": self.port,
                "rate_limit_per_minute": self.rate_limit_per_minute,
                "cors_origins": list(self.cors_origins),
                "limit_concurrency": self.limit_concurrency,
            },
            "asr": {
                "chain": list(self.asr_chain), "model": self.asr_model,
                "backend": self.asr_backend, "decoding": self.asr_decoding,
            },
            "tts": {
                "chain": list(self.tts_chain), "default_voice": self.default_voice,
                "default_engine": self.default_engine,
            },
            "limits": {
                "tts_slots": self.tts_slots, "tts_queue": self.tts_queue, "asr_slots": self.asr_slots,
                "cpu_threads": self.cpu_threads,
                "max_text_chars": self.max_text_chars,
                "max_chunk_chars": self.max_chunk_chars,
                "max_upload_mb": self.max_upload_mb,
                "max_audio_mb": self.max_audio_mb,
            },
            "cache": {
                "dir": self.cache_dir,
                "audio_cache_entries": self.audio_cache_entries,
                "audio_cache_max_mb": self.audio_cache_max_mb,
            },
            "voices": {
                "dirs": list(self.voices_dirs), "idle_secs": self.voice_idle_secs,
            },
            "offline": self.offline,
            "log_level": self.log_level,
            "sources": dict(self.sources),
            "config_file": self.config_file,
        }

    def redacted(self) -> dict[str, Any]:
        """Public view for GET /v1/config. Masks anything secret-shaped."""
        data = self.to_dict()

        def _mask(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    k: ("***" if any(s in k.lower() for s in _SECRET_SUBSTRINGS) else _mask(v))
                    for k, v in obj.items()
                }
            if isinstance(obj, list):
                return [_mask(v) for v in obj]
            return obj

        return _mask(data)

    # -- convenience, so callers do not re-derive these --------------------
    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def max_audio_bytes(self) -> int:
        return self.max_audio_mb * 1024 * 1024

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir).expanduser()


def load_config(config_path: str | Path | None = None,
                env: dict[str, str] | None = None) -> EffectiveConfig:
    """Load + validate the effective config. Raises ConfigError on any bad value.

    ``env`` is injectable for tests; ``None`` means ``os.environ``.
    Resolution per key: first env var that is set (non-empty) wins, else
    the config file, else the built-in default. ``limits.max_upload_mb``
    also accepts the legacy ``TELUGU_MAX_UPLOAD_BYTES`` in BYTES — that
    env var is converted (bytes -> MB, rounded up) before validation.
    """
    environ = os.environ if env is None else env
    # _find_config_file(None) honours VAANI_CONFIG, else the default
    # configs/vaani.yaml if it exists; an explicit missing path raises.
    path = _find_config_file(config_path)
    file_values: dict[str, Any] = {}
    if path is not None:
        file_values = _load_yaml_file(path)
    resolved: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for key, default in _DEFAULTS.items():
        source = "default"
        value: Any = default
        if key in file_values:
            value = file_values[key]
            source = "file"
        for env_name in _ENV_KEYS.get(key, ()):
            raw = environ.get(env_name)
            if raw is not None and str(raw).strip() != "":
                value = raw
                source = f"env:{env_name}"
                break
        if key == "limits.max_upload_mb" and source.startswith("env:") and source[4:] == "TELUGU_MAX_UPLOAD_BYTES":
            # Legacy var is in bytes; convert to MB (round up) first.
            try:
                raw_bytes = int(str(value).strip())
            except ValueError:
                raise ConfigError(
                    f"config key '{key}': value {value!r} from {source} is not an integer number of bytes"
                ) from None
            value = -(-raw_bytes // (1024 * 1024)) or 1
        resolved[key] = _validate_value(key, value, source)
        sources[key] = source
    # Cross-field checks.
    if resolved["tts.default_engine"] not in resolved["tts.chain"]:
        raise ConfigError(
            "config: tts.default_engine "
            f"{resolved['tts.default_engine']!r} is not in tts.chain "
            f"{list(resolved['tts.chain'])} — the default engine must be in the chain"
        )
    if resolved["asr.backend"] == "indicconformer" and not resolved["asr.chain"]:
        raise ConfigError("config: asr.chain must not be empty when asr.backend is indicconformer")
    return EffectiveConfig(
        host=resolved["server.host"],
        port=resolved["server.port"],
        rate_limit_per_minute=resolved["server.rate_limit_per_minute"],
        cors_origins=tuple(resolved["server.cors_origins"]),
        limit_concurrency=resolved["server.limit_concurrency"],
        asr_chain=tuple(resolved["asr.chain"]),
        asr_model=resolved["asr.model"],
        asr_backend=resolved["asr.backend"],
        asr_decoding=resolved["asr.decoding"],
        tts_chain=tuple(resolved["tts.chain"]),
        default_voice=resolved["tts.default_voice"],
        default_engine=resolved["tts.default_engine"],
        tts_slots=resolved["limits.tts_slots"],
        tts_queue=resolved["limits.tts_queue"],
        asr_slots=resolved["limits.asr_slots"],
        cpu_threads=resolved["limits.cpu_threads"],
        max_text_chars=resolved["limits.max_text_chars"],
        max_chunk_chars=resolved["limits.max_chunk_chars"],
        max_upload_mb=resolved["limits.max_upload_mb"],
        max_audio_mb=resolved["limits.max_audio_mb"],
        cache_dir=resolved["cache.dir"],
        audio_cache_entries=resolved["cache.audio_cache_entries"],
        audio_cache_max_mb=resolved["cache.audio_cache_max_mb"],
        voices_dirs=tuple(resolved["voices.dirs"]),
        voice_idle_secs=resolved["voices.idle_secs"],
        offline=resolved["offline"],
        log_level=resolved["log_level"],
        sources=sources,
        config_file=str(path) if path is not None else None,
    )


# Module-level singleton. ``load_config()`` is called once by the server
# at startup; ``reload_config()`` swaps it only after a successful load.
_config: EffectiveConfig | None = None


def get_config() -> EffectiveConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(config_path: str | Path | None = None) -> EffectiveConfig:
    """Re-read file + env. On success the singleton is swapped and returned.

    On ConfigError the OLD config stays in place and the error propagates
    — the caller (POST /v1/reload) reports it and the server keeps running.
    """
    global _config
    fresh = load_config(config_path=config_path)
    _config = fresh
    return fresh


def reset_config() -> None:
    """Test-only helper: drop the singleton so the next get_config() re-loads."""
    global _config
    _config = None
