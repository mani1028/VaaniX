"""Vaani X backend fallback chains (v2.1) — flexible without silent swaps.

Two rules govern everything here:

1. A CHAIN is ordered and honest. ``try_chain()`` tries each candidate
   in order; a missing or unhealthy candidate is skipped, LOGGED, and
   recorded in the returned ``attempts`` list. If one works, the caller
   learns WHICH one worked (``used``) and what was tried. If none work,
   the error names EVERYTHING that was tried and why each failed, and
   the API returns 503 — never a bare traceback, never a wrong answer.

2. A PER-REQUEST OVERRIDE is explicit, never a silent substitution.
   ``model=<name>`` (ASR) or ``engine=<name>`` (TTS) selects exactly one
   backend. If that backend is unknown the call fails with 404
   (``model_not_found`` / ``voice_not_found``); if it is known but
   unhealthy the call fails with 503 naming it. The caller is told what
   actually ran via the ``X-Vaani-Model`` / ``X-Vaani-Voice`` response
   headers (set by ``api/main.py`` from the contextvar below) and via
   the ``model`` / ``attempted`` fields in ASR results.

Health probes in this module NEVER load a model. They check only that a
model directory exists, or that an importable library is present
(``importlib.util.find_spec``). ``/health`` must stay cheap.

Known ASR chain entries (paths are relative to the project root):

    telugu-asr-micro     faster-whisper tiny INT8, models/telugu-asr-micro — SHIPPED
    telugu-asr           faster-whisper small INT8, models/telugu-asr — measured in
                         docs/benchmark_results.md; present only if the folder is
    vasista22-small      faster-whisper fine-tune, models/vasista22-whisper-telugu-small
                         — RECOMMENDED upgrade, NOT installed in this checkout
    indicconformer-600m  NeMo IndicConformer, ai4bharat/indicconformer_stt_te_hybrid_ctc_rnnt_large
                         — gated HF checkpoint, NOT installed; needs torch + NeMo

Unknown chain entries are NOT an error at config time: they are recorded
as ``unavailable`` with reason ``unknown backend`` and skipped, so a
config that lists a future backend does not break today's server. A
chain in which NO entry is available makes ASR ``unavailable`` in
``/health`` and every transcription a 503 naming what was tried.
"""

from __future__ import annotations

import contextvars
import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .errors import EngineNotLoadedError, VaaniError

log = logging.getLogger("vaani.core.fallback")

# ---------------------------------------------------------------------------
# What actually ran — read by api/main.py to set X-Vaani-* response headers
# ---------------------------------------------------------------------------

_USED: contextvars.ContextVar[dict] = contextvars.ContextVar("vaani_used", default={})


def set_used(*, model: str | None = None, voice: str | None = None) -> None:
    """Record the backend/voice that served the current request."""
    current = dict(_USED.get() or {})
    if model is not None:
        current["model"] = model
    if voice is not None:
        current["voice"] = voice
    _USED.set(current)


def get_used() -> dict:
    return dict(_USED.get() or {})


def clear_used() -> None:
    _USED.set({})


# ---------------------------------------------------------------------------
# Attempts — the audit trail of a chain resolution
# ---------------------------------------------------------------------------

@dataclass
class BackendAttempt:
    name: str
    ok: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "reason": self.reason}


@dataclass
class ChainResult:
    backend: object
    used: str
    attempted: list[BackendAttempt] = field(default_factory=list)

    def attempted_names(self) -> list[str]:
        return [a.name for a in self.attempted]


def try_chain(candidates: list[str],
              loader: Callable[[str], object],
              task: str) -> ChainResult:
    """Try each candidate in order; return the first that loads.

    ``loader(name)`` raises on failure (any exception is caught, logged
    and recorded). Raises EngineNotLoadedError naming everything tried
    when no candidate works.
    """
    attempted: list[BackendAttempt] = []
    for name in candidates:
        try:
            backend = loader(name)
        except Exception as exc:  # noqa: BLE001 — every failure is data here
            reason = str(exc) or exc.__class__.__name__
            attempted.append(BackendAttempt(name=name, ok=False, reason=reason))
            log.warning("fallback: %s backend %r failed: %s", task, name, reason)
            continue
        attempted.append(BackendAttempt(name=name, ok=True, reason="loaded"))
        if len(attempted) > 1:
            skipped = ", ".join(a.name for a in attempted[:-1])
            log.warning("fallback: %s fell back to %r after %s failed", task, name, skipped)
        return ChainResult(backend=backend, used=name, attempted=attempted)
    tried = "; ".join(f"{a.name} ({a.reason})" for a in attempted) or "(empty chain)"
    raise EngineNotLoadedError(
        f"no working {task} backend: tried {tried} — "
        "check the model paths in configs/vaani.yaml and that the weights are installed"
    )


# ---------------------------------------------------------------------------
# ASR chain — known entries and cheap (no model load) probes
# ---------------------------------------------------------------------------

# name -> (kind, location). kind "faster-whisper" takes a local model dir;
# kind "indicconformer" takes a Hugging Face checkpoint id.
ASR_SPECS: dict[str, tuple[str, str]] = {
    "telugu-asr-micro": ("faster-whisper", "models/telugu-asr-micro"),
    "telugu-asr": ("faster-whisper", "models/telugu-asr"),
    "vasista22-small": ("faster-whisper", "models/vasista22-whisper-telugu-small"),
    "vasista22-base": ("faster-whisper", "models/vasista22-whisper-telugu-base"),
    "indicconformer-600m": (
        "indicconformer",
        "ai4bharat/indicconformer_stt_te_hybrid_ctc_rnnt_large",
    ),
}

# OpenAI-style model aliases accepted by POST /v1/audio/transcriptions.
ASR_MODEL_ALIASES = {
    "vaani-asr-small": "telugu-asr-micro",
    "vaani-asr": "telugu-asr-micro",
}

# OpenAI-style TTS model aliases accepted by POST /v1/audio/speech.
TTS_MODEL_ALIASES = {
    "vaani-tts-piper": "piper",
    "vaani-tts-hima": "hima-nano",
    "vaani-tts": "hima-nano",
}

# Short voice aliases accepted by POST /v1/audio/speech (same map as
# api/routes_openai.py::_VOICE_ALIASES, kept here so api/main.py can
# label responses without importing the routes module).
VOICE_ALIASES = {
    "srinivas": "telugu_srinivas",
    "telugu_srinivas": "telugu_srinivas",
    "sravani": "telugu_sravani",
    "telugu_sravani": "telugu_sravani",
    "telugu_default": "telugu_srinivas",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def probe_asr_entry(name: str, project_root: Path | None = None) -> BackendAttempt:
    """Cheap probe: is this ASR chain entry even loadable? No model is loaded."""
    spec = ASR_SPECS.get(name)
    if spec is None:
        return BackendAttempt(
            name=name, ok=False,
            reason=f"unknown backend {name!r}; known: {', '.join(sorted(ASR_SPECS))}",
        )
    kind, location = spec
    root = Path(project_root) if project_root else _project_root()
    if kind == "faster-whisper":
        model_dir = (root / location) if not Path(location).is_absolute() else Path(location)
        if not model_dir.is_dir():
            return BackendAttempt(
                name=name, ok=False,
                reason=f"model directory not installed: {model_dir} — place the weights there or remove {name!r} from asr.chain",
            )
        if importlib.util.find_spec("faster_whisper") is None:
            return BackendAttempt(name=name, ok=False, reason="faster-whisper package is not installed")
        return BackendAttempt(name=name, ok=True, reason="model directory present")
    # indicconformer: needs torch + NeMo; the checkpoint itself is gated
    # and is only downloaded on first load, so the probe checks libraries.
    missing = [lib for lib in ("torch", "nemo") if importlib.util.find_spec(lib) is None]
    if missing:
        return BackendAttempt(
            name=name, ok=False,
            reason=f"{' and '.join(missing)} not installed — IndicConformer needs torch + NeMo (nemo-v2)",
        )
    return BackendAttempt(name=name, ok=True, reason="torch + NeMo present; checkpoint is gated (downloads on first load)")


def load_asr_backend(name: str, decoding: str = "ctc") -> object:
    """Construct ONE ASR backend by chain-entry name. Raises on any problem.

    Construction does NOT run inference and does NOT download a model;
    faster-whisper loads weights lazily, IndicConformer on first use.
    A missing model directory raises EngineNotLoadedError immediately so
    the chain can move on to the next candidate.
    """
    spec = ASR_SPECS.get(name)
    if spec is None:
        raise EngineNotLoadedError(
            f"unknown ASR backend {name!r}; known: {', '.join(sorted(ASR_SPECS))}"
        )
    kind, location = spec
    if kind == "faster-whisper":
        probe = probe_asr_entry(name)
        if not probe.ok:
            raise EngineNotLoadedError(probe.reason)
        from telugu_speech.asr.infer import FasterWhisperBackend, decoding_from_env

        # The measured defaults (beam 5, temp 0.0, VAD on, language te)
        # apply unless VAANI_ASR_BEAM / VAANI_ASR_TEMPERATURE /
        # VAANI_ASR_VAD / VAANI_ASR_NO_SPEECH_THRESHOLD override them.
        # A bad env value raises here and is recorded as a failed chain
        # attempt by try_chain, naming the value — never silently ignored.
        return FasterWhisperBackend(location, decoding=decoding_from_env())
    from telugu_speech.asr.infer import IndicConformerBackend

    if decoding not in ("ctc", "rnnt"):
        raise EngineNotLoadedError(f"asr.decoding must be ctc|rnnt, got {decoding!r}")
    probe = probe_asr_entry(name)
    if not probe.ok:
        raise EngineNotLoadedError(probe.reason)
    return IndicConformerBackend(location, decoding)


def resolve_asr_model(requested: str | None) -> str:
    """Normalise a per-request ASR model name. Raises 404 for unknown names."""
    if requested is None:
        return ""
    name = ASR_MODEL_ALIASES.get(requested.strip(), requested.strip())
    if name not in ASR_SPECS:
        raise VaaniError(
            f"Unknown ASR model: {requested!r}. Available models: "
            f"{', '.join(sorted(ASR_SPECS))} (aliases: {', '.join(sorted(ASR_MODEL_ALIASES))}).",
            code="model_not_found", http_status=404,
        )
    return name


def resolve_tts_engine(requested: str | None) -> str:
    """Normalise a per-request TTS model/engine name. Raises 404 for unknown names."""
    if requested is None:
        return ""
    name = TTS_MODEL_ALIASES.get(requested.strip(), requested.strip())
    if name not in ("hima-nano", "piper", "onnx-vits"):
        raise VaaniError(
            f"Unknown TTS model: {requested!r}. Available models: "
            "vaani-tts, vaani-tts-hima, vaani-tts-piper.",
            code="model_not_found", http_status=404,
        )
    return name


# ---------------------------------------------------------------------------
# Component status — for /health. Cheap probes only, no model loads.
# ---------------------------------------------------------------------------

def component_status(pool: object | None = None,
                     registry: object | None = None,
                     config: object | None = None,
                     legacy_voices: list[dict] | None = None) -> dict:
    """Per-component health: {overall, asr, tts, registry, config}.

    Status values: ok | degraded | unavailable.
      * asr is ok when at least one chain entry probes available;
        degraded when the chain has entries but only some are available
        AND the default entry is available; unavailable when none are.
      * tts is ok when the default voice is registered (or present via
        the legacy fallback) AND at least one pack is registered;
        degraded when some voices are rejected but the default works;
        unavailable when no voice works at all.
      * registry reports rejected packs with their reasons.
    ``legacy_voices`` is the list from tts/voices.list_installed_voices()
    (passed in by the API layer — this module must not import tts/).
    """
    cfg = config
    chain: list[str] = list(getattr(cfg, "asr_chain", ("telugu-asr-micro",))) if cfg is not None else ["telugu-asr-micro"]
    default_voice = str(getattr(cfg, "default_voice", "telugu_srinivas")) if cfg is not None else "telugu_srinivas"

    asr_entries = [probe_asr_entry(name).to_dict() for name in chain]
    asr_ok = [e for e in asr_entries if e["ok"]]
    if asr_ok:
        asr_status = "ok" if len(asr_ok) == len(asr_entries) else "degraded"
    else:
        asr_status = "unavailable"
    loaded_asr = None
    if pool is not None:
        loaded_asr = getattr(pool, "_asr", None) is not None or bool(getattr(pool, "_asr_by_model", {}))

    voices: dict[str, dict] = {}
    rejected: list[dict] = []
    if registry is not None:
        try:
            for pack in registry.list_voices():  # type: ignore[attr-defined]
                voices[pack.id] = {
                    "status": "ok", "engine": pack.engine,
                    "loaded": bool(pool is not None and pack.id in getattr(pool, "loaded_voices", lambda: [])()),
                }
        except Exception as exc:  # noqa: BLE001 — health must not raise
            rejected.append({"pack_dir": "registry", "reason": str(exc)})
        rejected.extend(getattr(registry, "rejected", []) or [])
    for item in legacy_voices or []:
        voice_id = str(item.get("id", ""))
        if voice_id and voice_id not in voices:
            voices[voice_id] = {
                "status": "ok (legacy fallback)", "engine": str(item.get("backend", "hima-onnx-int8")),
                "loaded": False,
            }
    if default_voice in voices:
        tts_status = "degraded" if rejected else "ok"
    elif voices:
        tts_status = "degraded"
    else:
        tts_status = "unavailable"

    overall = "ok"
    if asr_status == "unavailable" or tts_status == "unavailable":
        overall = "degraded" if (asr_status != "unavailable" or tts_status != "unavailable") else "unavailable"
    elif asr_status == "degraded" or tts_status == "degraded":
        overall = "degraded"
    return {
        "overall": overall,
        "asr": {
            "status": asr_status,
            "chain": asr_entries,
            "loaded": bool(loaded_asr),
            "default": chain[0] if chain else None,
        },
        "tts": {
            "status": tts_status,
            "voices": voices,
            "default_voice": default_voice,
            "rejected": rejected,
        },
        "registry": {
            "status": "ok" if not rejected else "degraded",
            "rejected": rejected,
        },
        "config": {
            "status": "ok",
            "offline": bool(getattr(cfg, "offline", False)) if cfg is not None else False,
            "file": getattr(cfg, "config_file", None) if cfg is not None else None,
        },
    }
