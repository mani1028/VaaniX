"""Vaani X voice registry — voices are USB connectors, the core is the port.

A voice is a FOLDER containing a ``manifest.yaml``. Plugging a new voice
in means dropping its folder into a voices directory and calling
``registry.reload()`` — no code change, no restart, no core edit.

Manifest contract (exact — the API, engine and test agents code against it):

    id:           telugu_srinivas        # unique, lowercase, [a-z0-9_]
    name:         Srinivas
    gender:       male                   # male | female | neutral
    language:     te                     # Telugu only in v2 Part 1
    engine:       hima-nano              # hima-nano | piper | onnx-vits
    model_path:   model.onnx             # relative to the pack folder
    config_path:  config.json            # relative to the pack folder
    sample_rate:  22050                  # Hz, int
    licence:      Apache-2.0             # SPDX id, or a short licence name
    licence_url:  https://...            # URL of the licence / voice card
    sha256:       <64 hex chars>         # SHA-256 of the file at model_path
    version:      "1.0.0"
    tags:         [conversational, male] # optional, list of strings

Discovery order (first match wins for a duplicate id, and the winner is
logged so a shadowing pack is never silent):
  1. ``VAANI_VOICES_DIR`` (colon-separated, in order)
  2. ``<project>/voices/`` — the canonical home for new packs
  3. ``<project>/lite/models/`` — legacy home (hima-tts-nano-te lives here)

Validation is strict but non-fatal: a bad pack is recorded in
``registry.rejected`` with the exact reason (missing field, missing file,
checksum mismatch, unknown engine, duplicate id) and the server keeps
running with the good packs. ``reload()`` re-scans without a restart.

The registry NEVER imports a voice engine. It only reads manifests and
checks files exist. Engine loading is ``engine.py``'s job, lazily, on
first use of a voice.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .errors import VoiceNotFoundError, VoicePackError

log = logging.getLogger("vaani.core.registry")

SUPPORTED_ENGINES = ("hima-nano", "piper", "onnx-vits")
REQUIRED_FIELDS = (
    "id", "name", "gender", "language", "engine",
    "model_path", "config_path", "sample_rate",
    "licence", "licence_url", "sha256", "version",
)
_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Minimal YAML reader for flat manifests (no PyYAML dependency in core)
# ---------------------------------------------------------------------------

def _parse_simple_yaml(text: str) -> dict:
    """Parse the flat subset of YAML our manifests use.

    Supports ``key: value``, quoted values, ``key:`` + ``- item`` block
    lists, and ``key: [a, b]`` inline lists. Anything else raises
    VoicePackError naming the offending line. If PyYAML is installed it
    is used instead — this parser is only the dependency-free fallback.
    """
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise VoicePackError("manifest.yaml must be a YAML mapping at the top level")
        return data
    except ImportError:
        pass

    data: dict = {}
    current_key: str | None = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- ") and current_key is not None:
            item = line[2:].strip().strip("\"'")
            data.setdefault(current_key, []).append(item)
            continue
        if ":" not in line:
            raise VoicePackError(f"manifest.yaml line {lineno}: cannot parse {raw!r}")
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        current_key = key
        if not value:
            data[key] = []
            continue
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            data[key] = [v.strip().strip("\"'") for v in inner.split(",") if v.strip()] if inner else []
        else:
            data[key] = value.strip("\"'")
    return data


# ---------------------------------------------------------------------------
# VoicePack
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VoicePack:
    """A validated voice pack. Immutable after discovery."""

    id: str
    name: str
    gender: str
    language: str
    engine: str
    model_path: Path          # absolute, resolved
    config_path: Path         # absolute, resolved
    sample_rate: int
    licence: str
    licence_url: str
    sha256: str
    version: str
    tags: tuple[str, ...] = ()
    pack_dir: Path = field(default_factory=Path)

    def verify_checksum(self) -> None:
        """Raise VoicePackError if model file hash != manifest sha256."""
        digest = hashlib.sha256()
        with open(self.model_path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != self.sha256.lower():
            raise VoicePackError(
                f"voice pack '{self.id}' ({self.pack_dir}): sha256 mismatch for "
                f"{self.model_path.name}: manifest says {self.sha256[:12]}…, "
                f"file is {actual[:12]}… — re-download the model or fix the manifest"
            )

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "gender": self.gender,
            "language": self.language, "engine": self.engine,
            "sample_rate": self.sample_rate, "licence": self.licence,
            "licence_url": self.licence_url, "version": self.version,
            "tags": list(self.tags),
        }


def _validate_manifest(raw: dict, pack_dir: Path) -> VoicePack:
    """Validate one parsed manifest. Raises VoicePackError naming the problem."""
    where = f"voice pack at '{pack_dir}'"
    missing = [f for f in REQUIRED_FIELDS if f not in raw or raw[f] in (None, "")]
    if missing:
        raise VoicePackError(f"{where}: manifest.yaml is missing required field(s): {', '.join(missing)}")
    if not _ID_RE.match(str(raw["id"])):
        raise VoicePackError(f"{where}: id {raw['id']!r} must match [a-z][a-z0-9_]*")
    if str(raw["gender"]) not in ("male", "female", "neutral"):
        raise VoicePackError(f"{where}: gender must be male|female|neutral, got {raw['gender']!r}")
    if str(raw["language"]) != "te":
        raise VoicePackError(f"{where}: language must be 'te' in v2 Part 1, got {raw['language']!r}")
    if str(raw["engine"]) not in SUPPORTED_ENGINES:
        raise VoicePackError(
            f"{where}: engine must be one of {', '.join(SUPPORTED_ENGINES)}, got {raw['engine']!r}"
        )
    try:
        sample_rate = int(raw["sample_rate"])
    except (TypeError, ValueError):
        raise VoicePackError(f"{where}: sample_rate must be an integer, got {raw['sample_rate']!r}")
    if sample_rate <= 0:
        raise VoicePackError(f"{where}: sample_rate must be positive, got {sample_rate}")
    if not _SHA_RE.match(str(raw["sha256"]).lower()):
        raise VoicePackError(f"{where}: sha256 must be 64 lowercase hex chars")
    model_path = (pack_dir / str(raw["model_path"])).resolve()
    config_path = (pack_dir / str(raw["config_path"])).resolve()
    if not model_path.is_file():
        raise VoicePackError(f"{where}: model file not found: {model_path}")
    if not config_path.is_file():
        raise VoicePackError(f"{where}: config file not found: {config_path}")
    tags = raw.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    pack = VoicePack(
        id=str(raw["id"]), name=str(raw["name"]), gender=str(raw["gender"]),
        language="te", engine=str(raw["engine"]), model_path=model_path,
        config_path=config_path, sample_rate=sample_rate,
        licence=str(raw["licence"]), licence_url=str(raw["licence_url"]),
        sha256=str(raw["sha256"]).lower(), version=str(raw["version"]),
        tags=tuple(str(t) for t in tags), pack_dir=pack_dir.resolve(),
    )
    pack.verify_checksum()
    return pack


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _infer_project_root() -> Path:
    """Project root from this file's location: .../<project>/src/telugu_speech/core/registry.py."""
    return Path(__file__).resolve().parents[3]


class VoiceRegistry:
    """Discovers, validates and lists voice packs. No engine imports here."""

    def __init__(self, project_root: Path | str | None = None,
                 extra_dirs: tuple[str, ...] | None = None) -> None:
        self.project_root = Path(project_root).resolve() if project_root else _infer_project_root()
        self.extra_dirs = tuple(Path(d) for d in (extra_dirs or ()))
        self._packs: dict[str, VoicePack] = {}
        self.rejected: list[dict[str, str]] = []  # {pack_dir, reason}
        self._scanned_dirs: list[str] = []

    # -- discovery ---------------------------------------------------------
    @property
    def search_dirs(self) -> list[Path]:
        dirs: list[Path] = list(self.extra_dirs)
        dirs.append(self.project_root / "voices")
        dirs.append(self.project_root / "lite" / "models")
        # De-duplicate, keep order, keep only existing directories.
        seen: set[Path] = set()
        out: list[Path] = []
        for d in dirs:
            if d not in seen:
                seen.add(d)
                out.append(d)
        return out

    def discover(self) -> list[VoicePack]:
        """(Re-)scan all search dirs. Returns the accepted packs.

        A bad pack never raises out of here — it is appended to
        ``self.rejected`` and discovery continues. Callers that need a
        hard failure for one specific pack should use ``load_pack()``.
        """
        self._packs = {}
        self.rejected = []
        self._scanned_dirs = []
        for root in self.search_dirs:
            if not root.is_dir():
                continue
            self._scanned_dirs.append(str(root))
            for manifest in sorted(root.rglob("manifest.yaml")):
                pack_dir = manifest.parent
                try:
                    pack = self.load_pack(pack_dir)
                except VoicePackError as exc:
                    self.rejected.append({"pack_dir": str(pack_dir), "reason": str(exc)})
                    log.warning("rejecting voice pack %s: %s", pack_dir, exc)
                    continue
                if pack.id in self._packs:
                    first = self._packs[pack.id].pack_dir
                    self.rejected.append({
                        "pack_dir": str(pack_dir),
                        "reason": (
                            f"duplicate voice id '{pack.id}': already registered "
                            f"from {first}; this copy is ignored"
                        ),
                    })
                    log.warning("duplicate voice id '%s' from %s (kept %s)", pack.id, pack_dir, first)
                    continue
                self._packs[pack.id] = pack
                log.info("registered voice '%s' (%s, %s) from %s",
                         pack.id, pack.engine, pack.version, pack_dir)
        return list(self._packs.values())

    def reload(self) -> list[VoicePack]:
        """Re-scan without restarting the server. Returns accepted packs."""
        return self.discover()

    @staticmethod
    def load_pack(pack_dir: Path | str) -> VoicePack:
        """Load + validate ONE pack directory. Raises VoicePackError on any problem."""
        pack_dir = Path(pack_dir).resolve()
        manifest_path = pack_dir / "manifest.yaml"
        if not manifest_path.is_file():
            raise VoicePackError(f"voice pack at '{pack_dir}': manifest.yaml not found")
        try:
            raw = _parse_simple_yaml(manifest_path.read_text(encoding="utf-8"))
        except VoicePackError:
            raise
        except Exception as exc:  # yaml syntax error etc.
            raise VoicePackError(f"voice pack at '{pack_dir}': manifest.yaml is not valid YAML: {exc}")
        return _validate_manifest(raw, pack_dir)

    # -- lookup ------------------------------------------------------------
    def list_voices(self) -> list[VoicePack]:
        return sorted(self._packs.values(), key=lambda p: p.id)

    def list(self) -> list[VoicePack]:
        """Alias for list_voices() — the name the Part 2 API codes against.

        ``VoiceRegistry().list()`` (no args) discovers with the inferred
        project root. Note: a fresh ``VoiceRegistry()`` has NOT discovered
        yet — call ``discover()`` first, or use ``get_registry()`` which
        discovers on first call.
        """
        if not self._packs and not self._scanned_dirs:
            self.discover()
        return self.list_voices()

    def get(self, voice_id: str) -> VoicePack:
        try:
            return self._packs[voice_id]
        except KeyError:
            available = ", ".join(sorted(self._packs)) or "(none registered)"
            raise VoiceNotFoundError(
                f"voice '{voice_id}' is not registered; available voices: {available}"
            )

    def has(self, voice_id: str) -> bool:
        return voice_id in self._packs

    def status(self) -> dict:
        """For /health and diagnostics: what was scanned, kept, rejected."""
        return {
            "scanned_dirs": self._scanned_dirs,
            "voices": [p.to_dict() for p in self.list_voices()],
            "rejected": self.rejected,
        }

    def health(self) -> dict:
        """Per-voice health for /health (v2.1). Cheap: no model is loaded."""
        return {
            "voices": {
                p.id: {"status": "ok", "engine": p.engine, "version": p.version}
                for p in self.list_voices()
            },
            "rejected": list(self.rejected),
        }

    def set_extra_dirs(self, extra_dirs: tuple[str, ...]) -> None:
        """Replace the extra search dirs (from configs/vaani.yaml on reload)."""
        self.extra_dirs = tuple(Path(d) for d in extra_dirs)


# ---------------------------------------------------------------------------
# Module-level singleton — created lazily so importing core never touches disk
# ---------------------------------------------------------------------------

_registry: VoiceRegistry | None = None


def get_registry(project_root: Path | str | None = None,
                 extra_dirs: tuple[str, ...] | None = None) -> VoiceRegistry:
    """Return the process-wide registry, creating + discovering on first call."""
    global _registry
    if _registry is None:
        root = Path(project_root) if project_root else _infer_project_root()
        from .config import settings

        _registry = VoiceRegistry(root, extra_dirs or settings.extra_voices_dirs)
        _registry.discover()
    return _registry


def reset_registry() -> None:
    """Test-only helper: drop the singleton so the next get_registry() re-scans."""
    global _registry
    _registry = None


def reload_registry(extra_dirs: tuple[str, ...] | None = None) -> dict:
    """Hot reload (v2.1): drop the singleton, re-discover, return status.

    Used by POST /v1/reload. A bad pack is recorded in ``rejected`` and
    never kills the server — the status dict says exactly what was kept
    and what was rejected and why.
    """
    global _registry
    if _registry is not None and extra_dirs is not None:
        _registry.set_extra_dirs(extra_dirs)
    reset_registry()
    registry = get_registry(extra_dirs=extra_dirs)
    return registry.status()
