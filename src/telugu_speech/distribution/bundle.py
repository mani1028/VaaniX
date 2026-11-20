"""Build a zip of the host code plus the voice packs the caller selected.

The archive leaves out virtualenvs, experiments, and secrets.
Srinivas and Sravani share the Hima weights. A download includes that pack once.
"""

from __future__ import annotations

import json
from pathlib import Path
import zipfile
from typing import BinaryIO

from telugu_speech.tts.voices import PACK, SAMPLE_DIR, VOICES, pack_ready

CORE_DIRS = ("src", "tests", "docs")
CORE_FILES = (
    "index.html",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    ".env.example",
    ".gitignore",
    "lite/README.md",
    "scripts/check_dependencies.py",
    "scripts/check_static.py",
    "scripts/scan_secrets.py",
    "scripts/fetch_lite_voice.py",
)
SKIP_PARTS = {
    "__pycache__",
    ".cache",
    ".git",
    ".venv",
    ".venv311",
    ".pytest_cache",
    ".engineering",
    "experiments",
}


def pack_for_voice(voice_id: str) -> tuple[Path, str] | None:
    if voice_id in VOICES:
        return PACK, "hima-tts-nano-te"
    return None


def normalize_voice_ids(voice_ids: str | list[str]) -> list[str]:
    """Keep catalog ids, in order, without duplicates. Raise KeyError if none remain."""
    raw = voice_ids.split(",") if isinstance(voice_ids, str) else voice_ids
    chosen: list[str] = []
    for item in raw:
        voice_id = str(item).strip()
        if not voice_id:
            continue
        if voice_id not in VOICES:
            raise KeyError(voice_id)
        if voice_id not in chosen:
            chosen.append(voice_id)
    if not chosen:
        raise KeyError("voices")
    return chosen


def _sample_names(voice_ids: list[str], include_pack_siblings: bool) -> list[str]:
    if include_pack_siblings and len(voice_ids) == 1:
        return [str(spec["sample"]) for spec in VOICES.values()]
    return [str(VOICES[voice_id]["sample"]) for voice_id in voice_ids]


def _include_file(root: Path, path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    if path.name == ".env" or path.suffix == ".pyc":
        return False
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return not any(part in SKIP_PARTS for part in path.parts)


def bundle_files(root: Path, voice_ids: str | list[str], include_pack_siblings: bool = False) -> list[tuple[Path, str]]:
    """Return (source path, zip name) pairs. Raises FileNotFoundError if a pack is absent."""
    ids = normalize_voice_ids(voice_ids)
    packs: list[tuple[Path, str]] = []
    for voice_id in ids:
        chosen = pack_for_voice(voice_id)
        if chosen is None:
            raise KeyError(voice_id)
        if chosen not in packs:
            packs.append(chosen)
        ready = pack_ready(root)
        if not ready:
            raise FileNotFoundError(f"Voice pack for {voice_id} is not in this project.")
    files: list[tuple[Path, str]] = []
    for relative in CORE_FILES:
        path = root / relative
        if _include_file(root, path):
            files.append((path, relative))
    for relative in CORE_DIRS:
        base = root / relative
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if _include_file(root, path):
                files.append((path, path.relative_to(root).as_posix()))
    for pack_path, _pack_id in packs:
        pack_root = root / pack_path
        for path in pack_root.rglob("*"):
            if _include_file(root, path):
                files.append((path, path.relative_to(root).as_posix()))
    for sample in _sample_names(ids, include_pack_siblings):
        path = root / SAMPLE_DIR / sample
        if _include_file(root, path):
            files.append((path, path.relative_to(root).as_posix()))
    return files


def write_voice_bundle(root: Path, voice_ids: str | list[str], destination: BinaryIO) -> None:
    exact = isinstance(voice_ids, list) or (isinstance(voice_ids, str) and "," in voice_ids)
    ids = normalize_voice_ids(voice_ids)
    packs: list[str] = []
    for voice_id in ids:
        chosen = pack_for_voice(voice_id)
        if chosen is None:
            raise KeyError(voice_id)
        if chosen[1] not in packs:
            packs.append(chosen[1])
    files = bundle_files(root, ids, include_pack_siblings=not exact)
    listed = ", ".join(ids)
    pack_list = ", ".join(packs)
    if exact:
        marker = {
            "voice": ids[0],
            "voices": ids,
            "packs": packs,
            "contains": "core code plus the selected voice packs",
        }
        scope = f"The hosted server speaks only: {listed}.\nPacks included: {pack_list}.\n"
    else:
        marker = {
            "voice": ids[0],
            "pack": packs[0],
            "contains": "core code plus this voice pack only",
        }
        scope = f"The default voice is {ids[0]}. Pack included: {packs[0]}.\n"
    host = (
        "This folder is the Telugu speech code plus the voices you picked.\n"
        f"{scope}"
        "Speech runs on your computer. Nothing is sent to an online voice service.\n\n"
        "1. Unzip this folder.\n"
        "2. Install with Python 3.11:\n"
        "     python -m pip install -e \".[host]\"\n"
        "3. On your own computer, no password is needed:\n"
        "     BHASHA_ENABLE_PLATFORM=0 TELUGU_SPEECH_AUTH=local_open \\\n"
        "       python -m uvicorn telugu_speech.api.main:app --host 127.0.0.1 --port 8000\n"
        "   Then open http://127.0.0.1:8000\n\n"
        "From another app, after the server is running:\n"
        "  Make it speak: POST /api/v1/tts/generate\n"
        "  Make it listen: POST /api/v1/asr/transcribe with a WAV file\n"
        "Listening needs a separate folder, models/telugu-asr-micro. It is not in this download.\n"
        "One speak engine stays loaded. Extra copies appear only while calls overlap, up to TELUGU_SPEECH_SLOTS. Set TELUGU_SPEECH_SLOTS=1 on a very small machine.\n\n"
        "A password is only for the server. If other computers can open the address, set\n"
        "TELUGU_SPEECH_AUTH=required and TELUGU_API_KEYS to a password you choose.\n\n"
        "From your own Python code, do not start the server and do not set a key:\n"
        "  from telugu_speech.tts.voices import backend_for_voice\n"
        "  from telugu_speech.asr.infer import FasterWhisperBackend\n"
    )
    stored = {".onnx", ".safetensors", ".wav"}
    with zipfile.ZipFile(destination, "w") as archive:
        archive.writestr("voice_pack.json", json.dumps(marker, indent=2) + "\n")
        archive.writestr("HOST.txt", host)
        for path, name in files:
            mode = zipfile.ZIP_STORED if path.suffix in stored else zipfile.ZIP_DEFLATED
            archive.write(path, name, compress_type=mode)
