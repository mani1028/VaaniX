"""Locate competitor checkpoints. Do not score a model that is not on disk."""

from __future__ import annotations

from pathlib import Path
import shutil


MIN_FREE_AFTER_DOWNLOAD = 8 * 1024 * 1024 * 1024

# Facts checked against public cards on 20 Nov 2026. A missing size is not guessed.
COMPETITORS: tuple[dict[str, object], ...] = (
    {
        "role": "stt",
        "model": "whisper.cpp",
        "license": "Apache-2.0 for the published Telugu ggml conversion",
        "repo": "ukta-app/indic-whisper-ggml",
        "file": "ggml-te-small.bin",
        "estimated_bytes": 181 * 1024 * 1024,
        "large": False,
        "cpu": True,
        "telugu": "small fine-tune, not the 42 MB tiny recognizer",
        "local": "models/competitors/whisper-cpp",
        "needs": "a whisper-cli binary plus ggml-te-small.bin",
        "download_block": "",
    },
    {
        "role": "stt",
        "model": "sherpa-onnx",
        "license": "Apache-2.0",
        "repo": "ippocode/indic-asr-onnx",
        "file": "whisper-tiny-te",
        "estimated_bytes": 99 * 1024 * 1024,
        "large": False,
        "cpu": True,
        "telugu": "same vasista22 tiny weights, sherpa-onnx int8 runtime",
        "local": "models/competitors/sherpa-whisper-tiny-te",
        "needs": "the sherpa-onnx package plus encoder.int8.onnx, decoder.int8.onnx, tokens.txt",
        "download_block": "",
    },
    {
        "role": "tts",
        "model": "chatterbox-telugu",
        "license": "CC-BY-4.0 weights, MIT base code",
        "repo": "shankarpandala/chatterbox-telugu",
        "file": "",
        "estimated_bytes": None,
        "large": True,
        "cpu": "card allows device=cpu; the published example uses Apple GPU",
        "telugu": "fine-tune that needs a reference recording to set the voice",
        "local": "models/competitors/chatterbox-telugu",
        "needs": "the chatterbox package, PyTorch, and a consented reference WAV",
        "download_block": "Size is not published as a small CPU pack. A reference voice is required, so it is not the same test as Srinivas or Sravani.",
    },
    {
        "role": "tts",
        "model": "laya-telugu",
        "license": "card text says CC-BY-4.0 with attribution; not re-checked as a local file",
        "repo": None,
        "file": "",
        "estimated_bytes": None,
        "large": True,
        "cpu": "card says laptop CPU",
        "telugu": "Matcha-TTS Telugu, two voices",
        "local": "models/competitors/laya-telugu",
        "needs": "the author's speak.py folder",
        "download_block": "The public card says the project is private for now, and podcast consent is still listed as open. It is not fetched.",
    },
)


def locate_competitors(root: Path) -> list[dict[str, object]]:
    rows = []
    for spec in COMPETITORS:
        path = root / str(spec["local"])
        present = path.is_dir() and any(item.is_file() for item in path.rglob("*") if ".cache" not in item.parts)
        rows.append(
            {
                "role": spec["role"],
                "model": spec["model"],
                "status": "present" if present else "not_installed",
                "license": spec["license"],
                "repo": spec["repo"],
                "cpu": spec["cpu"],
                "telugu": spec["telugu"],
                "path": str(path),
                "needs": spec["needs"],
                "measured": False,
            }
        )
    return rows


def download_decision(model: str, free_bytes: int, allow_large: bool = False) -> dict[str, object]:
    """Say whether a download is allowed. This does not contact the network."""
    spec = _spec(model)
    if spec is None:
        return {"model": model, "allowed": False, "reason": "Unknown competitor."}
    if spec["download_block"]:
        return {"model": model, "allowed": False, "reason": str(spec["download_block"])}
    if spec["repo"] is None:
        return {"model": model, "allowed": False, "reason": "No public download is recorded."}
    estimate = spec["estimated_bytes"]
    if not isinstance(estimate, int):
        return {"model": model, "allowed": False, "reason": "Download size is unknown, so it is refused."}
    if bool(spec["large"]) and not allow_large:
        return {"model": model, "allowed": False, "reason": "This pack is marked large. It is not fetched by the default command."}
    if free_bytes - estimate < MIN_FREE_AFTER_DOWNLOAD:
        return {
            "model": model,
            "allowed": False,
            "reason": "Fetching it would leave less than 8 GiB free.",
        }
    return {"model": model, "allowed": True, "reason": "Size and license are recorded. The file is still not downloaded until a fetch runs."}


def adapter_status(root: Path, model: str) -> dict[str, object]:
    """Report whether this process can run the competitor. Do not import missing packages."""
    spec = _spec(model)
    if spec is None:
        return {"model": model, "runnable": False, "reason": "Unknown competitor."}
    path = root / str(spec["local"])
    if not path.is_dir():
        return {"model": model, "runnable": False, "reason": "Weights are not on disk."}
    if model == "whisper.cpp":
        binary = shutil.which("whisper-cli")
        weights = list(path.glob("*.bin"))
        if binary is None or not weights:
            return {"model": model, "runnable": False, "reason": "whisper-cli or the ggml file is missing."}
        return {"model": model, "runnable": True, "reason": "Binary and ggml file are present. Not scored in this run."}
    if model == "sherpa-onnx":
        needed = ("encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt")
        if not all((path / name).is_file() for name in needed):
            return {"model": model, "runnable": False, "reason": "Sherpa files are incomplete."}
        try:
            import sherpa_onnx  # noqa: F401
        except Exception:
            return {"model": model, "runnable": False, "reason": "sherpa-onnx is not installed in this Python."}
        return {"model": model, "runnable": True, "reason": "Package and files are present. Not scored in this run."}
    if model == "chatterbox-telugu":
        return {
            "model": model,
            "runnable": False,
            "reason": "A consented reference WAV is required before this voice can be compared.",
        }
    if model == "laya-telugu":
        if not (path / "speak.py").is_file():
            return {"model": model, "runnable": False, "reason": "speak.py is not in the located folder."}
        return {"model": model, "runnable": True, "reason": "Folder is present. Not scored in this run."}
    return {"model": model, "runnable": False, "reason": "No adapter is registered."}


def _spec(model: str) -> dict[str, object] | None:
    for spec in COMPETITORS:
        if spec["model"] == model:
            return spec
    return None
