"""Download + SHA-verify ASR models from configs/models.yaml.

Usage:
    python scripts/fetch_models.py --list
    python scripts/fetch_models.py --model vasista22-whisper-telugu-small --dry-run
    python scripts/fetch_models.py --model vasista22-whisper-telugu-small

The script only — no weights are downloaded by writing this file, and
nothing in this task ran a download. A model with ``revision: TBD`` or
``sha256: TBD`` in the catalogue REFUSES to download: fill the pinned
Hugging Face revision and the expected SHA-256 first (from the model
card / a verified download), then re-run. A download whose SHA does not
match is reported as NOT trusted and must be deleted by hand — the
script never silently keeps or uses it.

Faster-whisper models are fetched as CTranslate2 INT8 conversions via
``huggingface_hub.snapshot_download`` into ``models/<id>/``. The
IndicConformer 120M/600M entries are sherpa-onnx ONNX exports, not NeMo
checkpoints — the script fetches the ONNX repo named in the catalogue.
Local models (``repo: local``) are never fetched.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

CATALOGUE = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"
MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"


def load_catalogue(path: Path = CATALOGUE) -> list[dict[str, object]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError(f"{path} must contain a 'models' list")
    return payload["models"]  # type: ignore[return-value]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_tree(root: Path, expected: str) -> bool:
    """SHA-256 over every file under root, sorted by relative path.

    The catalogue SHA is the hash of the concatenated per-file hashes
    (one line per file: ``<sha256>  <relative path>``). Record it the
    same way after a verified download: this script prints the value
    it computed, so the first verified run produces the number to pin.
    """
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        return False
    lines = [f"{sha256_file(p)}  {p.relative_to(root)}" for p in files]
    combined = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return combined == expected


def combined_sha(root: Path) -> str:
    files = sorted(p for p in root.rglob("*") if p.is_file())
    lines = [f"{sha256_file(p)}  {p.relative_to(root)}" for p in files]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch + SHA-verify ASR models.")
    parser.add_argument("--model", help="Model id from configs/models.yaml.")
    parser.add_argument("--list", action="store_true", help="List catalogue models and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be fetched, download nothing.")
    args = parser.parse_args(argv)

    try:
        models = load_catalogue()
    except (OSError, ValueError) as exc:
        print(f"catalogue error: {exc}", file=sys.stderr)
        return 2

    if args.list:
        for model in models:
            print(f"{model.get('id')}: {model.get('status')} — {model.get('repo')} ({model.get('size_mb')} MB)")
        return 0
    if not args.model:
        parser.error("--model is required (or --list)")
    model = next((m for m in models if m.get("id") == args.model), None)
    if model is None:
        print(f"unknown model {args.model!r}; see --list", file=sys.stderr)
        return 2
    if model.get("repo") == "local":
        print(f"{args.model} is a local model ({model.get('local_path')}); nothing to fetch.")
        return 0
    revision = str(model.get("revision", "TBD"))
    expected_sha = str(model.get("sha256", "TBD"))
    if revision == "TBD" or expected_sha == "TBD":
        print(
            f"REFUSED: {args.model} has revision/sha256 TBD in configs/models.yaml.\n"
            "Fill the pinned Hugging Face revision and the expected SHA-256 from the "
            "model card / a verified download first, then re-run. Nothing was downloaded.",
            file=sys.stderr,
        )
        return 2
    destination = MODELS_ROOT / str(args.model)
    print(f"model: {args.model}")
    print(f"repo: {model.get('repo')} revision: {revision}")
    print(f"destination: {destination}")
    print(f"expected combined sha256: {expected_sha}")
    if args.dry_run:
        print("dry-run: nothing downloaded.")
        return 0
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed; pip install huggingface-hub first.", file=sys.stderr)
        return 2
    snapshot_download(
        repo_id=str(model.get("repo")),
        revision=revision,
        local_dir=str(destination),
    )
    if verify_tree(destination, expected_sha):
        print(f"OK: {args.model} verified (combined sha256 matches).")
        return 0
    print(
        f"SHA MISMATCH for {args.model}: computed {combined_sha(destination)}, "
        f"expected {expected_sha}. The download is NOT trusted — delete it and re-check "
        "the pinned revision before retrying.",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
