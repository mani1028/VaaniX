"""Training entry points stay closed until a checkpoint and GPU job exist."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


class TrainingDeferred(RuntimeError):
    """Raised when a training command is invoked during the baseline phase."""


REQUIRED_FINETUNE_KEYS = (
    "optimizer",
    "learning_rate",
    "decoder",
    "speaker_independent",
    "datasets",
    "held_out",
)


def load_training_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training config must be a mapping")
    missing = [key for key in REQUIRED_FINETUNE_KEYS if key not in payload]
    if missing:
        raise ValueError(f"training config missing {missing}")
    if payload.get("speaker_independent") is not True:
        raise ValueError("speaker_independent must be true")
    return payload


def _utt_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    found: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict) and payload.get("utt_id"):
            found.add(str(payload["utt_id"]))
    return found


def assert_eval_not_in_train(payload: dict[str, object]) -> None:
    held = payload.get("held_out")
    if not isinstance(held, list) or "fleurs_te" not in held:
        raise ValueError("held_out must include fleurs_te")
    datasets = payload.get("datasets")
    if isinstance(datasets, list):
        for name in datasets:
            if "fleur" in str(name).lower():
                raise ValueError("FLEURS audio is evaluation-only")
    eval_manifest = Path(str(payload.get("held_out_manifest", "data/manifests/eval.jsonl")))
    blocked = _utt_ids(eval_manifest)
    train_manifest = payload.get("train_manifest")
    if not train_manifest or not blocked:
        return
    overlap = _utt_ids(Path(str(train_manifest))) & blocked
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"train manifest includes held-out utterances: {sample}")


def train(config_path: Path) -> None:
    payload = load_training_config(config_path)
    assert_eval_not_in_train(payload)
    raise TrainingDeferred(
        "Phase 1 records the fine-tune config and does not launch training."
    )
