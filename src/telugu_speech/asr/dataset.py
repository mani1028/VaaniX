"""Manifest schema and speaker-disjoint splits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from telugu_speech.text.normalization import has_code_mix

REGIONS = ("telangana", "coastal_andhra", "rayalaseema", "code_mix", "unspecified")
CONDITIONS = ("clean", "noisy", "phone_8k", "conversational", "unspecified")
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    audio: str
    raw_text: str
    norm_text: str
    speaker_id: str
    region: str
    condition: str
    duration_s: float | None = None
    code_mix: bool = False

    def __post_init__(self) -> None:
        if not self.utt_id or not self.speaker_id:
            raise ValueError("utt_id and speaker_id are required")
        if self.region not in REGIONS:
            raise ValueError(f"unknown region {self.region}")
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown condition {self.condition}")
        if Path(self.audio).name != self.audio and ".." in Path(self.audio).parts:
            raise ValueError("audio path must not contain parent traversal")


def utterance_from_mapping(payload: dict[str, object]) -> Utterance:
    required = ("utt_id", "audio", "raw_text", "norm_text", "speaker_id", "region", "condition")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"manifest row missing {missing}")
    raw_text = str(payload["raw_text"])
    norm_text = str(payload["norm_text"])
    duration = payload.get("duration_s")
    return Utterance(
        utt_id=str(payload["utt_id"]),
        audio=str(payload["audio"]),
        raw_text=raw_text,
        norm_text=norm_text,
        speaker_id=str(payload["speaker_id"]),
        region=str(payload["region"]),
        condition=str(payload["condition"]),
        duration_s=None if duration is None else float(duration),
        code_mix=bool(payload.get("code_mix", has_code_mix(norm_text) or str(payload["region"]) == "code_mix")),
    )


def write_manifest(path: Path, rows: list[Utterance]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def read_manifest(path: Path) -> list[Utterance]:
    rows: list[Utterance] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            rows.append(utterance_from_mapping(payload))
    return rows


def assert_speaker_disjoint(splits: dict[str, list[Utterance]]) -> None:
    unknown = set(splits) - set(SPLITS)
    if unknown:
        raise ValueError(f"unknown splits {sorted(unknown)}")
    owners: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            previous = owners.get(row.speaker_id)
            if previous is not None and previous != split:
                raise ValueError(
                    f"speaker {row.speaker_id} appears in {previous} and {split}"
                )
            owners[row.speaker_id] = split


def assign_splits(rows: list[Utterance], validation_ratio: float = 0.1, test_ratio: float = 0.1) -> dict[str, list[Utterance]]:
    """Assign whole speakers by a stable hash. Ratios are targets, not exact counts."""
    if not 0 < validation_ratio < 1 or not 0 < test_ratio < 1 or validation_ratio + test_ratio >= 1:
        raise ValueError("split ratios must leave a nonempty train share")
    buckets: dict[str, list[Utterance]] = {name: [] for name in SPLITS}
    speakers = sorted({row.speaker_id for row in rows})
    placement: dict[str, str] = {}
    for speaker in speakers:
        bucket = _stable_unit(speaker)
        if bucket < test_ratio:
            placement[speaker] = "test"
        elif bucket < test_ratio + validation_ratio:
            placement[speaker] = "validation"
        else:
            placement[speaker] = "train"
    for row in rows:
        buckets[placement[row.speaker_id]].append(row)
    assert_speaker_disjoint(buckets)
    return buckets


def _stable_unit(speaker_id: str) -> float:
    total = 0
    for index, char in enumerate(speaker_id.encode("utf-8")):
        total = (total + (char + 1) * (index + 3)) % 10000
    return total / 10000
