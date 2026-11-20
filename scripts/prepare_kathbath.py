"""Build a Telugu manifest from a local Kathbath-style directory. Does not download data."""

from __future__ import annotations

import argparse
from pathlib import Path

from telugu_speech.asr.dataset import Utterance, write_manifest
from telugu_speech.audio.preprocessing import duration_seconds, prepare_utterance
from telugu_speech.text.normalization import NormalizationConfig, has_code_mix, normalize_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--region", default="unspecified")
    parser.add_argument("--condition", default="clean")
    args = parser.parse_args()
    if not args.source.is_dir():
        raise SystemExit(f"source directory not found: {args.source}")
    config = NormalizationConfig()
    rows: list[Utterance] = []
    for audio in sorted(args.source.rglob("*.wav")):
        transcript_path = audio.with_suffix(".txt")
        if not transcript_path.is_file():
            continue
        raw, norm = normalize_text(transcript_path.read_text(encoding="utf-8"), config)
        samples, _spans = prepare_utterance(audio)
        speaker = audio.parent.name
        rows.append(
            Utterance(
                utt_id=audio.stem,
                audio=str(audio),
                raw_text=raw,
                norm_text=norm,
                speaker_id=speaker,
                region=args.region,
                condition=args.condition,
                duration_s=duration_seconds(samples, 16000),
                code_mix=has_code_mix(norm),
            )
        )
    write_manifest(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
