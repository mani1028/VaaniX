"""Save a small FLEURS Telugu test sample. CC-BY-4.0. Test split only."""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf

from telugu_speech.asr.dataset import Utterance, write_manifest
from telugu_speech.text.normalization import NormalizationConfig, has_code_mix, normalize_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/fleurs_te"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/eval.jsonl"))
    args = parser.parse_args()
    if args.count < 1 or args.count > 20:
        raise SystemExit("count must be between 1 and 20 for this sample")
    from datasets import Audio, load_dataset

    dataset = load_dataset("google/fleurs", "te_in", split="test", streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = NormalizationConfig(digit_policy="ascii")
    rows: list[Utterance] = []
    for index, example in enumerate(dataset):
        if index >= args.count:
            break
        audio = example["audio"]
        path = args.output_dir / f"fleurs_{index:03d}.wav"
        payload = audio.get("bytes")
        if not payload:
            raise RuntimeError("FLEURS row did not include audio bytes")
        path.write_bytes(payload)
        info = sf.info(path)
        raw, norm = normalize_text(str(example["transcription"]), config)
        speaker = str(example.get("speaker_id", example.get("id", index)))
        rows.append(
            Utterance(
                utt_id=f"fleurs-{index:03d}",
                audio=str(path),
                raw_text=raw,
                norm_text=norm,
                speaker_id=f"fleurs-{speaker}",
                region="unspecified",
                condition="clean",
                duration_s=info.duration,
                code_mix=has_code_mix(norm),
            )
        )
    write_manifest(args.manifest, rows)
    print(f"wrote {len(rows)} FLEURS te_in test clips to {args.manifest}")


if __name__ == "__main__":
    main()
