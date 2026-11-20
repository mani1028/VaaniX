"""Assign speaker-disjoint splits from one manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from telugu_speech.asr.dataset import assign_splits, read_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    splits = assign_splits(read_manifest(args.manifest))
    for name, rows in splits.items():
        destination = args.output_dir / f"{name}.jsonl"
        write_manifest(destination, rows)
        print(f"{name}: {len(rows)}")


if __name__ == "__main__":
    main()
