"""Score a manifest with one faster-whisper size. Load one model at a time."""

from __future__ import annotations

import argparse
from pathlib import Path

from telugu_speech.asr.infer import FasterWhisperBackend
from telugu_speech.evaluation.benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/eval.jsonl"))
    parser.add_argument("--model", default="ukta-app/whisper-telugu-ct2")
    parser.add_argument("--report", type=Path, default=Path("experiments/asr/whisper_report.md"))
    args = parser.parse_args()
    text = run_benchmark(
        args.manifest,
        args.report,
        [(f"faster_whisper_{args.model}", FasterWhisperBackend(args.model))],
        max_clips=200,
    )
    print(text)


if __name__ == "__main__":
    main()
