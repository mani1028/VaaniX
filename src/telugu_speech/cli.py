"""Benchmark command line."""

from __future__ import annotations

import argparse
from pathlib import Path

from telugu_speech.evaluation.benchmark import backends_from_config, load_eval_config, run_benchmark
from telugu_speech.evaluation.detection_report import write_detection_report


def main() -> None:
    parser = argparse.ArgumentParser(prog="telugu-bench")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Score configured ASR backends on a manifest")
    run.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_eval_config(args.config)
    manifest = Path(str(config["manifest"]))
    report = Path(str(config["report"]))
    if not manifest.is_file():
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "# Telugu Speech Benchmark\n\nNo manifest was found at "
            f"`{manifest}`. Add JSONL rows before scoring models.\n",
            encoding="utf-8",
        )
        print(report)
        return
    text = run_benchmark(
        manifest,
        report,
        backends_from_config(config),
        max_clips=int(config["max_clips"]),
    )
    detection_report = config.get("detection_report")
    if detection_report:
        print(write_detection_report(manifest, Path(str(detection_report))))
    print(text)


if __name__ == "__main__":
    main()
