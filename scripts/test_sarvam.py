"""Optional Sarvam Saaras v4 comparison. Sends audio only when SARVAM_API_KEY is set."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import httpx


API_URL = "https://api.sarvam.ai/speech-to-text"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default="saaras:v4")
    args = parser.parse_args()
    key = os.environ.get("SARVAM_API_KEY", "").strip()
    modes = ("transcribe", "codemix")
    if not key:
        print("SARVAM_NOT_RUN")
        for mode in modes:
            print(f"mode {mode}: not measured")
        print("Set SARVAM_API_KEY to score Saaras v4 on this file. No request was sent.")
        print("API latency is separate from local RTF and is not recorded until a key is set.")
        return
    if not args.audio.is_file():
        raise SystemExit(f"missing audio {args.audio}")
    for mode in modes:
        started = time.perf_counter()
        with args.audio.open("rb") as handle:
            response = httpx.post(
                API_URL,
                headers={"api-subscription-key": key},
                data={"model": args.model, "language_code": "te-IN", "mode": mode},
                files={"file": (args.audio.name, handle, "audio/wav")},
                timeout=60.0,
            )
        elapsed = time.perf_counter() - started
        print("mode", mode)
        print("model", args.model)
        print("status", response.status_code)
        print("api_elapsed_seconds", round(elapsed, 3))
        print(response.text[:2000])


if __name__ == "__main__":
    main()
