"""Transcribe one WAV with faster-whisper and print timing. Does not train."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import soundfile as sf

from telugu_speech.asr.infer import FasterWhisperBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default="ukta-app/whisper-telugu-ct2")
    args = parser.parse_args()
    info = sf.info(args.audio)
    backend = FasterWhisperBackend(args.model)
    started = time.perf_counter()
    transcript = backend.transcribe(args.audio)
    elapsed = time.perf_counter() - started
    print("TELUGU TRANSCRIPTION")
    print(transcript)
    print("PERFORMANCE")
    print("model", args.model)
    print("audio_seconds", round(info.duration, 2))
    print("inference_seconds", round(elapsed, 2))
    if info.duration > 0:
        print("rtf", round(elapsed / info.duration, 3))


if __name__ == "__main__":
    main()
