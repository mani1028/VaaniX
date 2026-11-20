"""End-to-end offline STT then TTS. No API keys. Loads models from disk only."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from telugu_speech.asr.infer import FasterWhisperBackend
from telugu_speech.tts.infer import CompactMmsTtsBackend, MmsTtsBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, default=Path("data/raw/fleurs_te/fleurs_000.wav"))
    parser.add_argument("--text", default="")
    parser.add_argument("--asr", default="models/telugu-asr-micro")
    parser.add_argument("--tts", default="models/telugu-tts-micro")
    parser.add_argument("--out", type=Path, default=Path("experiments/tts/e2e.wav"))
    args = parser.parse_args()

    asr_bytes = sum(path.stat().st_size for path in Path(args.asr).rglob("*") if path.is_file())
    tts_dir = Path(args.tts)
    tts_bytes = sum(path.stat().st_size for path in tts_dir.rglob("*") if path.is_file()) if tts_dir.is_dir() else 0
    print("MODEL_SIZE_BYTES")
    print("stt", asr_bytes)
    print("tts", tts_bytes)
    print("total", asr_bytes + tts_bytes)

    text = args.text.strip()
    if not text:
        if not args.audio.is_file():
            raise SystemExit(f"missing audio {args.audio}")
        started = time.perf_counter()
        text = FasterWhisperBackend(args.asr).transcribe(args.audio)
        print("STT_SECONDS", round(time.perf_counter() - started, 3))
        print("STT_TEXT", text)
    else:
        print("STT_TEXT", text)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tts_path = Path(args.tts)
    if (tts_path / "weights.int8.npz").is_file():
        backend = CompactMmsTtsBackend(args.tts)
    else:
        backend = MmsTtsBackend(args.tts)
    written = backend.synthesize(text, None, "", args.out)
    print("TTS_SECONDS", round(time.perf_counter() - started, 3))
    print("TTS_WAV", written)


if __name__ == "__main__":
    main()
