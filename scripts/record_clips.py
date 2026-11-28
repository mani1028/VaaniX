"""Record real-world Telugu clips and append them to a manifest.

Records YOUR voice (or a consenting speaker's) straight into
``data/realworld/clips/`` and appends one row per clip to
``data/realworld/manifest.csv`` — the manifest the real-world bench
(``python -m telugu_speech.evaluation.realworld``) scores.

Only record yourself, or someone who has agreed to be recorded for
this project. Clips stay on this Mac; this script never uploads
anything, and the bench never sends audio anywhere.

Recording needs ``sounddevice`` or ``pyaudio`` in the venv. If neither
is installed the script refuses to record and prints the no-mic
fallback instead (Voice Memos / QuickTime + ffmpeg) — it never writes
silence and calls it a recording.

Usage (from the repo root):
    # One clip, type the prompt on the command line:
    .venv311/bin/python scripts/record_clips.py --prompt "నమస్కారం, ఎలా ఉన్నారు?" \\
        --condition clean --speaker mani
    # Several prompts, one per line in a file:
    .venv311/bin/python scripts/record_clips.py --prompts-file prompts.txt \\
        --condition noise --speaker mani
    # Phone-quality condition records at 8 kHz automatically:
    .venv311/bin/python scripts/record_clips.py --prompt "..." --condition phone_8k --speaker mani
    # See what would be recorded, without touching the mic:
    .venv311/bin/python scripts/record_clips.py --prompts-file prompts.txt \\
        --condition fast --speaker mani --dry-run

For a clip, say exactly the prompt text — the prompt IS the reference
transcript. If you stumble or say something different, delete that WAV
and its manifest row and re-record; a wrong reference poisons the score.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telugu_speech.evaluation.realworld import CONDITIONS  # noqa: E402

MANIFEST_HEADER = ["path", "reference_text", "condition", "speaker", "region", "duration_s"]

NO_MIC_HELP = """\
No microphone recorder is installed (tried sounddevice, then pyaudio).
Nothing was recorded and no silence was written.

Install one recorder, then re-run:
    brew install portaudio          # sounddevice needs PortAudio on macOS
    .venv311/bin/pip install sounddevice

Or record without this script and add the clips by hand:
    1. Record with Voice Memos or QuickTime Player (File > New Audio Recording).
    2. Export / convert to 16 kHz mono WAV (8 kHz for phone_8k):
         ffmpeg -i in.m4a -ar 16000 -ac 1 clip.wav
         ffmpeg -i in.wav -ar 8000 -ac 1 clip_8k.wav
    3. Append a row to data/realworld/manifest.csv:
         path,reference_text,condition,speaker,region,duration_s
         data/realworld/clips/mani_clean_000.wav,<exactly what you said>,clean,mani,unspecified,
"""


def _load_prompts(prompt: str | None, prompts_file: Path | None) -> list[str]:
    if prompt is not None and prompts_file is not None:
        raise SystemExit("pass --prompt or --prompts-file, not both")
    if prompt is not None:
        if not prompt.strip():
            raise SystemExit("--prompt must not be empty")
        return [prompt.strip()]
    if prompts_file is None:
        raise SystemExit("pass --prompt or --prompts-file")
    if not prompts_file.is_file():
        raise SystemExit(f"prompts file not found: `{prompts_file}`")
    prompts = [line.strip() for line in prompts_file.read_text(encoding="utf-8").splitlines()]
    prompts = [line for line in prompts if line and not line.startswith("#")]
    if not prompts:
        raise SystemExit(f"no prompts in `{prompts_file}`")
    return prompts


def _next_index(out_dir: Path, prefix: str) -> int:
    existing = [p.stem for p in out_dir.glob(f"{prefix}_*.wav")] if out_dir.is_dir() else []
    numbers = [int(stem.rsplit("_", 1)[-1]) for stem in existing if stem.rsplit("_", 1)[-1].isdigit()]
    return (max(numbers) + 1) if numbers else 0


def _record_seconds(seconds: float, sample_rate: int) -> bytes:
    """Record mono int16 PCM. Raises RuntimeError naming the missing library."""
    try:
        import sounddevice as sd  # type: ignore[import-not-found]
    except ImportError:
        sd = None  # type: ignore[assignment]
    if sd is not None:
        frames = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
        sd.wait()
        return bytes(frames.tobytes())
    try:
        import pyaudio  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("no recorder library installed (tried sounddevice, then pyaudio)") from exc
    audio = pyaudio.PyAudio()
    try:
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)
        chunks: list[bytes] = []
        remaining = int(seconds * sample_rate)
        while remaining > 0:
            chunk = stream.read(min(1024, remaining))
            chunks.append(chunk)
            remaining -= len(chunk) // 2
        stream.stop_stream()
        stream.close()
    finally:
        audio.terminate()
    return b"".join(chunks)


def _write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def _append_manifest(manifest: Path, rows: list[dict[str, str]]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest.is_file() or manifest.stat().st_size == 0
    with manifest.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_HEADER)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record real-world Telugu clips into data/realworld/.")
    parser.add_argument("--prompt", default=None, help="One Telugu sentence, exactly as you will say it.")
    parser.add_argument("--prompts-file", type=Path, default=None, help="One Telugu sentence per line (# lines skipped).")
    parser.add_argument("--condition", required=True, choices=list(CONDITIONS), help="Real-world condition for every clip in this run.")
    parser.add_argument("--speaker", required=True, help="Speaker id, e.g. mani (use one id per person).")
    parser.add_argument("--region", default="unspecified", help="telangana | coastal_andhra | rayalaseema | unspecified.")
    parser.add_argument("--seconds", type=float, default=15, help="Max seconds per clip (default 15).")
    parser.add_argument("--sample-rate", type=int, default=None, help="Sample rate Hz (default 16000; 8000 when --condition phone_8k).")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "realworld" / "clips")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "realworld" / "manifest.csv")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be recorded; record nothing.")
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 120:
        raise SystemExit("--seconds must be 1..120")
    prompts = _load_prompts(args.prompt, args.prompts_file)
    sample_rate = args.sample_rate or (8000 if args.condition == "phone_8k" else 16000)
    if sample_rate not in {8000, 16000}:
        raise SystemExit("--sample-rate must be 8000 or 16000")
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    index = _next_index(out_dir, f"{args.speaker}_{args.condition}")
    planned: list[dict[str, str]] = []
    for offset, text in enumerate(prompts):
        name = f"{args.speaker}_{args.condition}_{index + offset:03d}.wav"
        target = out_dir / name
        try:
            display = str(target.relative_to(ROOT))
        except ValueError:
            display = str(target)
        planned.append(
            {"path": display, "reference_text": text, "condition": args.condition,
             "speaker": args.speaker, "region": args.region, "duration_s": ""}
        )
    if args.dry_run:
        print(f"would record {len(planned)} clip(s), {args.seconds:g}s max each, {sample_rate} Hz, condition {args.condition}:")
        for row in planned:
            print(f"  {row['path']}  <-  {row['reference_text']}")
        print(f"would append {len(planned)} row(s) to {manifest}")
        return 0
    print("Only record yourself or someone who agreed to be recorded for this project.")
    print(f"Condition: {args.condition} ({sample_rate} Hz). Say EXACTLY the prompt text.")
    recorded: list[dict[str, str]] = []
    for row in planned:
        input(f"\nReady: {row['reference_text']}\nPress Enter, then speak (up to {args.seconds:g}s)... ")
        try:
            pcm = _record_seconds(args.seconds, sample_rate)
        except RuntimeError:
            print(NO_MIC_HELP, file=sys.stderr)
            return 2
        _write_wav(out_dir / Path(row["path"]).name, pcm, sample_rate)
        recorded.append(row)
        print(f"saved {row['path']}")
    _append_manifest(manifest, recorded)
    print(f"\nappended {len(recorded)} row(s) to {manifest}")
    print("Score them with:")
    print(f"  .venv311/bin/python -m telugu_speech.evaluation.realworld --asr-model telugu-asr-micro --manifest {manifest.relative_to(ROOT) if manifest.is_relative_to(ROOT) else manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
