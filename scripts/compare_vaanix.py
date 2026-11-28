#!/usr/bin/env python3
"""Pair Vaani X with another ONNX voice pack on the same 12 sentences.

Vaani X is the server already running (Srinivas and Sravani). The other
system is ``--new-model``, a Hima-style folder with config.yaml and an ONNX
tree. Each sentence is spoken by both systems. Each WAV is then transcribed
by Vaani X's own recognizer.

The CER is relative. Both clips are scored by the same Whisper, so the gap
between the two systems is meaningful. The number is not an absolute CER for
either recipe, because the recognizer is not a human listener and it makes
its own mistakes. The voices are also different people, so listening still
decides which voice you prefer.

Writes paired WAVs and report.json under compare_out/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SENTENCES = (
    "నమస్కారం. ఈ రోజు వాతావరణం బాగుంది.",
    "రేపు ఉదయం పది గంటలకు కాల్ చేయండి.",
    "మీ ఆర్డర్ ఐదు వందల రూపాయలు. డెలివరీ రేపు.",
    "హైదరాబాద్ నుంచి విజయవాడకు రైలు ఎప్పుడు బయలుదేరుతుంది?",
    "దయచేసి తలుపు మూసి లైట్ ఆఫ్ చేయండి.",
    "నాకు ఒక కప్పు టీ ఇవ్వండి, చక్కెర వద్దు.",
    "సమావేశం మూడు గంటలకు ఆఫీసులో జరుగుతుంది.",
    "ఈ రోడ్డు మరమ్మత్తులో ఉంది. దయచేసి వేరే దారి వెళ్లండి.",
    "పిల్లలు పాఠశాల నుంచి సాయంత్రం నాలుగు గంటలకు వస్తారు.",
    "మీ ఖాతాలో రెండు వేల రూపాయలు మిగిలి ఉన్నాయి.",
    "వాన పడుతోంది, గొడుగు తీసుకుని రండి.",
    "ధన్యవాదాలు. మళ్ళీ మాట్లాడుదాం.",
)

VAANI_VOICES = ("srinivas", "sravani")
NEW_VOICE_FOR = {"srinivas": ("telugu_srinivas", "srinivas", "rasa_m", "male"), "sravani": ("telugu_sravani", "sravani", "rasa_f", "female")}


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref = [char for char in reference if not char.isspace()]
    hyp = [char for char in hypothesis if not char.isspace()]
    rows, cols = len(ref) + 1, len(hyp) + 1
    distance = list(range(cols))
    for i in range(1, rows):
        previous, distance[0] = distance[0], i
        for j in range(1, cols):
            insert = distance[j - 1] + 1
            delete = distance[j] + 1
            substitute = previous + (ref[i - 1] != hyp[j - 1])
            previous = distance[j]
            distance[j] = min(insert, delete, substitute)
    if not ref:
        return 0.0 if not hyp else 1.0
    return distance[-1] / len(ref)


def _request(url: str, data: bytes, headers: dict[str, str], timeout: float) -> bytes:
    for _attempt in range(8):
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code != 429:
                raise RuntimeError(f"{url} returned {exc.code}: {detail}") from exc
            match = re.search(r"Retry after (\d+)", detail)
            wait = int(match.group(1)) + 1 if match else 20
            print(f"    rate limit, waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"{url} stayed rate-limited")


def speak_vaani(base: str, key: str, text: str, voice: str, destination: Path, timeout: float) -> None:
    body = json.dumps(
        {"model": "vaani-tts", "input": text, "voice": voice, "response_format": "wav"}
    ).encode("utf-8")
    audio = _request(
        base.rstrip("/") + "/v1/audio/speech",
        body,
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout,
    )
    destination.write_bytes(audio)


def transcribe_vaani(base: str, key: str, wav_path: Path, timeout: float) -> str:
    boundary = "----vaani-compare"
    payload = wav_path.read_bytes()
    chunks = [
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="model"\r\n\r\n',
        b"vaani-asr-small\r\n",
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{wav_path.name}"\r\n'.encode(),
        b"Content-Type: audio/wav\r\n\r\n",
        payload,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    raw = _request(
        base.rstrip("/") + "/v1/audio/transcriptions",
        b"".join(chunks),
        {"Authorization": f"Bearer {key}", "Content-Type": f"multipart/form-data; boundary={boundary}"},
        timeout,
    )
    parsed = json.loads(raw.decode("utf-8"))
    return str(parsed.get("text", "")).strip()


def resolve_pack(path: Path) -> tuple[Path, str]:
    """Return the model folder and kind. Accepts a Hima pack or an indic-tiny int8 folder."""
    candidate = path.expanduser().resolve()
    if (candidate / "config.json").is_file() and (candidate / "onnx" / "text_encoder.onnx").is_file():
        return candidate, "matcha-int8"
    if (candidate / "config.yaml").is_file():
        return candidate, "hima"
    parent = candidate.parent
    if (parent / "config.yaml").is_file() and candidate.name in {"int8", "onnx"}:
        return parent, "hima"
    raise SystemExit(
        f"{candidate} is not a known voice pack. Expected config.json with onnx/*.onnx, or a Hima config.yaml."
    )


class _Speaker:
    def __init__(self, engine: object, kind: str, paired_voices: dict[str, str]) -> None:
        self.engine = engine
        self.kind = kind
        self.paired_voices = paired_voices

    def speak(self, text: str, voice: str, seed: int, destination: Path) -> None:
        if self.kind == "matcha-int8":
            audio = self.engine.synthesize(text, voice=voice, seed=seed)
            self.engine.save_wav(audio, destination)
        else:
            audio = self.engine.say(text, voice=voice, seed=seed)
            self.engine.save(audio, str(destination))
        if not destination.is_file():
            raise RuntimeError(f"New model did not write {destination}")


def load_new_voices(model_dir: Path, kind: str) -> _Speaker:
    if kind == "matcha-int8":
        from indic_tiny_tts import TinyTTSOnnx

        engine = TinyTTSOnnx(model_dir, threads=2)
        chosen = {}
        for vaani_name, options in NEW_VOICE_FOR.items():
            match = next((name for name in options if name in engine.voices), None)
            if match is None:
                raise SystemExit(f"New model voices are {sorted(engine.voices)}. None of {options} is present for {vaani_name}.")
            chosen[vaani_name] = match
        return _Speaker(engine, kind, chosen)
    sys.path.insert(0, str(model_dir))
    from hima_onnx import OnnxTTS

    engine = OnnxTTS(str(model_dir), precision="int8", threads=2)
    names = list(engine.voices)
    return _Speaker(engine, kind, {vaani: names[index % len(names)] for index, vaani in enumerate(VAANI_VOICES)})


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Vaani X with another ONNX voice pack.")
    parser.add_argument("--new-model", required=True, type=Path, help="Hima pack root, or its int8/ folder.")
    parser.add_argument("--base-url", default=os.environ.get("VAANI_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.environ.get("VAANI_API_KEY", ""))
    parser.add_argument("--out", type=Path, default=Path("compare_out"))
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("Set VAANI_API_KEY to the same key as TELUGU_API_KEYS.")

    model_dir, precision = resolve_pack(args.new_model)
    engine = load_new_voices(model_dir, precision)
    new_voices = tuple(engine.paired_voices[name] for name in VAANI_VOICES)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"Vaani X voices: {', '.join(VAANI_VOICES)}")
    print(f"New model voices: {', '.join(new_voices)}  ({model_dir}, {precision})")
    print("CER is relative: both WAVs are scored by Vaani X Whisper, not by a human.\n")

    for index, text in enumerate(SENTENCES, start=1):
        for slot, (vaani_voice, new_voice) in enumerate(zip(VAANI_VOICES, new_voices)):
            pair = out / f"{index:02d}-{vaani_voice}"
            pair.mkdir(parents=True, exist_ok=True)
            vaani_path = pair / "vaanix.wav"
            new_path = pair / "new.wav"
            print(f"[{index}/12] {vaani_voice} vs {new_voice}")
            speak_vaani(args.base_url, args.api_key, text, vaani_voice, vaani_path, args.timeout)
            engine.speak(text, new_voice, index * 10 + slot, new_path)
            vaani_text = transcribe_vaani(args.base_url, args.api_key, vaani_path, args.timeout)
            new_text = transcribe_vaani(args.base_url, args.api_key, new_path, args.timeout)
            row = {
                "index": index,
                "text": text,
                "vaanix_voice": vaani_voice,
                "new_voice": new_voice,
                "vaanix_wav": str(vaani_path),
                "new_wav": str(new_path),
                "vaanix_transcript": vaani_text,
                "new_transcript": new_text,
                "vaanix_cer": round(character_error_rate(text, vaani_text), 4),
                "new_cer": round(character_error_rate(text, new_text), 4),
            }
            rows.append(row)
            print(f"    CER  vaanix {row['vaanix_cer']:.3f}   new {row['new_cer']:.3f}")

    summary = {
        "note": (
            "CER is relative. Both systems were transcribed by Vaani X Whisper. "
            "The gap between the two columns is the useful number. Neither column "
            "is the recipe's absolute CER. Voices are different speakers; listen to the paired WAVs."
        ),
        "sentences": len(SENTENCES),
        "vaanix_voices": list(VAANI_VOICES),
        "new_voices": list(new_voices),
        "new_model": str(model_dir),
        "precision": precision,
        "vaanix_cer_mean": round(mean([row["vaanix_cer"] for row in rows]) or 0.0, 4),
        "new_cer_mean": round(mean([row["new_cer"] for row in rows]) or 0.0, 4),
        "rows": rows,
    }
    report = out / "report.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nMean CER (lower is closer to the sentence, as this Whisper heard it)")
    print(f"  vaanix  {summary['vaanix_cer_mean']:.3f}")
    print(f"  new     {summary['new_cer_mean']:.3f}")
    print(f"Paired WAVs and {report}")


if __name__ == "__main__":
    main()
