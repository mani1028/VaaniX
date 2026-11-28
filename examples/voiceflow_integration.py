# Vaani X + VoiceFlow — voice-broadcast integration example
#
# VoiceFlow is a single-admin voice broadcast portal for India
# (FastAPI backend, React frontend, Plivo/Exotel for calls). This
# example shows the Vaani X half of a broadcast: turn each recipient's
# Telugu message into audio with the OpenAI-compatible Vaani X API,
# then hand the audio to the telephony step.
#
# Prerequisites:
#   1. Vaani X is running on localhost:8000 (see README quickstart):
#        telugu-api
#      with TELUGU_API_KEYS set.
#   2. Install httpx:  pip install httpx
#
# What to expect:
#   - One WAV per recipient is written to ./broadcast_audio/
#     (e.g. broadcast_audio/919999999999.wav, 24 kHz, Srinivas voice).
#   - A call plan (JSON) is printed: phone + audio file per recipient.
#   - NO call is placed by this script. place_call() is a clearly
#     marked hook: wire it to your Plivo/Exotel call there. Vaani X
#     does not know your telephony credentials and never asks for them.
#
# Telugu only. Text is capped at 4096 chars per /v1/audio/speech
# request (2000 on the native route); longer messages must be split
# into sentences first. `speed` is accepted and currently ignored.

import json
import os
from pathlib import Path

import httpx

BASE = os.environ.get("VAANI_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("TELUGU_API_KEYS", "choose-a-long-random-key").split(",")[0]
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

OUT_DIR = Path("broadcast_audio")
OUT_DIR.mkdir(exist_ok=True)

# Example broadcast list: replace with your own recipients.
# Phone numbers are examples only — never commit real numbers.
RECIPIENTS = [
    {"phone": "+919999999999", "text": "నమస్కారం, మీ అపాయింట్మెంట్ రేపు ఉదయం పది గంటలకు.", "voice": "srinivas"},
    {"phone": "+918888888888", "text": "నమస్కారం, మీ ఆర్డర్ రేపు డెలివరీ అవుతుంది.", "voice": "sravani"},
]


def generate_audio(text: str, voice: str) -> bytes:
    """One Vaani X TTS call. Returns WAV bytes. Raises on HTTP error."""
    resp = httpx.post(
        f"{BASE}/v1/audio/speech",
        headers=HEADERS,
        json={"model": "vaani-tts-hima", "voice": voice, "input": text, "response_format": "wav"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.content


def place_call(phone: str, audio_path: Path) -> dict:
    """HOOK — not implemented here. Wire this to your Plivo/Exotel
    call in VoiceFlow (your telephony credentials stay in VoiceFlow).
    Return whatever your telephony step returns, e.g. a call UUID."""
    return {"phone": phone, "audio": str(audio_path), "status": "not_placed — wire place_call() to Plivo/Exotel"}


def main() -> None:
    plan = []
    for recipient in RECIPIENTS:
        audio = generate_audio(recipient["text"], recipient["voice"])
        path = OUT_DIR / f"{recipient['phone'].lstrip('+')}.wav"
        path.write_bytes(audio)
        plan.append(place_call(recipient["phone"], path))
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    print(f"Wrote {len(plan)} audio file(s) to {OUT_DIR}/ — no calls were placed.")


if __name__ == "__main__":
    main()
