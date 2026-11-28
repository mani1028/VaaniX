# Vaani X — native API example (Python, no OpenAI SDK)
#
# Prerequisites:
#   1. The server is running on localhost:8000 (see README quickstart):
#        telugu-api
#      with TELUGU_API_KEYS set.
#   2. Install httpx:  pip install httpx
#   3. Put a real WAV next to this file as clip.wav for the STT part.
#
# What to expect:
#   - speech.wav is written (native /api/v1/tts/generate returns JSON
#     with an audio_url; this example downloads it with the same key).
#   - The transcript of clip.wav is printed from /api/v1/asr/transcribe.
#   - The /v1/* OpenAI-compatible routes do the same thing with less
#     code — see python_openai_sdk.py. Use whichever surface fits.
#
# Both surfaces are real and live on the same server. Old /api/v1/*
# routes are unchanged; /v1/* is additive.

import os

import httpx

BASE = os.environ.get("VAANI_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("TELUGU_API_KEYS", "choose-a-long-random-key").split(",")[0]
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# --- Health (no key needed) ---
print(httpx.get(f"{BASE}/health", timeout=10).json())

# --- Text to speech (native route) ---
resp = httpx.post(
    f"{BASE}/api/v1/tts/generate",
    headers={**HEADERS, "Content-Type": "application/json"},
    json={"text": "నమస్కారం", "language": "te", "voice": "telugu_sravani", "format": "wav"},
    timeout=60,
)
resp.raise_for_status()
body = resp.json()
print("TTS response:", body)
# The response includes an audio_url — download it with the same key.
audio_url = body["audio_url"]
if audio_url.startswith("http"):
    audio_resp = httpx.get(audio_url, headers=HEADERS, timeout=60)
else:
    audio_resp = httpx.get(f"{BASE}{audio_url}", headers=HEADERS, timeout=60)
audio_resp.raise_for_status()
with open("speech.wav", "wb") as f:
    f.write(audio_resp.content)
print("Wrote speech.wav")

# --- Speech to text (native route) ---
with open("clip.wav", "rb") as audio:
    resp = httpx.post(
        f"{BASE}/api/v1/asr/transcribe",
        headers=HEADERS,
        files={"audio": ("clip.wav", audio, "audio/wav")},
        timeout=120,
    )
resp.raise_for_status()
print("Transcript:", resp.json().get("text"))
