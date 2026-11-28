# Vaani X — OpenAI SDK example (Python)
#
# Prerequisites:
#   1. The server is running on localhost:8000 (see README quickstart):
#        telugu-api
#      with TELUGU_API_KEYS set, e.g.:
#        export TELUGU_API_KEYS='choose-a-long-random-key'
#   2. Install the SDK:  pip install openai
#   3. Put a real WAV next to this file as clip.wav for the STT part,
#      or delete the transcription section.
#
# What to expect:
#   - speech.wav is written next to this file (24 kHz WAV from the
#     installed Hima voice; model name vaani-tts-piper is an alias for
#     that engine, not Piper — see docs/API_V2.md).
#   - The transcript of clip.wav is printed. Telugu only; English words
#     inside Telugu are often missed (measured: 0 of 53 recalled on
#     16 Andhra clips, 20 Nov 2026).
#   - `speed` is accepted and currently ignored by the local engine.
#
# Only base_url changes vs OpenAI. Everything else is the same code.

import os

from openai import OpenAI

BASE_URL = os.environ.get("VAANI_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("TELUGU_API_KEYS", "choose-a-long-random-key").split(",")[0]

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# --- Text to speech ---
with client.audio.speech.with_streaming_response.create(
    model="vaani-tts-hima",
    voice="srinivas",  # or "sravani", "telugu_srinivas", "telugu_sravani"
    input="నమస్కారం",
    response_format="wav",  # wav always works; mp3/opus/flac need ffmpeg
) as response:
    response.stream_to_file("speech.wav")
print("Wrote speech.wav")

# --- Speech to text ---
with open("clip.wav", "rb") as audio:
    result = client.audio.transcriptions.create(
        model="vaani-asr-small",
        file=audio,
        language="te",
        response_format="json",
    )
print("Transcript:", result.text)
