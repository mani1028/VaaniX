#!/bin/sh
# Vaani X — curl examples
#
# Prerequisites: the server is running on localhost:8000 (see README):
#   export TELUGU_API_KEYS='choose-a-long-random-key'
#   telugu-api
# Run:  sh examples/curl.sh
# Put a real WAV next to this file as clip.wav for the STT parts.
#
# What to expect: health returns ok without a key; every other call
# needs Authorization: Bearer <key> (or X-API-Key: <key> on /v1/*).
# /v1/* errors are OpenAI-shaped; /api/v1/* keeps its old shape.

set -e
BASE="${VAANI_BASE_URL:-http://127.0.0.1:8000}"
KEY="${TELUGU_API_KEYS:-choose-a-long-random-key}"

# --- Health (no key) ---
curl -s "$BASE/health"
echo
curl -s "$BASE/api/v1/health"
echo

# --- Models and voices (OpenAI-compatible surface) ---
curl -s -H "Authorization: Bearer $KEY" "$BASE/v1/models"
echo
curl -s -H "Authorization: Bearer $KEY" "$BASE/v1/audio/voices"
echo

# --- TTS, OpenAI-compatible: audio bytes straight back ---
curl -s -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"vaani-tts-hima","voice":"srinivas","input":"నమస్కారం","response_format":"wav"}' \
  "$BASE/v1/audio/speech" -o speech.wav
file speech.wav

# --- TTS, native route: JSON with audio_url, then download it ---
curl -s -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"text":"నమస్కారం","language":"te","voice":"telugu_sravani","format":"wav"}' \
  "$BASE/api/v1/tts/generate"
echo

# --- STT, OpenAI-compatible ---
curl -s -H "Authorization: Bearer $KEY" \
  -F "file=@clip.wav;type=audio/wav" -F "model=vaani-asr-small" -F "language=te" \
  -F "response_format=verbose_json" "$BASE/v1/audio/transcriptions"
echo

# --- STT, native route ---
curl -s -H "Authorization: Bearer $KEY" \
  -F "audio=@clip.wav;type=audio/wav" "$BASE/api/v1/asr/transcribe"
echo

# --- Error shape spot check: no key -> 401, OpenAI shape on /v1/* ---
curl -s -o /dev/null -w "%{http_code}\n" "$BASE/v1/models"
