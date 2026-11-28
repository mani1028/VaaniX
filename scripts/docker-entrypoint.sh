#!/bin/sh
# Vaani X container entrypoint.
#
# - Maps VAANI_ASR_MODEL to TELUGU_WHISPER_MODEL when set.
# - Links model folders found in the cache volume ($VAANI_CACHE_DIR) into
#   the paths the app expects. Weights are never baked into the image.
# - Refuses to start when a required model folder is absent, naming the
#   exact path. It never invents a model or a benchmark.
# - VAANI_OFFLINE=1: no download attempt at all; missing models are fatal.
# - VAANI_AUTO_FETCH=1: fetch the Hima voice pack with
#   scripts/fetch_lite_voice.py (Hugging Face repo
#   Prahaladha/hima-tts-nano-te-ONNX). The ASR model has no fetch script
#   yet: place it by hand (see docs/DEPLOYMENT_TIERS.md).
# - VAANI_TTS_PACK is NOT supported yet: the app reads the pack from
#   lite/models/hima-tts-nano-te only. Setting it is a fatal error, so a
#   misconfiguration fails loudly instead of silently using another pack.
set -eu

CACHE_DIR="${VAANI_CACHE_DIR:-/home/vaani/.cache/vaani}"
ASR_DIR="/app/models/telugu-asr-micro"
TTS_DIR="/app/lite/models/hima-tts-nano-te"

if [ -n "${VAANI_TTS_PACK:-}" ]; then
  echo "error: VAANI_TTS_PACK is not supported yet. Mount the voice pack at lite/models/hima-tts-nano-te." >&2
  exit 1
fi

if [ -n "${VAANI_ASR_MODEL:-}" ]; then
  export TELUGU_WHISPER_MODEL="$VAANI_ASR_MODEL"
fi

mkdir -p "$CACHE_DIR" /app/models /app/lite/models

# Cache volume wins: link cached models into the paths the app reads.
# The link can fail when /app/models is a read-only mount; that is fine,
# the missing-model check below then reports exactly what is absent.
if [ -d "$CACHE_DIR/telugu-asr-micro" ] && [ ! -e "$ASR_DIR" ]; then
  ln -s "$CACHE_DIR/telugu-asr-micro" "$ASR_DIR" 2>/dev/null || true
fi
if [ -d "$CACHE_DIR/hima-tts-nano-te" ] && [ ! -e "$TTS_DIR" ]; then
  ln -s "$CACHE_DIR/hima-tts-nano-te" "$TTS_DIR" 2>/dev/null || true
fi

missing=""
if [ ! -d "$ASR_DIR" ]; then
  missing="$missing\n  - ASR model folder: $ASR_DIR (or $CACHE_DIR/telugu-asr-micro)"
fi
if [ ! -f "$TTS_DIR/onnx/voices/step_int8.onnx" ]; then
  missing="$missing\n  - Voice pack marker: $TTS_DIR/onnx/voices/step_int8.onnx"
fi

if [ -n "$missing" ]; then
  if [ "${VAANI_AUTO_FETCH:-0}" = "1" ] && [ "${VAANI_OFFLINE:-0}" != "1" ] && [ ! -f "$TTS_DIR/onnx/voices/step_int8.onnx" ]; then
    echo "Voice pack missing; fetching Prahaladha/hima-tts-nano-te-ONNX ..."
    python scripts/fetch_lite_voice.py || true
  fi
  missing=""
  [ -d "$ASR_DIR" ] || missing="$missing\n  - ASR model folder: $ASR_DIR (or $CACHE_DIR/telugu-asr-micro)"
  [ -f "$TTS_DIR/onnx/voices/step_int8.onnx" ] || missing="$missing\n  - Voice pack marker: $TTS_DIR/onnx/voices/step_int8.onnx"
  if [ -n "$missing" ]; then
    printf 'error: required model files are missing, refusing to start:%b\n' "$missing" >&2
    printf 'Mount them (see docker-compose.yml) or set VAANI_AUTO_FETCH=1 for the voice pack.\n' >&2
    printf 'SHA-256 verification: run sha256sum over the mounted folders and compare with\n' >&2
    printf 'docs/DEPLOYMENT_TIERS.md. Hashes are TBD until the first verified download.\n' >&2
    exit 1
  fi
fi

if [ -z "${TELUGU_API_KEYS:-}" ] && [ "${TELUGU_SPEECH_AUTH:-required}" != "local_open" ]; then
  echo "warning: TELUGU_API_KEYS is empty; protected routes will return 503 until keys are set." >&2
fi

exec "$@"
