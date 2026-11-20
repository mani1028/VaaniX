# Architecture

Telugu ASR and Telugu TTS share text normalization, manifests, and the HTTP process. They do not share weights.

## Trust boundary

- Callers are anonymous until `TELUGU_SPEECH_AUTH=required` and a bearer token in `TELUGU_API_KEYS` match.
- `TELUGU_SPEECH_AUTH=local_open` disables that check for a private workstation. Do not expose that mode on a network.
- Upload names are reduced to a basename and must end in `.wav`. Bytes are written under a temporary directory and deleted after transcription.
- The API never accepts a filesystem path from the client.
- Health and model metadata are public and do not include audio or transcripts.

There is no tenant model. Recordings in manifests are local research files, not customer tenants.

## ASR path

1. Read mono PCM WAV.
2. Resample to 16 kHz and peak-normalize.
3. Energy VAD marks speech for inspection. Silero is optional and is not downloaded by tests.
4. IndicConformer (CTC or RNN-T) or faster-whisper produces text.
5. Raw and normalized transcripts are both returned. Latin words stay in Latin.

The student model TeluguTinyASR is specified in `configs/asr/compact.yaml` and is not trained in this phase.

## TTS path

IndicF5 is the quality reference and needs reference audio plus its transcript from a speaker who consented. Compact FastPitch or Matcha training is configured and refused until that corpus exists.

## Failure behavior

- Missing weights: HTTP 503 `model_unavailable` or a benchmark line `Status: unavailable`.
- Missing API keys in required mode: HTTP 503 `api_keys_not_configured`.
- Bad token: HTTP 401.
- Oversized or non-WAV upload: HTTP 413 or 400.
- Training entry points raise `TrainingDeferred` after the config is validated.
