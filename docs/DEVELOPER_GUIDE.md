# Developer Guide — Vaani X 0.2.0

This is the guide for developers who build *on* Vaani X (call the API) and developers who build *in* it (change the core). The README is the 60-second version; this is the working version.

## 1. Calling Vaani X

### Which surface?

| Surface | Use it when | Notes |
| --- | --- | --- |
| `/v1/*` (OpenAI-compatible) | You already use the OpenAI SDK, or you want the shortest migration | Change only `base_url` to `http://localhost:8000/v1`. Full spec: `docs/API_V2.md`. |
| `/api/v1/*` (native) | You want the original API, or the `audio_url` flow | Unchanged since v1. TTS returns JSON with `audio_url`; download it with the same key. |
| Direct import (no server) | You are embedding the engine in one Python process | `backend_for_voice("telugu_srinivas")`, `FasterWhisperBackend("models/telugu-asr-micro")` — see README. |

### Auth

One key scheme for everything: `Authorization: Bearer <key>` or `X-API-Key: <key>` (on `/v1/*`), from `TELUGU_API_KEYS` (comma-separated). `GET /health` and `GET /api/v1/health` need no key. `local_open` skips the key on loopback only, for development — do not publish that mode. TLS and quotas belong on a proxy; this process does neither.

### Limits you must design around

- Telugu (`te`) only. `language` other than `te` → 400.
- TTS input: 1–4096 chars on `/v1/audio/speech` (400 beyond), 1–2000 chars on the native route. The engine chunks long text into sentences server-side; chunk cap is 180 chars per synthesis piece (Hima limit).
- ASR upload: 25 MB on `/v1/*` (413 beyond), 20 MB default on the native route (hard max 50 MB).
- Rate limit: 60 req/min per key on `/v1/audio/*` and `/v1/models` (429 + `Retry-After`). `/health` and `/v1/audio/voices` are exempt. The limiter is in-memory and does not survive a restart.
- `speed` and ASR `temperature` are accepted and currently ignored. `mp3`/`opus`/`flac` need ffmpeg on the host (400 otherwise). WAV always works, 24 kHz.
- `vaani-tts-piper` is an alias for the installed Hima ONNX engine, **not** Piper. Piper is a future voice pack.
- ASR is VAD-segmented batch decode. There is no streaming transcription endpoint and none is faked. The `/api/v1/stt/stream` websocket answers a whole chunk with `streaming: false`.
- Errors on `/v1/*` are OpenAI-shaped: `{"error": {"message", "type", "code"}}`. `/api/v1/*` keeps its old shape.

### Working examples

`examples/` — every file states its prerequisites and what to expect, and runs against the real endpoints on `localhost:8000`:

- `python_openai_sdk.py` — OpenAI SDK, TTS + STT.
- `python_native_api.py` — httpx against `/api/v1/*`, including the `audio_url` download.
- `javascript_openai_sdk.js` — OpenAI SDK in Node.
- `curl.sh` — health, models, voices, both TTS and STT routes, error-shape spot check.
- `voiceflow_integration.py` — broadcast flow: per-recipient Telugu audio via `/v1/audio/speech`, then a clearly-marked hook to wire Plivo/Exotel. It places no calls.

## 2. How the core works

Full architecture: `docs/ARCHITECTURE.md`. The one-paragraph version:

- **Voices are USB connectors.** A voice is a folder with a `manifest.yaml`. The registry (`src/telugu_speech/core/registry.py`) discovers `VAANI_VOICES_DIR` → `voices/` → `lite/models/`, validates fields + file existence + SHA-256, lazy-loads, and `reload()`s without a restart. A bad pack is recorded in `registry.rejected` and the server keeps running.
- **One engine per loaded voice, one shared ASR.** The engine pool defaults to 2 slots (the legacy runtime default of 8 was dangerous on a 1–2 GB VPS: 1 engine 260–470 MB, 2 engines 640–720 MB). Threads are pinned (`TELUGU_CPU_THREADS`, default 2, cap 4). Idle voices unload after `VAANI_VOICE_IDLE_SECS` (default 1800 s).
- **Telugu text lives in the core.** NFC normalisation, sentence splitting, number/date/currency expansion (lakh/crore, ordinals, decimals), chunking. Code-mixed English is kept in Latin and passed through — dropping user-typed words is data loss.
- **Dependency direction is one-way:** API → core → providers (lazy, by engine name). Core never imports a voice engine or FastAPI at startup.

## 3. Adding things

- **A voice:** folder + manifest + sha256 + test. See CONTRIBUTING.md and `docs/VOICE_PACK.md`. No code.
- **A backend:** one provider module + one line in `core/engine.py::_BACKENDS`. See CONTRIBUTING.md and `docs/ARCHITECTURE.md` §5. A pack must never contain code the core executes.
- **A model tier:** the recommended ladder is Piper-class (~62 MB) as the small default, IndicF5 (0.35B, 1.4 GB) as the GPU quality tier, faster-whisper INT8 `vasista22/whisper-telugu-small` (244M) as the ASR default, IndicConformer-600M INT8 (659 MB) as the accuracy tier. Do not ship XTTS v2 (no Telugu, CPML), MMS (CC-BY-NC), Vosk-te (87.9% WER on FLEURS), or vanilla Whisper for Telugu (it collapses on Telugu script). Each new model needs its licence, RAM, and RTF measured or labelled TBD.

## 4. Measuring things

- Bench: `python -m telugu_speech.bench --asr-model telugu-asr-micro --sets fleurs-te kathbath-te cv-te --n 200`. Frozen sets, speaker-disjoint, n ≥ 200/set from ≥ 20 speakers. Report WER + CER + AER together. Results to `bench/results/<model>/<date>.json`, never hand-edited.
- PR gate: a WER regression over 2 points absolute on FLEURS-te fails the PR. CI (`.github/workflows/ci.yml`) is a file, not live CI, until it is enabled in GitHub — until then, paste bench results into the PR by hand.
- Current measured position: live decoder WER 0.2899 / CER 0.0811 on 10 clean clips (20 Nov 2026). Targets: WER ≤ 20%, CER ≤ 5%, AER ≤ 15%, RTF ≤ 0.5. All targets are TBD until a frozen-set run measures them.

## 5. Debugging checklist

1. `curl http://127.0.0.1:8000/health` — if this fails, the server is not up or not on port 8000.
2. Voice pack on disk? `models/telugu-asr-micro` and `lite/models/hima-tts-nano-te` must exist; a missing folder → 503 naming the path.
3. Voice-pack `sha256` filled in? A zero hash means the registry rejects the pack (legacy fallback keeps the voices working — check `GET /v1/audio/voices` to see which path served you).
4. Key right? Missing/wrong key → 401, OpenAI shape on `/v1/*`.
5. Text has Telugu letters? TTS input with no Telugu letters → 400 `invalid_text`.
6. Format needs ffmpeg? `mp3`/`opus`/`flac` without ffmpeg → 400 `unsupported_format`. Use `wav`.
7. Upload too big? Over 25 MB on `/v1/*` → 413.
8. Rate-limited? Over 60 req/min per key → 429 with `Retry-After`.
