# Changelog — Vaani X (telugu-speech)

All dates 2026. Unverified items are marked as such — nothing here claims a test run that did not happen.

## 2.1 — Developer experience (21 Nov 2026, Part C)

- README rewritten: 60-second quickstart at the top, OpenAI SDK snippet, paid-API price table (November 2026 snapshot — re-check before publishing), benchmark table, licence box per model, limitations section. Honest accuracy box kept; all TBD labels kept.
- `examples/` (new): `python_openai_sdk.py`, `python_native_api.py`, `javascript_openai_sdk.js`, `curl.sh`, `voiceflow_integration.py` — all against the real `/v1/*` and `/api/v1/*` endpoints on `localhost:8000`, no fake URLs, no mock responses.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` (new).
- `.github/` (new): bug / voice-request / feature-request issue templates, PR template (with the WER-gate checklist), `workflows/ci.yml` — a file, not live CI, until enabled in GitHub.
- `docs/DEVELOPER_GUIDE.md`, `docs/LAUNCH.md` (new): calling guide, debugging checklist, and the launch gates (pytest green, bench n≥200, Docker size measured, sha256 filled, prices re-checked).
- `index.html`: headline, accuracy box, and quickstart copy updated; existing JS functions untouched.
- **Unverified:** nothing in 2.1 was run — no pytest, no example execution, no Docker run. Mani verifies on the Mac.

## 0.2.0 — Core Engine v2 (21 Nov 2026, Parts 1–3)

### Part 1 — Core + USB-style voice plugins

- New `src/telugu_speech/core/`: registry (voice = folder + `manifest.yaml`, SHA-256 validated, lazy-load, `reload()` without restart), engine pool (one engine per voice, one shared ASR, slots default 2), Telugu-first text pipeline, errors, config.
- New `src/telugu_speech/tts/providers/`: `base`, `hima_nano`, `piper`, `onnx_vits` behind one engine protocol.
- `tts/voices.py` refactored, `backend_for_voice()` signature preserved exactly.
- `configs/voices/srinivas.yaml` + `sravani.yaml`, `docs/ARCHITECTURE.md`, `docs/VOICE_PACK.md`.
- Real Hima facts corrected: 62M params, ONNX int8, 24 kHz, 15.0× real-time at 4 threads, 84 MB model files; chunks cap 180 chars, text cap 2000.
- **Blocker:** both manifests ship with a zero `sha256` — the registry rejects both packs until Mani fills the real hash (legacy fallback keeps voices working).

### Part 2 — OpenAI-compatible API

- New `src/telugu_speech/api/routes_openai.py`: `POST /v1/audio/speech`, `POST /v1/audio/transcriptions`, `GET /v1/models`, `GET /v1/audio/voices`, `GET /health`. Bearer or `X-API-Key`, 60 req/min per key on `/v1/audio/*`, OpenAI-shaped errors.
- `docs/API_V2.md` with Python + JS SDK examples (only `base_url` changes).
- Old `/api/v1/*` routes untouched.
- Caveats: `vaani-tts-piper` is an alias for the installed Hima engine, not Piper; `speed` and ASR `temperature` are accepted no-ops; `mp3`/`opus`/`flac` need ffmpeg.

### Part 3 — Benchmarks, Docker, tests

- Bench harness (`src/telugu_speech/bench.py`, `scripts/bench.py`): WER + CER + AER, per-set/per-dialect slices, RTF + peak RAM, model SHA, JSON to `bench/results/<model>/<date>.json`. Requires n ≥ 200, ≥ 20 speakers, speaker-disjoint.
- 5 new test files (registry, OpenAI compat, Telugu text, engine pool, bench harness) — no weights needed.
- Slim Dockerfile (`python:3.11-slim`, no torch/CUDA, no weights baked in, non-root user), `docker-compose.yml` Tiny VPS defaults (slots 2), `docs/DEPLOYMENT_TIERS.md`.
- **TBD, not results:** Docker image size, Pi/VPS RTF/RAM, v1.0 targets (WER ≤ 20%, CER ≤ 5%, AER ≤ 15%, RTF ≤ 0.5), voice-pack SHA hashes, `:gpu` image, CI gate script, bench datasets.

### Verification status (all of 0.2.0)

Unverified. No shell was available on the Mac for any part. Mani must run: `py_compile` + `.venv311/bin/python -m pytest -q` + smoke curls for old and new routes + `docker build`.

## 0.1.0 — Telugu Speech API (20 Nov 2026)

- One Python process, FastAPI, Python 3.11. Telugu-only TTS + STT.
- ASR: FasterWhisper tiny INT8 (`models/telugu-asr-micro`, ~42 MB). TTS: Hima Nano int8 pack (`lite/models/hima-tts-nano-te`, ~150 MB), voices Srinivas (male) and Sravani (female), 24 kHz.
- Routes: `POST /api/v1/tts/generate`, `POST /api/v1/asr/transcribe`, `POST /api/v1/tts/stream`, `GET /api/v1/health`.
- Measured 20 Nov 2026: live decoder WER 0.2899 / CER 0.0811 on 10 clean clips (~71% of words right); tiny int8 on 10 FLEURS clips WER 0.4379 / CER 0.2760 / RTF 0.1095; 1 engine 260–470 MB, 2 engines 640–720 MB, recognizer +350–540 MB; ASR ~5× realtime; TTS 6–8 short lines/sec with 2 engines (Apple M5, 16 GB).
- Known gaps: English words inside Telugu are often missed (English recall 0 of 53 on 16 Andhra clips); websocket answers a whole chunk with `streaming: false`; no measured Sarvam parity; voice naturalness not measured against other vendors.
