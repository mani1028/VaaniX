# Vaani X — Telugu voice API in 60 MB, runs on a $5-class VPS, OpenAI-compatible

> **The 60 MB number is the Piper-class voice size** (~62 MB for a Piper `te_IN-padmavathi-medium` voice), not a measured Docker image size — the Docker image size is still **TBD**. The shipped Hima voice pack is about 150 MB on disk (84 MB of model files). The **$5 VPS** is the Tiny VPS tier this project aims at (1–2 GB RAM, 2 vCPU, no GPU); no VPS price was measured — see `docs/DEPLOYMENT_TIERS.md`.

One Python process that your services call for Telugu speech. It transcribes a WAV file and speaks Telugu text. You host it. Callers share that one API.

Vaani X is the smallest, cheapest, genuinely self-hostable Telugu voice API that is OpenAI-compatible: one slim CPU container, two Telugu voices today, more voices later as USB-style voice packs — the core engine stays the same. Version 0.2.0.

The server code is MIT. Recognition weights are about 42 MB. The voice pack is about 150 MB on disk. Weights are not inside the PyPI package. Python 3.11 is the runtime that runs speech.

This is a low-memory CPU server. It is not a measured match for Sarvam. It does not translate, and it does not speak languages other than Telugu.

## Honest accuracy — read this first

The live decoder scored **WER 0.2899 and CER 0.0811 on 10 clean clips** (20 Nov 2026, tiny recognizer). That is about 71% of words exactly right. It is a usable prototype, **not 100% accurate**, and 10 clips are not a benchmark. The v1.0 targets are WER ≤ 20% and CER ≤ 5% on FLEURS-te + Kathbath-te, n ≥ 200 clips per set from ≥ 20 speakers — see `docs/benchmark_results.md`. No number in this repo claims Sarvam parity.

**Status, 21 Nov 2026: v2 is built but unverified.** The core registry, the OpenAI-compatible `/v1/*` routes, the bench harness, the slim Docker image, and 5 new test files are all written. No `pytest` or Docker run has passed yet — there is no shell on the author's Mac, so verification is Mani's next step. One blocker: the voice manifests ship with a zero `sha256`, so the registry rejects both packs until the real hash is filled in (see below). Until then the legacy fallback keeps the voices working.

## 60-second quickstart

```bash
# 1. Clone and install (Python 3.11)
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -e ".[host]"

# 2. Run it (keys are required unless you use local_open on loopback)
export TELUGU_SPEECH_AUTH=required
export TELUGU_API_KEYS='replace-me'
export TELUGU_HOST=127.0.0.1
export TELUGU_PORT=8000
export BHASHA_ENABLE_PLATFORM=0
telugu-api

# 3. Check it (no key needed)
curl -s http://127.0.0.1:8000/health
# {"status":"ok", ...}
```

Or with Docker (weights are mounted, never baked into the image):

```bash
docker build -t vaani-x:0.2.0-slim .
docker compose up --build -d   # keys from .env; see docker-compose.yml
curl -s http://127.0.0.1:8000/health
```

## Try it with the OpenAI SDK — only `base_url` changes

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="replace-me")

# Text to speech — wav always works; mp3 needs ffmpeg on the host.
with client.audio.speech.with_streaming_response.create(
    model="vaani-tts-hima", voice="srinivas", input="నమస్కారం", response_format="wav",
) as response:
    response.stream_to_file("speech.wav")

# Speech to text
with open("clip.wav", "rb") as audio:
    print(client.audio.transcriptions.create(
        model="vaani-asr-small", file=audio, language="te", response_format="json").text)
```

Working examples live in `examples/`: [python_openai_sdk.py](examples/python_openai_sdk.py), [python_native_api.py](examples/python_native_api.py), [javascript_openai_sdk.js](examples/javascript_openai_sdk.js), [curl.sh](examples/curl.sh), and [voiceflow_integration.py](examples/voiceflow_integration.py) (calling Vaani X from a voice-broadcast flow). Every example runs against the real endpoints on `localhost:8000` — no fake URLs, no mock responses.

## Why developers switch — paid APIs, November 2026 snapshot

Prices move. **Re-check every vendor page before publishing this table.** Snapshot from research on 21 Nov 2026:

| Provider | STT (Telugu) | TTS (Telugu) |
| --- | --- | --- |
| **Sarvam** | Saaras ₹0.49/min PAYG | Bulbul v4 ₹20 / 10,000 chars |
| **Google Cloud** | STT ~$0.016/min realtime, ~$0.003/min batch | TTS $4–30 / 1M chars by tier |
| **OpenAI** | whisper-1 $0.006/min | tts-1 $15 / 1M chars |
| **Vaani X (self-hosted)** | **$0/min after compute** | **$0/char after compute** |

The pitch is arithmetic, not adjectives: at 30 hours of STT + 1M chars of TTS a month, paid APIs cost roughly $11 + $4–80; a small VPS running Vaani X costs that VPS flat. Publish this table only with your own measured VPS cost, not a slogan. Full comparison with sources and caveats: `docs/` research notes and `docs/benchmark_results.md`.

## Benchmarks vs paid APIs — what we can and cannot say yet

| Metric | Vaani X today (measured) | v1.0 target |
| --- | --- | --- |
| ASR WER, 10 clean clips (20 Nov 2026) | 0.2899 (~71% words right) | ≤ 20% on FLEURS-te + Kathbath-te, n ≥ 200/set |
| ASR CER, same 10 clips | 0.0811 | ≤ 5% |
| ASR AER (akshara error) | not measured | ≤ 15% |
| TTS intelligibility (ASR back-transcription) | not measured | WER ≤ 15%, CER ≤ 6% |
| ASR / TTS RTF on 2 vCPU | not measured | ≤ 0.5 |

The bench harness (`python -m telugu_speech.bench`) records WER + CER + AER together, per-set and per-dialect, with RTF, peak RAM, and model revision SHA, to `bench/results/<model>/<date>.json`. Datasets are NOT in the repo — the command names the exact manifest path when one is missing. A "better than Sarvam" claim needs Sarvam scored with this normalizer on the same clips; that has not happened.

## Install

From PyPI, after the package is published:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install "telugu-speech[host]"
```

From a git checkout:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -e ".[host]"
```

`telugu-speech` installs the API. The `[host]` extra adds the recognizer and ONNX runtime. Place weights next to the process:

| Path | Role |
| --- | --- |
| `models/telugu-asr-micro` | Speech to text. Int8 Telugu Whisper tiny. |
| `lite/models/hima-tts-nano-te` | Text to speech. Srinivas and Sravani. |

If a folder is missing, that route returns 503.

## Install a voice pack (voices are USB connectors)

Voices are packs, not code. The core engine never changes when a voice is added: drop a pack in, register it, and it becomes speakable through the same `GET /api/v1/tts/voices` and `GET /v1/audio/voices` lists. Today the only shipped pack is Hima Nano int8 with Srinivas and Sravani — a folder alone, without a registry entry, does not appear yet.

```bash
# Fetch the shipped pack (Hugging Face: Prahaladha/hima-tts-nano-te-ONNX).
# Run only when lite/models/hima-tts-nano-te is missing.
.venv311/bin/python scripts/fetch_lite_voice.py
```

Check the pack's card (`lite/models/hima-tts-nano-te/README.md`) before redistributing it — its license is `other`. A future pack (Piper, IndicF5, …) is a new folder + a registry entry, not a new engine. Voice-pack SHA-256 hashes are TBD; see `docs/DEPLOYMENT_TIERS.md`.

**Blocker — fill the voice-pack hash before the registry will accept the packs:**

```bash
shasum -a 256 lite/models/hima-tts-nano-te/onnx/voices/step_int8.onnx
# Paste the hash into configs/voices/srinivas.yaml AND configs/voices/sravani.yaml,
# then copy them into place:
mkdir -p voices/telugu_srinivas voices/telugu_sravani
cp configs/voices/srinivas.yaml voices/telugu_srinivas/manifest.yaml
cp configs/voices/sravani.yaml  voices/telugu_sravani/manifest.yaml
```

## Run one API

```bash
export TELUGU_SPEECH_AUTH=required
export TELUGU_API_KEYS='replace-me'
export TELUGU_HOST=127.0.0.1
export TELUGU_PORT=8000
export BHASHA_ENABLE_PLATFORM=0
telugu-api
```

| URL | Use |
| --- | --- |
| `GET /api/v1/health` | No key. Returns `ok`. |
| `GET /health` | No key. Returns `ok` + voices, models, slots. |
| `GET /docs` | Request and response shapes. |
| `http://127.0.0.1:8000/` | Project page and voice studio. The file is `index.html` in the repository root. |

Every service uses the same base URL and a bearer key. Several keys can be comma-separated in `TELUGU_API_KEYS`. Restart after you change them. There is no database and no per-service usage record.

`local_open` skips the key. Use it only on loopback while developing. Do not publish that mode.

Put TLS and a rate limit on a proxy. This process does not terminate TLS and does not quota callers. Set `TELUGU_CORS_ORIGINS` to the exact browser origins that call it. A `*` entry is ignored. Copy names from `.env.example`. Do not commit real keys.

## Call the API

Speech to text. WAV, Telugu only. Upload cap is 20 MB (hard max 50 MB).

```bash
curl -s -H "Authorization: Bearer $TELUGU_API_KEYS" \
  -F "audio=@clip.wav;type=audio/wav" \
  http://127.0.0.1:8000/api/v1/asr/transcribe
```

Text to speech. Default voice is `telugu_srinivas`. Text max is 2000 characters. The JSON includes `audio_url`. Download that URL with the same key. The process keeps the newest 32 WAV files.

```bash
curl -s -H "Authorization: Bearer $TELUGU_API_KEYS" \
  -H "Content-Type: application/json" \
  -d '{"text":"నమస్కారం","language":"te","voice":"telugu_sravani","format":"wav"}' \
  http://127.0.0.1:8000/api/v1/tts/generate
```

`POST /api/v1/tts/stream` sends 16-bit PCM as each sentence finishes. The sample rate is the `X-Sample-Rate` header.

`style` accepts only `conversational`.

## OpenAI-compatible API (v2)

Drop-in routes for OpenAI SDKs — change only `base_url`. Landed in v2 (Part 2); **verification on the Mac is still pending**, so treat them as implemented-but-unverified until `.venv311/bin/python -m pytest` and the smoke curls below pass.

| Method | Use |
| --- | --- |
| `POST /v1/audio/speech` | TTS. JSON `{model, input, voice, response_format, speed}` → audio bytes. |
| `POST /v1/audio/transcriptions` | STT. Multipart `file` + `model` + `language="te"` → `{text}`. |
| `GET /v1/audio/voices` | Installed voices (Srinivas, Sravani). |
| `GET /v1/models` | OpenAI-shaped model list. |
| `GET /health` | No key. Returns `ok`. |

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="replace-me")
client.audio.speech.create(model="vaani-tts-piper", voice="telugu_srinivas", input="నమస్కారం")
```

Honest notes, do not remove them: `vaani-tts-piper` is an alias for the installed Hima ONNX engine, **not** Piper — the Piper voice is a future pack. `speed` and ASR `temperature` are accepted and ignored for now. `mp3`/`opus`/`flac` output needs ffmpeg. Errors are OpenAI-shaped (`{"error": {"message", "type", "code"}}`). Auth accepts `Authorization: Bearer <key>` or `X-API-Key`. Rate limit: 600 requests/min per key on `/v1/audio/*`. The old `/api/v1/*` routes are unchanged. Full spec: `docs/API_V2.md`.

## Deployment tiers

Pi 4 (TTS only) · Tiny VPS 1–2 GB / 2 vCPU, slots 1–2 · Small VPS 4 GB, slots 2–4 · GPU opt-in (no `:gpu` image yet). Full table, Docker commands, cache/SHA procedure, and offline mode: `docs/DEPLOYMENT_TIERS.md`.

```bash
docker build -t vaani-x:0.2.0-slim .
docker compose up --build -d   # keys from .env; see docker-compose.yml
```

| Voice id | What you hear |
| --- | --- |
| `telugu_srinivas` | Deep male, about 100–115 Hz. Default. |
| `telugu_sravani` | Female, about 185–225 Hz. |

Same process, no HTTP server, after `pip install -e ".[host]"` from the project folder:

```python
from pathlib import Path
from telugu_speech.asr.infer import FasterWhisperBackend
from telugu_speech.tts.voices import backend_for_voice

voice = backend_for_voice("telugu_srinivas")
voice.synthesize("నమస్కారం", None, "", Path("speech.wav"))

recognizer = FasterWhisperBackend("models/telugu-asr-micro")
print(recognizer.transcribe(Path("clip.wav")))
```

## Memory and concurrency

One process. Idle speech keeps **one** engine. Extra engines are created only while calls overlap, then dropped. The legacy ceiling is `TELUGU_SPEECH_SLOTS` (default **8**, hard cap **16**); the new core engine pool defaults to **2** — 1 engine is 260–470 MB, 2 engines 640–720 MB, so 8 would need roughly 1.5–1.8 GB. HTTP accepts up to **256** calls (`TELUGU_LIMIT_CONCURRENCY`). Calls past the speech ceiling wait. They do not each load a new model.

Measured 20 Nov 2026 on an Apple M5, 16 GB RAM, CPU, 2 threads, short Telugu lines:

| Load | What is in memory | RAM |
| --- | --- | --- |
| 1 speech request | 1 voice engine | **260–470 MB** |
| 2 speech requests overlapping | 2 voice engines | **640–720 MB** |
| 8 speech requests (default ceiling) | 8 engines | **not measured.** About **160 MB** more per extra engine, so roughly **1.5–1.8 GB** |
| 16 speech requests (hard cap) | 16 engines | **not measured.** Roughly **2.7–3.0 GB** for speech |
| Recognition added | 1 shared recognizer, not one per call | about **350–540 MB** on top of speech |

After the burst, RAM returns to the one-engine figure. A 2 GB machine should set `TELUGU_SPEECH_SLOTS=2` or `1`.

Throughput on that machine was about **6–8 short spoken lines per second** with 2 engines. A longer clip holds one slot for the whole clip. Recognition on the tiny model is about **5 times realtime** and stays one shared model.

## What you configure

| Variable | Default | Meaning |
| --- | --- | --- |
| `TELUGU_SPEECH_AUTH` | `required` | `local_open` skips the key on loopback only. |
| `TELUGU_API_KEYS` | empty | Comma-separated bearer keys. |
| `TELUGU_HOST` | `127.0.0.1` | Interface the proxy uses. |
| `TELUGU_PORT` | `8000` | Port. |
| `TELUGU_SPEECH_SLOTS` | `8` (legacy) / `2` (core pool) | Max overlapping speech engines. Cap 16 legacy. |
| `TELUGU_CPU_THREADS` | `2` | Threads per engine. Cap 4. |
| `TELUGU_LIMIT_CONCURRENCY` | `256` | HTTP calls in flight. |
| `TELUGU_MAX_UPLOAD_BYTES` | `20000000` | Upload cap. Hard max 50 MB. |
| `TELUGU_CORS_ORIGINS` | empty | Exact browser origins. |
| `BHASHA_ENABLE_PLATFORM` | `1` in code | Set `0` for this speech API only. |

## Licences — per model, not one blanket claim

| Part | Licence | Note |
| --- | --- | --- |
| Server code (`src/`, API, core) | MIT | `LICENSE` |
| Recognizer weights (`models/telugu-asr-micro`) | Apache-2.0 | |
| Hima voice pack (`lite/models/hima-tts-nano-te`) | `other` | Read its card before redistributing |
| Older MMS pack (`models/telugu-tts-micro`, if present) | CC-BY-NC-4.0 | Not a hosted voice |
| Future packs (Piper, IndicF5, …) | Per voice card | Each voice carries its own licence — check it before shipping |

Do not ship XTTS v2 as a voice (CPML, non-commercial, no licensor left since Coqui shut down) or MMS (CC-BY-NC) as a commercial-facing voice.

## Repository map

| Path | Keep for |
| --- | --- |
| `src/telugu_speech` | The API and engines. This is the PyPI package. |
| `src/telugu_speech/core` | Voice registry, engine pool, Telugu text pipeline. |
| `examples/` | Working SDK / curl / VoiceFlow examples. |
| `index.html` | The page served at `/`. |
| `tests` | `pytest`. |
| `docs/` | `API_V2.md`, `ARCHITECTURE.md`, `VOICE_PACK.md`, `DEPLOYMENT_TIERS.md`, `DEVELOPER_GUIDE.md`, `LAUNCH.md`, `CHANGELOG.md`, `benchmark_results.md`. |
| `docs/deployment.md` | Host, TLS, rollback. |
| `src/telugu_speech/bench.py` | `python -m telugu_speech.bench` — frozen-set scoring, JSON to `bench/results/<model>/<date>.json`. |
| `Dockerfile`, `docker-compose.yml` | Slim CPU image (no torch/CUDA, no weights baked in). |
| `lite/models/hima-tts-nano-te` | Voice weights. License on that folder's card is `other`. |
| `platform/` | Optional catalog. Not required to host this API. |

`data/`, virtualenvs, checkpoints, and generated WAV files stay off git. Training scripts under `scripts/` and notes under `docs/` are for this lab, not for callers of the API.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — how to add a voice pack (folder + manifest + sha256 + test), add a backend, run the bench, and the PR gate (a WER regression over 2 points absolute on FLEURS-te fails). [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and [SECURITY.md](SECURITY.md) apply to every contributor.

## Publish

Build a wheel from a clean tree. The wheel is code. Ship weights beside the deployed process, not inside the wheel.

```bash
pip install build
python -m build
```

Upload `dist/*` with your PyPI account when you are ready. This repository does not publish for you.

## Checks

```bash
.venv311/bin/python -m pytest -q
```

The tests need no model weights and no `[bench]` extra. Install it only for the jiwer cross-check: `pip install -e ".[bench]"`.

Benchmark harness (datasets are NOT in the repo — prepare them first; the command names the exact manifest path when one is missing):

```bash
.venv311/bin/python -m telugu_speech.bench --asr-model telugu-asr-micro --sets fleurs-te kathbath-te cv-te --n 200
.venv311/bin/python -m telugu_speech.bench --asr-model telugu-asr-micro --sets fleurs-te --n 10 --allow-small  # smoke test only
```

Docker smoke test after building:

```bash
docker build -t vaani-x:0.2.0-slim .
# Without the mounts below the container exits 1: weights are not in the image.
docker run --rm -p 8000:8000 -e TELUGU_API_KEYS='test-key' \
  -v vaani-cache:/home/vaani/.cache/vaani \
  -v "$PWD/models:/app/models:ro" -v "$PWD/lite:/app/lite:ro" \
  vaani-x:0.2.0-slim
curl http://127.0.0.1:8000/api/v1/health
curl -H "Authorization: Bearer test-key" http://127.0.0.1:8000/v1/models
```

The new tests in this part need no model weights to collect; tests that need a real voice pack or the `/v1` router skip with a reason when it is absent. Nothing here claims the tests pass — run them on the Mac and report failures.

## License

Server code is MIT (`LICENSE`). Recognizer weights are Apache-2.0. The Hima voice card license is `other`; read `lite/models/hima-tts-nano-te/README.md` before you redistribute those weights. An older MMS pack, if present under `models/telugu-tts-micro`, is CC-BY-NC-4.0 and is not a hosted voice.

## Limits of this product

- Telugu (`te`) only.
- No translation.
- No measured Sarvam parity. Ten clean clips on the tiny recognizer scored WER 0.2899 and CER 0.0811 with the current decoder (20 Nov 2026). English words inside Telugu are often missed — on 16 Andhra clips, English-word recall was 0 of 53.
- The websocket route answers a whole chunk and sets `streaming` to false.
- Voice naturalness against other vendors was not measured.
- `speed` and ASR `temperature` on `/v1/*` are accepted and currently ignored; `mp3`/`opus`/`flac` need ffmpeg; the `:gpu` image does not exist yet; the CI bench gate is documented, not yet wired.