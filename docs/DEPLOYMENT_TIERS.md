# Deployment tiers — Vaani X 0.2.0

One image, one process, no database. The only thing that changes between
tiers is how much RAM you give it and how many speech engines may overlap
(`TELUGU_SPEECH_SLOTS`). A call past the slot ceiling waits inside the
process; it never loads another model copy.

## Measured on this project (not a target)

20 Nov 2026, Apple M5, 16 GB RAM, CPU, 2 threads, short Telugu lines:

- 1 speech engine: **260–470 MB**
- 2 engines overlapping: **640–720 MB**
- Recognizer added (shared, one copy): **+350–540 MB** on top of speech
- ASR (tiny int8): about **5x realtime**
- TTS: about **6–8 short lines/sec** with 2 engines

Everything else in the table below is a **TBD** until it is measured on
that hardware with `scripts/bench.py` and the suite in
docs/benchmark_results.md. Do not quote a tier number that is still TBD.

## Tiers

| Tier | Hardware | What runs | Slots | Expected behaviour |
| --- | --- | --- | ---: | --- |
| **Pi 4 (edge)** | Raspberry Pi 4, 2–4 GB RAM, ARM | Hima int8 voice only; ASR only if the tiny model is validated on ARM (Piper is a future pack, not the shipped engine) | `TELUGU_SPEECH_SLOTS=1` | TBD — Pi 4 RTF/RAM not measured yet |
| **Tiny VPS (the #1 claim)** | 1–2 GB RAM, 2 vCPU, no GPU | 1 ASR model + 1 voice pack, lazy-loaded | `TELUGU_SPEECH_SLOTS=1` (1 GB) or `2` (2 GB) | 1 engine 260–470 MB (M5 figure; VPS figure TBD). Load ONE ASR model at a time. |
| **Small VPS** | 4 GB RAM, 2–4 vCPU | ASR + voice pack, room for a second voice | `TELUGU_SPEECH_SLOTS=2`–`4` | 2 engines 640–720 MB + recognizer 350–540 MB (M5 figures; VPS figure TBD) |
| **GPU (opt-in)** | 4–6 GB VRAM | Quality-tier voices/models only when a `:gpu` image exists | `TELUGU_SPEECH_SLOTS=4` | TBD — no `:gpu` image is built by this repo yet |

Default in `docker-compose.yml` is the Tiny VPS shape: slots 2, 2 CPU
threads. A 1 GB machine must set `TELUGU_SPEECH_SLOTS=1`.

## Docker (slim CPU image)

```bash
docker build -t vaani-x:0.2.0-slim .

# Host weights, read-only mounts (the normal path):
docker run --rm -p 8000:8000 \
  -e TELUGU_API_KEYS='choose-a-long-random-key' \
  -e TELUGU_SPEECH_SLOTS=2 \
  -v vaani-cache:/home/vaani/.cache/vaani \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/lite:/app/lite:ro" \
  vaani-x:0.2.0-slim

# Or with Compose (keys come from .env; see docker-compose.yml):
docker compose up --build -d
docker compose ps
curl http://127.0.0.1:8000/api/v1/health
```

The image has no torch and no CUDA. Model weights are never baked in
(`.dockerignore` excludes `models/`, `lite/models/`, `data/`). Image size
is not claimed here: measure it after building with
`docker images vaani-x` and record the number here.

## Model cache, SHA verification, offline mode

- Cache volume: `/home/vaani/.cache/vaani` (Docker volume `vaani-cache`).
  The entrypoint links `telugu-asr-micro` and `hima-tts-nano-te` from the
  cache into the paths the app reads. Models survive container rebuilds.
- Overrides: `VAANI_ASR_MODEL` sets the ASR folder (`TELUGU_WHISPER_MODEL`).
  `VAANI_TTS_PACK` is **not supported yet** — the app only reads
  `lite/models/hima-tts-nano-te`; setting it fails startup loudly.
- SHA-256: after placing weights, record and verify them:

```bash
sha256sum models/telugu-asr-micro/* lite/models/hima-tts-nano-te/onnx/voices/step_int8.onnx
```

  Expected hashes: **TBD** — fill them in after the first verified
download and pin the download revision. `scripts/fetch_lite_voice.py`
  downloads the voice pack from `Prahaladha/hima-tts-nano-te-ONNX` without
a pinned revision today; that is a known gap, not a guarantee.
- Offline: `VAANI_OFFLINE=1` disables the voice-pack fetch attempt.
  Missing models then fail startup with the exact paths expected.
- Auto-fetch: `VAANI_AUTO_FETCH=1` runs `scripts/fetch_lite_voice.py`
  once when the voice pack is missing. There is no auto-fetch for the
  ASR model yet; place `models/telugu-asr-micro` by hand.

## Rollback

Stop the container and start the previous image tag. There is no
migration and no database. Removing the cache volume deletes downloaded
models only; host-mounted weights are untouched.

## What not to claim

- No tier above is a measured Sarvam replacement. The ten-clip
  recognizer result (WER 0.2899, CER 0.0811, live decoder, 20 Nov 2026)
  is in docs/benchmark_results.md; it is not a benchmark.
- No Docker image size, Pi 4 RTF, or VPS RTF is claimed until measured.
- The `:gpu` tag does not exist yet. Do not advertise it.
