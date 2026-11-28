# Vaani X — lightweight audit (22 Nov 2026)

Work of the LIGHTWEIGHT worker. Scope: dependencies, lazy loading, INT8
on the default path, Docker. The ACCURACY worker owns ASR decoding, VAD,
normalisation, and model defaults — nothing in this file changes a
decoding parameter, a beam/VAD setting, a model default, or
`docs/ACCURACY.md`. No route, auth, or API shape changed. No mock data.

Baseline this builds on: `docs/BASELINE.md` (measured 22 Nov 2026).
Numbers copied from there are marked with their source; anything not
yet measured by Mani is marked **UNMEASURED** with the exact command.

## Files changed

| File | Change |
| --- | --- |
| `Dockerfile` | Added `COPY configs ./configs` + header comment (see §4). No base-image, install, or entrypoint change. |
| `pyproject.toml` | Comment only: torch / transformers / datasets / huggingface_hub / soundfile are deliberately undeclared (see §1). No dependency added or removed. |
| `tests/test_lightweight.py` | NEW: 5 tests, real constructors, no mocks, no weights — lazy-load + INT8-default pins (see §2–§3). |
| `docs/LIGHTWEIGHT.md` | This file (NEW). |

`docker-compose.yml` unchanged (already mounts models + lite read-only
and a cache volume — see §4).

## 1. Dependencies — before / after

Audit method: read `pyproject.toml` plus the import headers of every
module in the server import closure (`api/main.py` → routes →
`core/*` → `asr/infer.py`, `tts/infer.py`, `tts/voices.py`,
`tts/runtime.py`, `tts/chunks.py`, `text/*`, `audio/*`,
`detection/lid.py`, `tts/providers/*`), the bench path
(`bench.py`, `evaluation/*`), the `platform/` attach, the `scripts/`
and the `tests/` — no grep is available on this Mac, so each file was
read directly. Result: **nothing was dropped. Every declared
dependency is imported by code that runs; there is no torch-sized
dependency to remove, because torch was never declared.**

### Declared dependencies — all kept, with evidence

| Dependency (pin) | Imported by (first use found) | Verdict |
| --- | --- | --- |
| `fastapi>=0.115,<1` (core) | `api/main.py`, all `api/routes_*.py`, `platform/.../api.py` | KEEP |
| `numpy>=1.26,<3` (core) | `api/routes_openai.py`, `audio/*`, `tts/providers/hima_nano.py`, Hima pack `hima_onnx` | KEEP |
| `pydantic>=2.6,<3` (core) | `api/schemas.py`, `platform/.../api.py` | KEEP |
| `python-multipart>=0.0.9,<1` (core) | FastAPI `UploadFile`/`File(...)` in `routes_openai.py`, `routes_stt.py`, `routes_asr.py`, `routes_detect.py`, `routes_tts.py` | KEEP |
| `pyyaml>=6.0,<7` (core) | `core/config.py` (`_load_yaml_file`, lazy import), `core/registry.py` (lazy, with a dependency-free fallback parser), `detection/lid.py` (top-level), `evaluation/benchmark.py`, `asr/train.py`, `tts/train.py`, Hima pack `hima_onnx` | KEEP |
| `uvicorn>=0.30,<1` (core) | `api/main.py:main()` (`uvicorn.run`), `telugu-api` entry point | KEEP |
| `faster-whisper>=1.0,<2` (`asr`, `host`) | `asr/infer.py:FasterWhisperBackend._load` (lazy), `detection/lid.py` (lazy, int8) | KEEP |
| `onnxruntime>=1.17,<2` (`host`) | Hima pack `hima_onnx` (top-level), `tts/providers/onnx_vits.py` (lazy) | KEEP |
| `safetensors>=0.4,<1` (`host`) | Hima pack `hima_onnx` (`safe_open`, `load_file`) | KEEP |
| `sentencepiece>=0.2,<1` (`host`) | Hima pack `hima_onnx:OnnxTTS.__init__` (`SentencePieceProcessor`, loads `tokenizer.model`) | KEEP |
| `httpx>=0.27,<1` (`dev`) | `fastapi.testclient.TestClient` (used by `tests/test_api.py`, `test_openai_compat.py`, `test_engine_pool.py`, …) and `scripts/test_sarvam.py` (opt-in Sarvam comparison) | KEEP |
| `pytest>=8,<9` (`dev`, `bench`) | the whole `tests/` suite | KEEP |
| `jiwer>=3,<4` (`bench`) | `telugu_speech/bench.py --crosscheck-jiwer` only (lazy import inside the flag branch) | KEEP |

Two direct imports are NOT declared in `pyproject.toml` but are safe
to leave as-is: `anyio` (`routes_openai.py`, `routes_stt.py`,
`routes_asr.py`, `routes_detect.py`) and `starlette`
(`routes_tts.py`). Both are mandatory transitive dependencies of
FastAPI, so any working FastAPI install provides them.

### Used but undeclared — intentionally left out (do NOT add blindly)

| Package | Used by | Why it stays undeclared |
| --- | --- | --- |
| `torch` | `tts/infer.py` (`MmsTtsBackend._load`, `CompactMmsTtsBackend._load`, `IndicF5Backend._load` — all lazy), `asr/infer.py` (`IndicConformerBackend._load` — lazy), `audio/vad.py:silero_vad` (lazy), `compression/export_onnx.py` (lazy), `scripts/compress_models.py` (top-level, one-off model build) | Adding it to any installed extra would pull ~1–2 GB into the server install / Docker image. Every runtime use fails closed with `ModelUnavailable` when torch is absent. Install it manually only if a legacy MMS / IndicConformer / Silero path is actually needed. |
| `transformers` | Same lazy `_load` paths as torch (`tts/infer.py`, `asr/infer.py` via NeMo), `platform/.../translate.py` (lazy, IndicTrans2), `scripts/fetch_mms_tts.py`, `scripts/compress_models.py` | Same as torch: lazy, optional, fails closed. |
| `datasets`, `huggingface_hub` | `scripts/fetch_fleurs_sample.py` (`datasets` + `soundfile`, top-level), `scripts/fetch_models.py` + `scripts/fetch_lite_voice.py` (`huggingface_hub`, lazy) | One-off data/model fetch scripts, not the server. |
| `soundfile` | `audio/resampling.py:read_wav` (optional, `try/except ImportError` → stdlib `wave` fallback), Hima pack `hima_onnx.save()` (never called by this repo — WAVs are written with stdlib `wave`), `scripts/fetch_fleurs_sample.py` | Not needed on the server path. Do not add. |
| `piper-tts` | `tts/providers/piper.py` (lazy; raises a clear `EngineNotLoadedError` telling the operator to `pip install piper-tts`) | Optional future engine; no Piper pack is installed. |
| `sounddevice` / `pyaudio` | `scripts/record_clips.py` (optional; refuses to record without one) | Recording script only. |

Installed versions and installed sizes: **UNMEASURED** (no shell on
this Mac for this worker). Run:

```bash
cd "/Users/visys/.cursor/tts sst"
.venv311/bin/pip list
.venv311/bin/pip freeze > /tmp/vaani-pip-freeze.txt
du -sh .venv311
.venv311/bin/pip show torch transformers 2>&1 | head -40   # expect "not installed" unless Mani installed them by hand
```

There is no `requirements*.txt` at the repo root; `pyproject.toml`
is the only dependency declaration (the single `requirements.txt`
in the tree is the Hima pack's own:
`onnxruntime, numpy, safetensors, sentencepiece, soundfile, pyyaml` —
all covered by core + `[host]` except `soundfile`, see above).

## 2. Lazy loading — verified, no eager model load found

Checked: the FastAPI startup path (`api/main.py:configure()`, called
at import), the core pool (`core/engine.py`), the registry
(`core/registry.py`), the fallback chains (`core/fallback.py`), and
every backend constructor. **No model weights load at import or at
server start.** What runs at startup is cheap:

| Path | What happens at import / startup | Weights loaded? |
| --- | --- | --- |
| `api/main.py` import | `configure()`: validates config, constructs `FasterWhisperBackend` + a TTS backend object, stats model dirs (`_model_bytes`) | NO — both constructors only store paths/strings; weights load in `_load()` on first `transcribe()` / `synthesize()` |
| `core/engine.py` `get_pool()` | Constructs `EnginePool` (semaphores, empty LRU cache); providers resolved by name and imported lazily in `_load_backend()` | NO |
| `core/registry.py` `get_registry()` | Scans `voices/` + `lite/models/` for `manifest.yaml`, validates manifests | NO — but see the checksum note below |
| `tts/voices.py` `backend_for_voice()` | Constructs `HimaTeluguBackend` (stores checkpoint/speaker/seed) | NO — the ONNX engine is built inside `borrow_engine` on first `synthesize()` / `iter_pcm()` |
| `api/main.py` warmup | `_warm_default_voice()` only when `TELUGU_WARMUP=1` (default `0` in `.env.example`) | Only when explicitly enabled |
| `/health`, `/v1/audio/voices`, `/v1/models` | Probe-only: directory-existence checks, registry listing, byte-size stats | NO model load (`component_status` uses `find_spec` + `is_dir`, never a model) |
| `detection/lid.py` | `FasterWhisperLanguageDetector` constructs empty; `WhisperModel` built inside `shared_engine` on first `/api/v1/detect` | NO at startup |

Pinned by the new tests in `tests/test_lightweight.py` (real
constructors, no mocks): `FasterWhisperBackend(...)._model is None`,
`IndicConformerBackend(...)._model is None`, both MMS backends'
`_model`/`_tokenizer` are `None` after construction,
`backend_for_voice("telugu_srinivas")._engine is None`, and
`configure()` leaves `app.state.asr_backend._model` and
`app.state.tts_backend._model` as `None`.

Known first-request cost, NOT changed: the first call that touches
the voice registry (`GET /health`, `GET /v1/audio/voices`,
`list_installed_voices()`) runs `VoiceRegistry.discover()`, and
`_validate_manifest()` calls `verify_checksum()`, which reads and
SHA-256-hashes each registered pack's model file (e.g. the 48 MB
`onnx/voices/step_int8.onnx`). That is disk I/O once per process
(singleton, cached after), not a model load — but on a slow disk it
is the first request's latency, not startup's. Weakening the
checksum was out of scope for this worker; if it ever shows up in
cold-start timing, that is a deliberate safety-vs-latency decision.

Also documented, NOT changed: `app.state.tts_backend` (the legacy
MMS backend object built in `configure()`) is never used for
synthesis by any server route — `/api/v1/tts/generate` and
`/v1/audio/speech` both go through `backend_for_voice()` (Hima
int8). The only server read of it is the `.name` check in
`/api/v1/tts/voices/custom` (IndicF5 gate). Its constructor is
cheap and loads nothing, so removing it would save nothing
measurable and would touch two route files; left alone.

## 3. INT8 on the default path — verified, no switch needed

| Default path | What actually runs | Precision |
| --- | --- | --- |
| ASR default chain `asr.chain: [telugu-asr-micro]` (`configs/vaani.yaml`) | faster-whisper tiny, `models/telugu-asr-micro` (42.2 MiB on disk, BASELINE) | INT8 — `asr/decode.py:DecodingConfig.compute_type = "int8"` (default), passed as `compute_type=` to `WhisperModel` in `asr/infer.py:FasterWhisperBackend._load` |
| ASR alternate `telugu-asr` (small, 241.4 MiB) | Same code path, `models/telugu-asr` | INT8 (same `compute_type`, same int8 `model.bin`) |
| LID (`/api/v1/detect`) | faster-whisper multilingual tiny, `models/whisper-multilingual-tiny` | INT8 — hardcoded `compute_type="int8"` in `detection/lid.py` |
| TTS default chain `tts.chain: [hima-nano]`, voice `telugu_srinivas` | Hima ONNX, `lite/models/hima-tts-nano-te` (62M params) | INT8 — `tts/voices.py` builds `OnnxTTS(..., precision="int8")`; loads `onnx/voices/step_int8.onnx` + `frame_int8.onnx` + `decoder_int8.onnx` |
| TTS alternate voice `telugu_sravani` | Same shared pack, other speaker preset | INT8 (same pack) |

FP32 paths that still exist, and why they were left alone:

* `models/telugu-tts` (Meta MMS Telugu VITS, FP32, 138.5 MiB) loads
  ONLY when the operator explicitly sets `TELUGU_TTS_BACKEND=mms`
  (or `mms-tts-tel`/`mms_tts`) **and** `models/telugu-tts-micro/`
  is absent. The shipped default is `mms-micro` (`.env.example`
  sets no `TELUGU_TTS_BACKEND`, so the `Settings` default
  `mms-micro` applies), and with the micro dir present even an
explicit `mms` setting is routed to the compact backend by
  `configure()`. No default path touches the FP32 file.
* `models/telugu-tts-micro` (CompactMms) is int8 **on disk**
  (`weights.int8.npz` + `scales.json`, 25.1 MiB) but dequantizes
  into FP32 torch tensors in RAM at `_load()` time — so its RAM
  footprint is FP32-sized. It is not on any default serving path
  (see §2: the server synthesises through Hima), and it is only
  reachable via the explicit `mms-micro` legacy backend or
  `scripts/e2e_offline.py`. Left as-is; do not present the micro
  MMS pack as an INT8-RAM option.
* The vendored Hima pack's own default is `precision="fp32"`
  (`hima_onnx/__init__.py:OnnxTTS.__init__`), but this repo always
  passes `precision="int8"`. Upstream default left untouched.
* `IndicConformerBackend` / `IndicF5Backend` need torch +
  transformers / NeMo and are not installed; they fail closed with
  `ModelUnavailable` and are not in any default chain.

No model was downloaded, moved, or deleted. Disk totals unchanged
from BASELINE: `models/` 547,149,697 B (~521.8 MiB); the duplicate
`lite/models/telugu-asr-micro/` copy was not deduplicated (whether
it is a copy or a hard link was not verified through the file API —
left alone, do not delete on a guess).

## 4. Docker — already slim; one fix

The Dockerfile was already slim before this worker: `python:3.11-slim`
base, `ffmpeg` only (`--no-install-recommends`, apt lists removed),
`PIP_NO_CACHE_DIR=1`, non-root user `vaani` (uid 10001), `pip install
--user ".[host]"`, no torch, no CUDA, no model weights in the image
(models come from the `./models` + `./lite` read-only mounts and the
`vaani-cache` volume; the entrypoint refuses to start when a required
model folder is absent). No multi-stage rebuild was done: there is no
compiler toolchain in the image to strip, so a second stage would
save nothing meaningful. `docker-compose.yml` needed no change.

The one fix (in `Dockerfile`):

```dockerfile
COPY --chown=vaani:vaani configs ./configs
```

Why: the image previously shipped **without** `configs/`, so in a
container (a) the shipped `configs/vaani.yaml` was silently ignored
— `core/config.py:_find_config_file()` falls back to built-in
defaults when the file is absent (same values today, but any edit to
`vaani.yaml` would never take effect in Docker), and (b)
`POST /api/v1/detect` always returned 503 because
`configs/detection.yaml` was missing (`routes_detect.py` requires
that file). `configs/` is text only (yamls, one small csv, voice
manifests) — no weights, no secrets — so the image stays slim.
Behaviour on the host is unchanged: the file values equal the
built-in defaults today.

Image size: **UNMEASURED** (no Docker build was run by this worker).
Record it after building:

```bash
docker build -t vaani-x:0.2.0-slim .
docker images vaani-x
# then write the number into docs/DEPLOYMENT_TIERS.md (it currently says the size is not claimed)
```

Docker gap noted, not fixed: `VAANI_AUTO_FETCH=1` in the entrypoint
runs `scripts/fetch_lite_voice.py`, which imports `huggingface_hub`.
That package is **not** a declared dependency — it is only present
in the image transitively via `faster-whisper` (unverified here;
see §1). If auto-fetch ever fails in the container with
`ModuleNotFoundError: huggingface_hub`, either mount the pack
(the documented path) or verify with `.venv311/bin/pip show
huggingface-hub` on the host first. No pin was added for it: the
right bound could not be verified from the file API alone.

## 5. Before / after

| Item | Before (BASELINE, 22 Nov 2026) | After (this worker) |
| --- | --- | --- |
| Declared dependencies | core 6, `asr` 1, `host` 4, `dev` 2, `bench` 2 (`pyproject.toml`) | **Unchanged** — every one is imported; nothing dropped, nothing added |
| Undeclared but used | torch / transformers / datasets / huggingface_hub / soundfile (BASELINE noted soundfile + datasets + huggingface_hub) | **Unchanged, now documented** in §1 + a `pyproject.toml` comment — deliberately not added |
| Models on disk | `models/` 547,149,697 B (~521.8 MiB); Hima pack ~151 MB | **Unchanged** — no model added, moved, or deleted |
| Model load at import/startup | None (constructors lazy; verified by re-reading `main.py`, `core/engine.py`, `asr/infer.py`, `tts/infer.py`, `tts/voices.py`) | **Unchanged, now pinned** by `tests/test_lightweight.py` (5 tests) |
| Default ASR precision | int8 (`compute_type="int8"`) | **Unchanged** |
| Default TTS precision | Hima int8 (`precision="int8"`) | **Unchanged** |
| FP32 MMS on any default path | None | **Unchanged** (reachable only via explicit `TELUGU_TTS_BACKEND=mms` + missing micro dir) |
| Dockerfile | slim, no `configs/` | **+1 COPY line** (`configs/`); base, install, entrypoint unchanged |
| docker-compose.yml | models + lite mounted read-only, cache volume | **Unchanged** |
| Default-path RAM (≤1–2 GB target) | Small-model peak RSS ~956 MB vs tiny ~362 MB (20 Nov 2026 bench, BASELINE); live-decoder peak RSS **UNMEASURED** | **UNMEASURED — no change claimed.** See commands below. |
| Cold start (import → first response) | **UNMEASURED** | **UNMEASURED — no change claimed.** See commands below. |
| `.venv311` size / installed versions | **UNMEASURED** | **UNMEASURED.** See §1 commands. |
| Docker image size | **UNMEASURED** | **UNMEASURED.** See §4 commands. |

## 6. Commands for Mani (run from the repo root)

```bash
cd "/Users/visys/.cursor/tts sst"

# 1. Tests — full suite, then just the new lightweight pins
.venv311/bin/python -m pytest -q
.venv311/bin/python -m pytest -q tests/test_lightweight.py

# 2. Installed deps + venv size (fills the UNMEASURED rows in §1/§5)
.venv311/bin/pip list
du -sh .venv311
du -sh . models lite/models data

# 3. Cold start: import time only (no model load expected), then time to
#    first health response. Neither loads weights.
time .venv311/bin/python -c "import telugu_speech.api.main; print('imported, no weights')"
# Start the server as before (kill any stale :8000 listener first), then:
curl -s http://127.0.0.1:8000/api/v1/health
curl -s http://127.0.0.1:8000/health

# 4. Smoke the protected routes (replace $KEY with your key from
#    TELUGU_API_KEYS; the server must already be running)
curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:8000/v1/models
curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:8000/v1/audio/voices
curl -s -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "vaani-tts-hima", "input": "నమస్కారం", "voice": "srinivas", "response_format": "wav"}' \
  http://127.0.0.1:8000/v1/audio/speech -o /tmp/vaani-smoke.wav && ls -l /tmp/vaani-smoke.wav
curl -s -H "Authorization: Bearer $KEY" -F file=@data/raw/fleurs_te/fleurs_000.wav \
  -F model=vaani-asr-small http://127.0.0.1:8000/v1/audio/transcriptions

# 5. Peak RSS while both engines are resident (run after one TTS + one ASR call):
.venv311/bin/python -c "import resource; print('peak RSS MB:', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)"
# (That prints the python -c process only. For the server process, use Activity
# Monitor or: ps -o rss= -p <server pid> — record the number with the call mix.)

# 6. Docker (only if you want the image rebuilt with configs/):
docker build -t vaani-x:0.2.0-slim .
docker images vaani-x
printf 'TELUGU_API_KEYS=choose-a-long-random-key\n' > .env   # only if no .env yet
docker compose up --build -d
curl http://127.0.0.1:8000/api/v1/health
```

## 7. Open items / deliberately not done

* **Slot default mismatch (needs Mani's decision, not silently
  changed):** the legacy TTS pool (`tts/runtime.py:speech_slots()`)
  defaults to **8** concurrent engine copies, and `.env.example`
  also shows `TELUGU_SPEECH_SLOTS=8` — while the v2.1 config
  (`configs/vaani.yaml`, `core/config.py`) and `docker-compose.yml`
  all use **2**, and the config docs say 2 is the safe default on
  1–2 GB (one engine ~260–470 MB, so 8 overlapping copies could
  need ~2–3.7 GB). Two existing tests pin the 8
  (`tests/test_tts.py::test_speech_slots_default_and_cap`,
  `tests/test_engine_pool.py::test_slots_are_clamped`), so changing
  the default would also mean changing tests — that is a deliberate
  behaviour + test change, not a lightweight fix. The live `/v1`
  routes still use the legacy pool (pool migration is a documented
  v2.1 gap), so the 8 is the live cap on the Mac when
  `TELUGU_SPEECH_SLOTS` is unset; Docker is already capped at 2 by
  compose. Decide: keep 8 on the Mac, or set `TELUGU_SPEECH_SLOTS=2`
  in the Mac environment.
* **Registry checksum on first request** (§2): first
  `/health` / `/v1/audio/voices` call hashes pack model files.
  Left as-is (safety trade-off).
* **`lite/models/telugu-asr-micro/` duplicate** of
  `models/telugu-asr-micro/` (same byte sizes, BASELINE): not
  deduplicated — copy-vs-hardlink unverified, do not delete on a
  guess.
* **No cold-start / RAM / image-size numbers were measured** by
  this worker (no shell on the Mac for it). Every such number above
  is marked UNMEASURED with its command; none is presented as a
  result.
* **Not touched:** decoding parameters, beam/VAD settings, model
  defaults, `docs/ACCURACY.md`, routes, auth, API shapes,
  `configs/vaani.yaml` values.
