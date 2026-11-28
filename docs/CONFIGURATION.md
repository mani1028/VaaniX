# Vaani X v2.1 — Configuration, fallback chains, hot reload, graceful degradation

> Status: built 2026-11-21, deployed to the Mac the same day. NOT verified
> — there is no shell on the Mac for the builder, so nothing here is
> claimed to pass. Mani runs the checks at the bottom. The v2.0 core
> (registry, engine pool, OpenAI routes, bench harness, slim Docker) is
> unchanged in behaviour; this part only adds flexibility around it.
>
> **Still open (not fixed here):** both voice manifests ship with a zero
> `sha256`, so the registry rejects both packs until Mani runs
> `shasum -a 256 lite/models/hima-tts-nano-te/onnx/voices/step_int8.onnx`
> and pastes the hash into `configs/voices/srinivas.yaml` and
> `configs/voices/sravani.yaml`, then copies them to
> `voices/*/manifest.yaml`. The legacy fallback keeps both voices
> working meanwhile.

## 1. One config file: `configs/vaani.yaml`

Everything a developer tunes lives in one file. Precedence, highest wins:

```
environment variable (VAANI_*)  >  configs/vaani.yaml  >  built-in default
```

The file is found via `VAANI_CONFIG`, else `<project>/configs/vaani.yaml`
if it exists, else built-in defaults alone. If `VAANI_CONFIG` names a
file that does not exist, startup fails with a clear error — it does
not silently fall through to defaults.

### 1A. Every key

| Key (file) | Env override | Default | Allowed |
|---|---|---|---|
| `server.host` | `VAANI_HOST` / `TELUGU_HOST` | `127.0.0.1` | any non-empty string |
| `server.port` | `VAANI_PORT` / `TELUGU_PORT` | `8000` | 1–65535 |
| `server.rate_limit_per_minute` | `VAANI_RATE_LIMIT_PER_MINUTE` | `600` | 0–100000 (0 = no limit) |
| `server.cors_origins` | `VAANI_CORS_ORIGINS` / `TELUGU_CORS_ORIGINS` | `[]` | list / comma-separated; `*` is ignored |
| `server.limit_concurrency` | `VAANI_LIMIT_CONCURRENCY` / `TELUGU_LIMIT_CONCURRENCY` | `256` | 1–10000 |
| `asr.chain` | `VAANI_ASR_CHAIN` (comma-sep) | `[telugu-asr-micro]` | non-empty list of known/unknown names (unknown entries are skipped, §2) |
| `asr.model` | `VAANI_ASR_MODEL` / `TELUGU_WHISPER_MODEL` | `models/telugu-asr-micro` | any non-empty string |
| `asr.backend` | `VAANI_ASR_BACKEND` / `TELUGU_ASR_BACKEND` | `faster-whisper` | `faster-whisper` \| `indicconformer` |
| `asr.decoding` | `VAANI_ASR_DECODING` / `TELUGU_ASR_DECODING` | `ctc` | `ctc` \| `rnnt` (indicconformer only) |
| `tts.chain` | `VAANI_TTS_CHAIN` (comma-sep) | `[hima-nano]` | non-empty list of engine names |
| `tts.default_voice` | `VAANI_DEFAULT_VOICE` | `telugu_srinivas` | any non-empty string |
| `tts.default_engine` | `VAANI_DEFAULT_ENGINE` | `hima-nano` | must be in `tts.chain` |
| `limits.tts_slots` | `VAANI_TTS_SLOTS` / `TELUGU_SPEECH_SLOTS` | `2` | 1–4 |
| `limits.asr_slots` | `VAANI_ASR_SLOTS` | `2` | 1–4 |
| `limits.cpu_threads` | `VAANI_CPU_THREADS` / `TELUGU_CPU_THREADS` | `2` | 1–4 |
| `limits.max_text_chars` | `VAANI_MAX_TEXT_CHARS` | `2000` | 1–20000 |
| `limits.max_chunk_chars` | `VAANI_MAX_CHUNK_CHARS` | `180` | 100–2000 (180 is the Hima speech-piece limit) |
| `limits.max_upload_mb` | `VAANI_MAX_UPLOAD_MB` / `TELUGU_MAX_UPLOAD_BYTES` (bytes!) | `25` | 1–50 MB |
| `limits.max_audio_mb` | `VAANI_MAX_AUDIO_MB` | `25` | 1–100 MB |
| `cache.dir` | `VAANI_CACHE_DIR` | `~/.cache/vaani` | any path |
| `cache.audio_cache_entries` | `VAANI_AUDIO_CACHE_ENTRIES` | `128` | 0–1024 (0 = cache off) |
| `cache.audio_cache_max_mb` | `VAANI_AUDIO_CACHE_MAX_MB` | `256` | 0–2048 |
| `voices.dirs` | `VAANI_VOICES_DIR` (colon-sep) | `[]` | list of dirs |
| `voices.idle_secs` | `VAANI_VOICE_IDLE_SECS` | `1800` | 0–86400 (0 = never unload) |
| `offline` | `VAANI_OFFLINE` | `false` | true/false/1/0/yes/no/on/off |
| `log_level` | `VAANI_LOG_LEVEL` | `INFO` | DEBUG \| INFO \| WARNING \| ERROR |

Notes:
- `limits.max_text_chars` (2000) is the **core** TTS cap. The OpenAI
  `/v1/audio/speech` input cap is 4096, enforced separately by its
  request schema. Do not confuse the two.
- `limits.max_upload_mb` (legacy `/api/v1/*`) and `limits.max_audio_mb`
  (OpenAI `/v1/*`) are both 25 MB by default, matching OpenAI's cap.
- `TELUGU_MAX_UPLOAD_BYTES` is in **bytes** and is converted to MB
  (rounded up) before validation; `VAANI_MAX_UPLOAD_MB` is in MB.
- No secrets in this file. API keys stay in `TELUGU_API_KEYS` (env).
- Unknown keys in the file are an error, not silently ignored.

### 1B. Strict validation — never a silent default

`core/config.py:load_config()` validates every resolved value and raises
`ConfigError` naming the key, the value supplied, and what is allowed:

```
config key 'limits.tts_slots': value 9 from env:VAANI_TTS_SLOTS is out of range 1-4
config key 'asr.backend': value 'whisper' from file must be one of faster-whisper, indicconformer
config file 'configs/vaani.yaml': unknown key(s): server.foo
```

- **At startup** (`api/main.py:configure()` calls `get_config()`), a
  bad value fails startup. The server refuses to start misconfigured.
- **On reload** (`POST /v1/reload`), a bad value fails the reload with
  HTTP 400 and the last good config keeps serving (§4).
- The legacy `Settings` object (`core/config.py`) keeps its old
  silent-fallback behaviour for backwards compatibility only. New code
  must use `get_config()` / `EffectiveConfig`.

## 2. Fallback chains — flexible, never silent

Defined in `core/fallback.py`, driven by `configs/vaani.yaml`.

### 2A. ASR chain (`asr.chain`)

```yaml
asr:
  chain: [telugu-asr-micro, telugu-asr, vasista22-small]
```

The pool tries entries in order. The first that constructs wins. A
missing or unhealthy entry is skipped, logged, and recorded in the
`attempted` list returned with every transcription
(`{text, language, segments, duration, model, attempted}`). If none
work, the call is a 503 naming **everything** that was tried and why.

Known entries (what is actually installed matters — a name is not a model):

| Entry | Kind | Location | Installed? |
|---|---|---|---|
| `telugu-asr-micro` | faster-whisper | `models/telugu-asr-micro` | **Yes — shipped** |
| `telugu-asr` | faster-whisper | `models/telugu-asr` | Only if the folder exists |
| `vasista22-small` | faster-whisper | `models/vasista22-whisper-telugu-small` | **No** — recommended upgrade (Part B: FLEURS 24.0%); fetch with `scripts/fetch_models.py` once revision/SHA are filled in |
| `vasista22-base` | faster-whisper | `models/vasista22-whisper-telugu-base` | **No** |
| `indicconformer-600m` | indicconformer | `ai4bharat/indicconformer_stt_te_hybrid_ctc_rnnt_large` | **No** — needs torch + NeMo + gated HF checkpoint |

Unknown chain entries are skipped with reason `unknown backend`, not a
startup error — a config that lists a future backend does not break
today's server. Health probes (`probe_asr_entry`) check only that a
model directory exists or that torch/NeMo are importable; **no model
is ever loaded by a probe**.

### 2B. TTS chain (`tts.chain`)

Engines, not voices: `[hima-nano]` today; `piper` and `onnx-vits` are
provider modules that exist, but no Piper/onnx-vits voice pack is
installed in this checkout, so listing them changes nothing until a
pack for that engine is dropped into `voices/`. A voice pack names its
own engine in its manifest; the chain is the order engines are tried
when a voice does not name one, and the order listed in `/v1/models`.

### 2C. Per-request override — explicit, never a silent swap

- ASR: `model=<name>` on `POST /v1/audio/transcriptions` (or
  `EnginePool.transcribe(path, model=<name>)`) selects exactly one
  chain entry. Unknown name → **404** `model_not_found`. Known but
  unhealthy → **503** naming that entry. It is never swapped for
  another recogniser.
- TTS: `model=<name>` on `POST /v1/audio/speech` (or
  `EnginePool.synthesize(text, voice, engine=<name>)`) must match the
  voice pack's engine, else **400** — a request for a Piper voice never
  silently falls back to Hima.
- The caller is told what actually ran:
  - Response headers `X-Vaani-Model` / `X-Vaani-Voice` (set by the
    middleware in `api/main.py` from the pool's record of what ran;
    for `/v1/audio/speech` the header is derived from the request
    JSON until that route migrates to the pool — see §6).
  - ASR results carry `model` (the entry that ran) and `attempted`
    (every entry tried, with outcome).

## 3. Graceful degradation — one failure never takes the server down

- **Missing voice pack** → 503 for that voice only; other voices keep
  working. The pack appears in `registry.rejected` with the reason.
- **Missing ASR model** → transcriptions are 503 naming what was
  tried; TTS still works, and vice versa. The two tasks never share a
  failure mode: TTS engines are per-voice, ASR is one shared instance,
  and neither is loaded at startup (lazy on first use).
- **Bad config file** → startup fails loudly (§1B), or a reload fails
  and the last good config keeps serving (§4).
- **Bad voice manifest** → rejected with the named reason; the server
  keeps running with the good packs.

## 4. Hot reload — `POST /v1/reload` and `GET /v1/config`

Both require auth (Bearer or `X-API-Key`, same keys as everything else).

```bash
# What is the server actually running on, and where did each value come from?
curl -s -H "Authorization: Bearer $TELUGU_API_KEYS" http://127.0.0.1:8000/v1/config

# Re-read configs/vaani.yaml + env, re-discover voices, apply to the live pool:
curl -s -X POST -H "Authorization: Bearer $TELUGU_API_KEYS" http://127.0.0.1:8000/v1/reload
```

`POST /v1/reload` returns:

```json
{
  "reloaded": true,
  "errors": [],
  "warnings": ["..."],
  "voices": ["telugu_srinivas", "telugu_sravani"],
  "models": ["vaani-tts-hima-nano", "vaani-asr-telugu-asr-micro"],
  "rejected": [{"pack_dir": "...", "reason": "..."}],
  "config": {"...redacted effective config...", "sources": {"limits.tts_slots": "env:VAANI_TTS_SLOTS"}}
}
```

What applies when:

| Change | When it applies |
|---|---|
| Voice packs (add/remove/fix manifest) | Immediately on reload; removed voices are unloaded |
| ASR chain | Next transcription (already-loaded backends are kept) |
| Cache limits, idle timeout, text/audio caps | Immediately (cache limits clear the audio cache — reported in `warnings`) |
| Slot counts | Immediately **only if no slot is in use**; otherwise the old count stays and `warnings` says so — restart to force it |
| CPU threads | Engines loaded **after** the reload (ONNX reads them at session creation) |
| Rate limit, upload cap (`/v1/*`) | Immediately (the reload handler updates the live limiter) |
| CORS origins, host, port | **Restart required** (middleware / uvicorn are fixed at startup) |
| `offline`, `log_level`, `cache.dir` | Immediately (recorded; `cache.dir` is used by fetch/entrypoint scripts, not by the in-memory audio cache) |

A bad reload returns HTTP 400 with `reloaded: false`, `errors` naming
the bad key/value, and the **current (last good)** config. The server
keeps serving throughout.

## 5. Health — per-component status

`GET /api/v1/health` (no key) now returns:

```json
{
  "status": "degraded",
  "auth_mode": "required",
  "version": "0.2.0",
  "components": {
    "overall": "degraded",
    "asr": {"status": "ok", "chain": [{"name": "telugu-asr-micro", "ok": true, "reason": "model directory present"}], "loaded": false, "default": "telugu-asr-micro"},
    "tts": {"status": "degraded", "voices": {"telugu_srinivas": {"status": "ok (legacy fallback)", "engine": "hima-onnx-int8", "loaded": false}}, "default_voice": "telugu_srinivas", "rejected": [{"pack_dir": "...", "reason": "sha256 mismatch ..."}]},
    "registry": {"status": "degraded", "rejected": [...]},
    "config": {"status": "ok", "offline": false, "file": ".../configs/vaani.yaml"}
  }
}
```

Status values: `ok` | `degraded` | `unavailable`. `status` is `"ok"`
only when every component is ok. The example above is the **current
shipped state**: the registry rejects both packs (zero sha256) so
health honestly says `degraded`, while the legacy fallback still serves
both voices. Health probes never load a model.

## 6. Known gaps in this part (honest, not hidden)

1. **`/v1/audio/speech` and `/v1/audio/transcriptions` do not use the
   core pool yet.** They use the legacy `backend_for_voice()` and
   `app.state.asr_backend` (Part 2). The pool's chain, per-request
   override, and `model`/`attempted` result fields exist and are tested
   by Part 3's pool tests, but the `/v1` routes do not call them. The
   `X-Vaani-Model` / `X-Vaani-Voice` headers for `/v1/audio/speech`
   are therefore derived from the request JSON (requested, not
   measured); for transcriptions the header is absent until the route
   migrates to the pool. Migrating those two routes to
   `EnginePool.synthesize/transcribe` is the next API step — it is a
   small, well-defined change (route handlers only) and must keep the
   old `/api/v1/*` behaviour untouched.
2. **Rate limit is in-memory, per-process.** It does not survive a
   restart and does not scale past one process. Production still needs
   TLS + real quotas on a proxy (the process does not terminate TLS).
3. **Slot counts cannot resize mid-flight** (§4 table).
4. **CORS/host/port need a restart** (§4 table).
5. **The sha256 blocker is Mani's step** (top of this file) — v2.1
   does not and cannot compute it (no shell on the Mac).

## 7. Verification (Mani — run on the Mac, repo root)

```bash
cd "/Users/visys/.cursor/tts sst"
.venv311/bin/python -m py_compile src/telugu_speech/core/config.py src/telugu_speech/core/fallback.py src/telugu_speech/core/registry.py src/telugu_speech/core/engine.py src/telugu_speech/api/main.py
.venv311/bin/python -m pytest -q
# Config validation: a bad value must fail loudly, not silently default:
VAANI_TTS_SLOTS=9 .venv311/bin/python -c "from telugu_speech.core.config import load_config; load_config()"  # expect ConfigError naming limits.tts_slots
VAANI_ASR_BACKEND=whisper .venv311/bin/python -c "from telugu_speech.core.config import load_config; load_config()"  # expect ConfigError naming asr.backend
# Server:
telugu-api  # separate terminal, with TELUGU_API_KEYS set
curl -s http://127.0.0.1:8000/api/v1/health
curl -s -H "Authorization: Bearer $TELUGU_API_KEYS" http://127.0.0.1:8000/v1/config
curl -s -X POST -H "Authorization: Bearer $TELUGU_API_KEYS" http://127.0.0.1:8000/v1/reload
# Bad reload must keep the server up (edit vaani.yaml: tts_slots: 9, reload -> 400, health still answers, then restore)
# Regression: old + new routes still work:
curl -s -H "Authorization: Bearer $TELUGU_API_KEYS" -H "Content-Type: application/json" -d '{"text":"నమస్కారం","language":"te","voice":"telugu_sravani","format":"wav"}' http://127.0.0.1:8000/api/v1/tts/generate
curl -s -H "Authorization: Bearer $TELUGU_API_KEYS" -H "Content-Type: application/json" -d '{"model":"vaani-tts-hima","voice":"srinivas","input":"నమస్కారం","response_format":"wav"}' http://127.0.0.1:8000/v1/audio/speech -o /tmp/speech.wav -D -  # check X-Vaani-Model / X-Vaani-Voice headers
```
