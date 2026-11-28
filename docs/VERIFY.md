# Verify Vaani X on the Mac — the correct commands

This file exists because the first verification run on 2026-11-21 failed
in a misleading way. Read the root causes first, then run the steps in
order. All commands run in a terminal on the Mac, from the repo root.

## Root causes of the 2026-11-21 run

### 1. `zsh: command not found: telugu-api`

`telugu-api` is a console script created by `pip install -e ".[host]"`
into `.venv311/bin/`. It is **not** on the shell PATH unless the venv
is activated. The script itself still exists in the project:
`pyproject.toml` `[project.scripts]` still maps
`telugu-api = "telugu_speech.api.main:main"`. Nothing was deleted.

Fix: use the full path `.venv311/bin/telugu-api`, **or** activate the
venv first (`source .venv311/bin/activate`, then `telugu-api`). If
`.venv311/bin/telugu-api` does not exist at all, the package was never
installed into that venv — run `pip install -e ".[host]"` inside it.

### 2. All three curls returned `{"detail":"Not Found"}`

That is a FastAPI 404 body, which means **something answered on port
8000** — if no server were running at all, curl would have printed
`Failed to connect to 127.0.0.1 port 8000`, not a JSON 404. So either
an old/stale process (an earlier `telugu-api`, or another app) already
owned port 8000 and did not have these routes, or the server that
answered was not this project.

All three routes **do exist in this codebase** (checked 2026-11-21 in
`src/telugu_speech/api/main.py`):

- `GET /api/v1/health` — canonical v1 health, per-component status.
- `GET /v1/config` — auth required, redacted effective config.
- `POST /v1/reload` — auth required, hot reload.
- `GET /health` — OpenAI-surface liveness, no key (in
  `src/telugu_speech/api/routes_openai.py`, router prefix `/v1`).

No route wiring was changed to fix this: the 404s were a startup /
stale-process problem, not a missing-route problem. Step 3 below
checks what owns port 8000 before starting the server.

### 3. The SHA-256 blocker is fixed (2026-11-21)

Mani measured on the Mac:

```text
shasum -a 256 lite/models/hima-tts-nano-te/onnx/voices/step_int8.onnx
# a2d0dbbac3cd39f86d7b3125c405418a02d0cf1132b9e3cf64546ca4f3487f22
```

That hash is now in `configs/voices/srinivas.yaml` and
`configs/voices/sravani.yaml`, and copies are deployed at
`voices/telugu_srinivas/manifest.yaml` and
`voices/telugu_sravani/manifest.yaml`. The registry can now verify
both packs instead of rejecting them.

## Steps

### Step 0 — go to the repo

```bash
cd "/Users/visys/.cursor/tts sst"
```

### Step 1 — tests

```bash
.venv311/bin/python -m py_compile src/telugu_speech/core/config.py src/telugu_speech/core/fallback.py src/telugu_speech/core/registry.py src/telugu_speech/core/engine.py src/telugu_speech/api/main.py
.venv311/bin/python -m pytest -q
```

Expected: all tests pass. If `.venv311/bin/python` does not exist,
create the venv first: `python3.11 -m venv .venv311 && source
.venv311/bin/activate && pip install -e ".[host]"` (plus
`pip install -e ".[dev,bench]"` for pytest/jiwer).

### Step 2 — check what owns port 8000

```bash
lsof -i :8000 -sTCP:LISTEN
```

If a process is listed and it is not the server you are about to
start, stop it first (`kill <PID>`), otherwise your new server cannot
bind and the old one keeps answering 404s for the new routes.

### Step 3 — set keys and start the server (separate terminal)

In a **separate terminal window** (the server runs in the foreground):

```bash
cd "/Users/visys/.cursor/tts sst"
export TELUGU_SPEECH_AUTH=required
export TELUGU_API_KEYS='choose-a-long-random-key'
export TELUGU_HOST=127.0.0.1
export TELUGU_PORT=8000
export BHASHA_ENABLE_PLATFORM=0
.venv311/bin/telugu-api
```

Alternative with the venv activated:

```bash
source .venv311/bin/activate
telugu-api
```

Wait for the uvicorn startup lines (it prints the address it is
listening on). Do **not** run the curls in this terminal — the server
is in the foreground here.

### Step 4 — health (no key needed)

In the first terminal:

```bash
curl -s http://127.0.0.1:8000/api/v1/health
curl -s http://127.0.0.1:8000/health
```

Expected: JSON with `"status"` (`"ok"` when every component is up,
`"degraded"` with a `components` section naming what is missing —
never `{"detail":"Not Found"}`). `/api/v1/health` is the canonical
v1 health route; `/health` is the OpenAI-surface liveness alias.

### Step 5 — config + reload (key required)

```bash
export TELUGU_API_KEYS='choose-a-long-random-key'   # same key as Step 3
curl -s -H "Authorization: Bearer $TELUGU_API_KEYS" http://127.0.0.1:8000/v1/config
curl -s -X POST -H "Authorization: Bearer $TELUGU_API_KEYS" http://127.0.0.1:8000/v1/reload
```

Expected from reload: `"reloaded": true` and
`"voices": ["telugu_srinivas", "telugu_sravani"]`. If `reloaded` is
false, the response `errors` field names the bad config key — the
server keeps serving the last good config.

### Step 6 — smoke the speech routes (key required)

```bash
curl -s -H "Authorization: Bearer $TELUGU_API_KEYS" -H "Content-Type: application/json" \
  -d '{"model":"vaani-tts-hima","voice":"srinivas","input":"నమస్కారం","response_format":"wav"}' \
  http://127.0.0.1:8000/v1/audio/speech -o /tmp/speech.wav
file /tmp/speech.wav
curl -s -H "Authorization: Bearer $TELUGU_API_KEYS" \
  -H "Content-Type: application/json" \
  -d '{"text":"నమస్కారం","language":"te","voice":"telugu_sravani","format":"wav"}' \
  http://127.0.0.1:8000/api/v1/tts/generate
curl -s -H "Authorization: Bearer $TELUGU_API_KEYS" \
  -F "audio=@/tmp/speech.wav;type=audio/wav" -F "model=vaani-asr-small" -F "language=te" \
  -F "response_format=json" http://127.0.0.1:8000/v1/audio/transcriptions
```

Expected: `/tmp/speech.wav` is a real WAV file, the TTS generate call
returns JSON with an `audio_url`, and the transcription returns the
spoken text. A 401 means the key does not match Step 3; a 503 names
the missing component in its `detail`/`error` body.

## If it still fails

- `command not found: telugu-api` → you used the bare name without
  activating the venv. Use `.venv311/bin/telugu-api`.
- `Failed to connect` → the server is not running in the other
  terminal. Check its output for a startup error (a bad config value
  fails startup with a `ConfigError` naming the key).
- `{"detail":"Not Found"}` on a route listed above → a stale process
  owns port 8000 (Step 2). Kill it and restart.
- `{"detail":{"code":"model_unavailable"}}` → that component's
  weights are missing; the message names the path.
- New files may be macOS-quarantined after being copied to the Mac.
  If an import looks odd, check with `xattr` on the file.
