# configs/voices — voice manifests (source of truth for voices)

These YAML files are the **source manifests** for the two shipped voices.
They are NOT the live packs. The registry discovers live packs in:

1. `VAANI_VOICES_DIR` (colon-separated, in order)
2. `<project>/voices/` — canonical home for new packs
3. `<project>/lite/models/` — legacy home

## Deploy a voice (USB-style: plug it in, no code change)

```bash
# 1. Fill in the TBD fields in srinivas.yaml / sravani.yaml from the real pack:
sha256sum "lite/models/hima-tts-nano-te/<model file>"   # -> sha256
#    sample_rate + licence + licence_url from the pack's config.json / voice card

# 2. Copy each manifest into its pack folder:
mkdir -p voices/telugu_srinivas voices/telugu_sravani
cp configs/voices/srinivas.yaml voices/telugu_srinivas/manifest.yaml
cp configs/voices/sravani.yaml  voices/telugu_sravani/manifest.yaml

# 3. Reload without restarting the server (v2.1 — live):
#    curl -s -X POST -H "Authorization: Bearer $TELUGU_API_KEYS" \
#      http://127.0.0.1:8000/v1/reload
#    It re-reads configs/vaani.yaml, re-discovers voice packs, and returns
#    {reloaded, voices, rejected, warnings}. A bad config fails the reload
#    (400) and the last good config keeps serving — the server never dies
#    on a bad reload. See docs/CONFIGURATION.md.
```

Or point the registry at this directory directly while developing:

```bash
VAANI_VOICES_DIR="$PWD/configs/voices" uvicorn telugu_speech.api.main:app
```

Note: `model_path` / `config_path` in a manifest are resolved **relative
to the folder containing that manifest**, so a manifest copied into
`voices/telugu_srinivas/` uses `../../lite/models/...` to reach the
shared lite model. If a voice ships its own model files, put them in
its pack folder and use plain filenames (`model.onnx`, `config.json`).

## Adding a third voice (e.g. a Piper voice)

1. Create `voices/telugu_<name>/` with the model + config files.
2. Copy this manifest shape, change `id`, `name`, `gender`, `engine`
   (`piper` for a Piper voice), `model_path`, `config_path`,
   `sample_rate`, `licence`, `licence_url`, `sha256`, `version`, `tags`.
3. `sha256sum` the model file — the registry verifies it on discovery.
4. Reload. No core, registry, API or test change is needed.

## Manifest contract (exact — other agents code against it)

| Field | Required | Notes |
|---|---|---|
| `id` | yes | unique, `[a-z][a-z0-9_]*`, e.g. `telugu_srinivas` |
| `name` | yes | display name, e.g. `Srinivas` |
| `gender` | yes | `male` \| `female` \| `neutral` |
| `language` | yes | `te` in v2 Part 1 |
| `engine` | yes | `hima-nano` \| `piper` \| `onnx-vits` |
| `model_path` | yes | relative to pack folder; file must exist |
| `config_path` | yes | relative to pack folder; file must exist |
| `sample_rate` | yes | Hz, int, e.g. `22050` |
| `licence` | yes | SPDX id or short name; `TBD` is NOT shippable |
| `licence_url` | yes | URL of licence / voice card |
| `sha256` | yes | 64 hex chars, SHA-256 of the model file |
| `version` | yes | e.g. `"1.0.0"` (quote it — YAML would read `1.0` as a float otherwise) |
| `tags` | no | list of strings, e.g. `[male, conversational]` |

A pack failing any of these is **rejected with a named reason** in
`registry.rejected` and the server keeps running with the good packs.
See `docs/VOICE_PACK.md` for the full contract and failure modes.

## v2.1 — config and health

- Voice discovery dirs, idle unload, and the default voice now live in
  `configs/vaani.yaml` (`voices.dirs`, `voices.idle_secs`,
  `tts.default_voice`) with `VAANI_*` env overrides. See
  `docs/CONFIGURATION.md`.
- `GET /v1/config` (auth required) shows the effective config and where
  each value came from; `GET /api/v1/health` reports per-voice status
  and lists rejected packs with their reasons.
- **Still open (Mani's step, not fixed by v2.1):** both manifests ship
  with a zero `sha256`, so the registry rejects both packs until the
  real hash is pasted in (see the STATUS note at the top of each yaml).
  The legacy fallback keeps Srinivas and Sravani working meanwhile.
