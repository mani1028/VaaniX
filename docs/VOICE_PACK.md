# VOICE_PACK.md — the USB connector contract

A voice is a **folder with a `manifest.yaml`**. That is the whole
contract. If a folder satisfies it, it works in Vaani X. If it does not,
the registry rejects it with a named reason and the server keeps running.

## 1. Pack layout

```
voices/telugu_srinivas/
├── manifest.yaml        # required — the contract below
├── model.onnx           # required — the model file (name is yours; manifest points at it)
└── config.json          # required — the model's config (name is yours; manifest points at it)
```

`model_path` and `config_path` in the manifest are resolved **relative
to the pack folder**. A pack may share a model with another voice by
using `../../lite/models/...` paths — but the shared file must still
exist and its SHA-256 must still match.

## 2. Manifest contract (exact)

```yaml
id: telugu_srinivas
name: Srinivas
gender: male            # male | female | neutral
language: te            # Telugu only in v2 Part 1
engine: hima-nano       # hima-nano | piper | onnx-vits
model_path: model.onnx
config_path: config.json
sample_rate: 22050      # Hz, int
licence: Apache-2.0     # SPDX id or short name — TBD is NOT shippable
licence_url: https://example.com/voice-card
sha256: <64 lowercase hex chars>   # SHA-256 of the file at model_path
version: "1.0.0"        # quote it
tags: [male, conversational]       # optional
```

## 3. How to add a voice (5 steps, no code)

1. Make the folder: `mkdir -p voices/telugu_<name>/` and put the model +
   config files in it.
2. Write `manifest.yaml` in that folder using the contract above.
3. Hash the model: `sha256sum voices/telugu_<name>/model.onnx` and paste
   the hash into `sha256`. (A wrong hash is a **rejection**, not a warning.)
4. Check the licence: read the voice card at `licence_url` and write the
   real licence into `licence`. Piper voices each carry their own licence;
   XTTS v2 is CPML (non-commercial, no licensor left) and MMS is
   CC-BY-NC — neither may be shipped as a commercial-facing voice.
5. Reload: `registry.reload()` (Part 2 will expose this as an API call;
   until then, restart the process). The voice appears in
   `list_voices()` immediately; nothing else changes.

Removing a voice is the reverse: delete (or move) the folder and reload.
The core never needs an edit for either operation.

## 4. Failure modes (all non-fatal)

| Problem | Registry behaviour |
|---|---|
| `manifest.yaml` missing | Pack ignored, reason recorded: "manifest.yaml not found" |
| Missing required field | Rejected, reason names the field(s), e.g. "missing required field(s): sha256" |
| Bad `id` / `gender` / `language` / `engine` | Rejected, reason names the bad value and the allowed values |
| `model_path` / `config_path` file missing | Rejected, reason names the absolute path that was not found |
| `sha256` mismatch | Rejected, reason shows the first 12 chars of manifest vs file hash |
| Duplicate `id` | Second copy rejected, reason names the folder that won |
| `manifest.yaml` not valid YAML | Rejected, reason includes the parse error |

All rejections are visible in `registry.rejected` and in `registry.status()`
(for `/health` / diagnostics). A rejected pack never takes down the server
and never silently substitutes another voice.

## 5. The two shipped voices

| id | name | gender | engine | status in Part 1 |
|---|---|---|---|---|
| `telugu_srinivas` | Srinivas | male | `hima-nano` | Manifest template in `configs/voices/srinivas.yaml` — `sha256` / `licence` / `sample_rate` are **TBD** (Mac offline, pack unread). Fill them from `lite/models/hima-tts-nano-te/` before shipping. |
| `telugu_sravani` | Sravani | female | `hima-nano` | Same as above — `configs/voices/sravani.yaml`. |

Defaults are unchanged: default voice `telugu_srinivas`, style
`conversational` only. Any other style is a 400, not a silent mapping.

## 6. What a voice pack must NOT do

- Must not contain code that the core executes (no hooks, no scripts).
  A pack is data: manifest + model + config. An engine that needs code
  is a **provider** (`tts/providers/`), reviewed and shipped with the
  product — never smuggled in through a voice folder.
- Must not claim a licence it does not have. The manifest's `licence`
  is the operator's attestation; the core trusts it but does not verify
  it against the voice card. The operator must.
