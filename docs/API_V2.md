# Vaani X API v2 — OpenAI-compatible drop-in

Vaani X v2 speaks the OpenAI audio API on `/v1/*`, backed by the local
Telugu engines. Existing code that calls OpenAI works with **only a
`base_url` change** — no new SDK, no new auth scheme, no cloud.

The old API (`/api/v1/*`) is unchanged and keeps working. Nothing in
v2 replaces it; v2 is an additional, compatible surface.

## Quick start

```bash
export TELUGU_SPEECH_AUTH=required
export TELUGU_API_KEYS='replace-me'
export TELUGU_HOST=127.0.0.1
export TELUGU_PORT=8000
telugu-api
```

Health needs no key:

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"ok","version":"0.2.0","auth_mode":"required",
#  "voices":["telugu_srinivas","telugu_sravani"],
#  "models":["vaani-tts-piper","vaani-tts-hima","vaani-tts","vaani-asr-small","vaani-asr"],
#  "slots":8,"asr_backend":"faster_whisper"}
```

## Drop-in: Python (OpenAI SDK)

Only `base_url` changes. Everything else is the same code you would
write against OpenAI.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="replace-me")

# Text to speech — wav always works locally; mp3 needs ffmpeg on the host.
with client.audio.speech.with_streaming_response.create(
    model="vaani-tts-hima",
    voice="srinivas",
    input="నమస్కారం",
    response_format="wav",
) as response:
    response.stream_to_file("speech.wav")

# Speech to text
with open("clip.wav", "rb") as audio:
    result = client.audio.transcriptions.create(
        model="vaani-asr-small",
        file=audio,
        language="te",
        response_format="json",
    )
print(result.text)
```

## Drop-in: JavaScript (OpenAI SDK)

```js
import OpenAI from "openai";
import fs from "node:fs";

const client = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "replace-me",
});

// Text to speech
const speech = await client.audio.speech.create({
  model: "vaani-tts-hima",
  voice: "srinivas",
  input: "నమస్కారం",
  response_format: "wav",
});
fs.writeFileSync("speech.wav", Buffer.from(await speech.arrayBuffer()));

// Speech to text
const transcription = await client.audio.transcriptions.create({
  model: "vaani-asr-small",
  file: fs.createReadStream("clip.wav"),
  language: "te",
  response_format: "json",
});
console.log(transcription.text);
```

## Endpoints

| Method | Path | Auth | What it does |
| --- | --- | --- | --- |
| `GET` | `/health` | none | `ok` + voices, models, slots, ASR backend. |
| `GET` | `/v1/models` | key | `{"object":"list","data":[{"id","object":"model","created","owned_by":"vaani"}]}` |
| `GET` | `/v1/audio/voices` | key | Voice list from the registry (or the installed pack): `id, name, gender, language, engine, sample_rate, licence`. |
| `POST` | `/v1/audio/speech` | key | TTS. JSON body, audio bytes back. |
| `POST` | `/v1/audio/transcriptions` | key | ASR. Multipart file upload. |

Auth accepts **either** `Authorization: Bearer <key>` **or**
`X-API-Key: <key>`, using the same keys from `TELUGU_API_KEYS`.
`local_open` mode skips the key on loopback only, as before.

### POST /v1/audio/speech

JSON body:

```json
{
  "model": "vaani-tts-hima",
  "input": "నమస్కారం",
  "voice": "srinivas",
  "response_format": "wav",
  "speed": 1.0,
  "stream": false
}
```

| Field | Values | Notes |
| --- | --- | --- |
| `model` | `vaani-tts-piper`, `vaani-tts-hima`, `vaani-tts` | All three are aliases for the one installed local engine today (see below). |
| `input` | Telugu text, 1–4096 chars | Longer than 4096 → 400. The engine chunks long text into sentences server-side. Must include Telugu letters. |
| `voice` | `srinivas`, `sravani`, `telugu_srinivas`, `telugu_sravani` | Aliases for the installed voices. Unknown voice → 400. |
| `response_format` | `mp3`, `wav`, `opus`, `flac` | `wav` always works. The others need `ffmpeg` on the host (400 otherwise). |
| `speed` | 0.25–4.0 | Accepted for compatibility. **Currently a no-op** — the local engine speaks at its natural rate. |
| `stream` | `true` / `false` | `true` → SSE sentence stream, WAV only (see below). |

Response: audio bytes with `Content-Type: audio/wav` (or
`audio/mpeg` / `audio/opus` / `audio/flac`) and an `X-Sample-Rate`
header (24000 for the installed voices).

```bash
curl -s -H "Authorization: Bearer $TELUGU_API_KEYS" \
  -H "Content-Type: application/json" \
  -d '{"model":"vaani-tts-hima","voice":"srinivas","input":"నమస్కారం","response_format":"wav"}' \
  http://127.0.0.1:8000/v1/audio/speech -o speech.wav
```

### Streaming TTS (SSE, Vaani extension)

OpenAI's speech endpoint has no SSE standard, so this is Vaani's own
streaming shape — not an OpenAI behaviour. Send `"stream": true`
with `"response_format": "wav"` and read `text/event-stream`: one
event per sentence, each carrying base64 WAV, then `data: [DONE]`.

```
data: {"audio": "<base64 wav>", "index": 0, "text": "నమస్కారం.", "voice": "telugu_srinivas", "format": "wav", "sample_rate": 24000}

data: [DONE]
```

```bash
curl -s -N -H "Authorization: Bearer $TELUGU_API_KEYS" \
  -H "Content-Type: application/json" \
  -d '{"model":"vaani-tts-hima","voice":"sravani","input":"నమస్కారం. ఎలా ఉన్నారు?","response_format":"wav","stream":true}' \
  http://127.0.0.1:8000/v1/audio/speech
```

Streaming with `mp3`/`opus`/`flac` returns 400 — streaming is WAV
only. The old routes are unchanged: `POST /api/v1/tts/stream` still
sends 16-bit PCM per sentence, and the `/api/v1/stt/stream`
websocket still answers a whole chunk with `streaming: false`
(batch decode, not live streaming).

### POST /v1/audio/transcriptions

Multipart form, mirroring OpenAI:

| Field | Values | Notes |
| --- | --- | --- |
| `file` | audio upload | 25 MB cap (413 beyond). See formats below. |
| `model` | `vaani-asr-small`, `vaani-asr` | Both map to the local recognizer. |
| `language` | `te` | Only Telugu. Anything else → 400. |
| `response_format` | `json`, `verbose_json`, `srt`, `vtt`, `text` | Default `json`. |
| `temperature` | 0.0–1.0 | Accepted for compatibility. **Currently a no-op** — the local decoder always runs at temperature 0.0. |

```bash
curl -s -H "Authorization: Bearer $TELUGU_API_KEYS" \
  -F "file=@clip.wav;type=audio/wav" \
  -F "model=vaani-asr-small" -F "language=te" -F "response_format=verbose_json" \
  http://127.0.0.1:8000/v1/audio/transcriptions
```

`json` returns `{"text": "..."}`. `text` returns the transcript as
plain text. `verbose_json` returns:

```json
{
  "task": "transcribe",
  "language": "te",
  "duration": 12.4,
  "text": "...",
  "words": [{"word": "...", "start": 0.0, "end": 0.5}],
  "segments": [{"id": 0, "seek": 0, "start": 0.0, "end": 2.1, "text": "...",
                "tokens": [], "temperature": 0.0, "avg_logprob": 0.0,
                "compression_ratio": 1.0, "no_speech_prob": 0.0}]
}
```

Segment and word timestamps come from faster-whisper when it is the
active ASR backend. With any other ASR backend the response carries
one segment covering the whole clip, with duration read from the WAV
header. ASR is VAD-segmented batch decode — there is no streaming
transcription endpoint and none is faked.

Accepted upload extensions: `wav, mp3, m4a, mp4, mpeg, mpga, webm,
flac, ogg`. WAV always works. Other formats are decoded by
faster-whisper; with a non-faster-whisper ASR backend they return
400 and only WAV is accepted.

## Errors (OpenAI shape)

All `/v1/*` errors look like OpenAI's:

```json
{"error": {"message": "Incorrect API key provided.", "type": "authentication_error", "code": "invalid_api_key"}}
```

| Status | `code` | Meaning |
| --- | --- | --- |
| 400 | `invalid_request` | Bad field, bad format, bad temperature, non-Telugu text, non-streaming format misuse. |
| 400 | `invalid_text` | TTS input has no Telugu letters. |
| 400 | `invalid_voice` | Unknown voice name. |
| 400 | `invalid_audio` | Empty, unreadable, or wrong-extension upload. |
| 400 | `unsupported_language` | `language` is not `te`. |
| 400 | `unsupported_format` | mp3/opus/flac requested without `ffmpeg` on the host. |
| 401 | `invalid_api_key` | Missing or wrong key. |
| 404 | `model_not_found` | Unknown `model`. |
| 413 | `file_too_large` | Upload over 25 MB. |
| 429 | `rate_limit_exceeded` | Over 600 requests/minute for one key. Includes a `Retry-After` header. |
| 500 | `transcode_failed` | ffmpeg could not convert the audio. |
| 503 | `model_unavailable` | Voice or ASR weights are not on disk. |
| 503 | `api_keys_not_configured` | Server has no `TELUGU_API_KEYS` set. |

Validation failures on `/v1/*` (missing field, text over 4096 chars,
speed out of range) also return this shape with status 400.
`/api/v1/*` keeps its existing error shape and 422 behaviour.

## Rate limit

In-memory, per key: **600 requests per minute** on the `/v1/audio/*`
and `/v1/models` routes. Over the limit returns 429 with a
`Retry-After` header. `/health` and `/v1/audio/voices` are not
rate-limited. This limiter does not survive a restart and does not
share state across processes — production should quota at the proxy,
as the README already requires for TLS.

## What the model names mean (read this before claiming more)

| API model | What runs today |
| --- | --- |
| `vaani-tts-piper` | Alias for the installed Hima ONNX int8 engine. **Not Piper.** Piper is a future tier. |
| `vaani-tts-hima` | The installed Hima ONNX int8 engine (`lite/models/hima-tts-nano-te`). |
| `vaani-tts` | Same engine, short alias. |
| `vaani-asr-small` | The local recognizer (`models/telugu-asr-micro`, faster-whisper int8 tiny). |
| `vaani-asr` | Same recognizer, short alias. |

When Part 1's voice registry lands, `GET /v1/audio/voices` reads it;
until then it reads the installed voice pack. Either way the server
starts — the registry is optional, the pack is the fallback.

No latency or accuracy numbers are claimed here. The measured numbers
live in `docs/benchmark_results.md` (10 clean clips, WER 0.2899 /
CER 0.0811 with the current decoder, 20 Nov 2026).

## Old API (unchanged)

`GET /api/v1/health`, `POST /api/v1/tts/generate`,
`POST /api/v1/asr/transcribe`, `POST /api/v1/tts/stream`,
`POST /api/v1/stt`, the `/api/v1/stt/stream` websocket, and
`GET /api/v1/models` all behave exactly as before. The v2 routes are
additive. The only shared change is that request-validation failures
on `/v1/*` paths now render in the OpenAI error shape instead of the
old 422 shape; `/api/v1/*` validation is untouched.
