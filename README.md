# Telugu Speech API

One Python process that your services call for Telugu speech. It transcribes a WAV file and speaks Telugu text. You host it. Callers share that one API.

The server code is MIT. Recognition weights are about 42 MB. The voice pack is about 150 MB on disk. Weights are not inside the PyPI package. Python 3.11 is the runtime that runs speech.

This is a low-memory CPU server. It is not a measured match for Sarvam. It does not translate, and it does not speak languages other than Telugu.

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

## Run one API

```bash
export TELUGU_SPEECH_AUTH=required
export TELUGU_API_KEYS='choose-a-long-random-key'
export TELUGU_HOST=127.0.0.1
export TELUGU_PORT=8000
export BHASHA_ENABLE_PLATFORM=0
telugu-api
```

| URL | Use |
| --- | --- |
| `GET /api/v1/health` | No key. Returns `ok`. |
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

One process. Idle speech keeps **one** engine. Extra engines are created only while calls overlap, then dropped. The ceiling is `TELUGU_SPEECH_SLOTS` (default **8**, hard cap **16**). HTTP accepts up to **256** calls (`TELUGU_LIMIT_CONCURRENCY`). Calls past the speech ceiling wait. They do not each load a new model.

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
| `TELUGU_SPEECH_SLOTS` | `8` | Max overlapping speech engines. Cap 16. |
| `TELUGU_CPU_THREADS` | `2` | Threads per engine. Cap 4. |
| `TELUGU_LIMIT_CONCURRENCY` | `256` | HTTP calls in flight. |
| `TELUGU_MAX_UPLOAD_BYTES` | `20000000` | Upload cap. Hard max 50 MB. |
| `TELUGU_CORS_ORIGINS` | empty | Exact browser origins. |
| `BHASHA_ENABLE_PLATFORM` | `1` in code | Set `0` for this speech API only. |

## Repository map

| Path | Keep for |
| --- | --- |
| `src/telugu_speech` | The API and engines. This is the PyPI package. |
| `index.html` | The page served at `/`. |
| `tests` | `pytest`. |
| `docs/deployment.md` | Host, TLS, rollback. |
| `docs/benchmark_results.md` | Measured error rates and the memory runs above. |
| `lite/models/hima-tts-nano-te` | Voice weights. License on that folder's card is `other`. |
| `platform/` | Optional catalog. Not required to host this API. |

`data/`, virtualenvs, checkpoints, and generated WAV files stay off git. Training scripts under `scripts/` and notes under `docs/` are for this lab, not for callers of the API.

## Publish

Build a wheel from a clean tree. The wheel is code. Ship weights beside the deployed process, not inside the wheel.

```bash
pip install build
python -m build
```

Upload `dist/*` with your PyPI account when you are ready. This repository does not publish for you.

## Checks

```bash
.venv311/bin/python -m pytest
```

## License

Server code is MIT (`LICENSE`). Recognizer weights are Apache-2.0. The Hima voice card license is `other`; read `lite/models/hima-tts-nano-te/README.md` before you redistribute those weights. An older MMS pack, if present under `models/telugu-tts-micro`, is CC-BY-NC-4.0 and is not a hosted voice.

## Limits of this product

- Telugu (`te`) only.
- No translation.
- No measured Sarvam parity. Ten clean clips on the tiny recognizer scored WER 0.2899 and CER 0.0811 with the current decoder (20 Nov 2026). English words inside Telugu are often missed.
- The websocket route answers a whole chunk and sets `streaming` to false.
- Voice naturalness against other vendors was not measured.
