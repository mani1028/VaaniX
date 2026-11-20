# Hosting

This server is one Python process. It has no database. Stopping it drops in-memory state. Generated WAV files under `experiments/tts/` are temporary; the process keeps the newest 32.

## Private machine

```bash
export TELUGU_SPEECH_AUTH=required
export TELUGU_API_KEYS='choose-a-long-random-key'
export TELUGU_HOST=127.0.0.1
export BHASHA_ENABLE_PLATFORM=0
telugu-api
```

`local_open` skips the key. Use it only on loopback while you are developing. Do not publish `local_open`.

## A host other people call

- Put TLS in front of the process. The app itself does not terminate TLS.
- Set `TELUGU_HOST` to the interface your proxy uses. The default is `127.0.0.1`.
- Keep `TELUGU_SPEECH_AUTH=required` and a long `TELUGU_API_KEYS` value. Several keys may be comma-separated. Rotate by editing the list and restarting.
- Set `TELUGU_CORS_ORIGINS` only to the browser origins that call the API. A `*` entry is ignored.
- Cap body size with the proxy as well. The app rejects uploads above `TELUGU_MAX_UPLOAD_BYTES` (default 20 MB, hard maximum 50 MB).
- Rate-limit at the proxy. This process does not have its own request quota. It accepts up to `TELUGU_LIMIT_CONCURRENCY` HTTP calls (default 256). Speech stays at one loaded engine until calls overlap, then grows to `TELUGU_SPEECH_SLOTS` (default 8, hard cap 16) and drops the extras when idle. Extra speech calls wait inside the process.
- Speech to text needs `models/telugu-asr-micro`. Text to speech needs `lite/models/hima-tts-nano-te`.

## Rollback

Stop the process and start the previous build. There is no migration. Removing `experiments/tts/*.wav` deletes generated audio only.

## What not to claim

Do not publish this as a measured Sarvam replacement. The ten-clip recognizer result is in the README. Voice quality against Sarvam was not measured here.
