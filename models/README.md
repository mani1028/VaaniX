# Local model packages

| Folder | Role | Size | Notes |
| --- | --- | ---: | --- |
| `telugu-asr-micro/` | **Default STT** | ~42 MB | Telugu Whisper tiny int8 |
| `telugu-tts-micro/` | **Default TTS** | ~25 MB | MMS Telugu int8-on-disk; CC-BY-NC-4.0 |
| `telugu-asr/` | Optional STT | ~241 MB | More accurate Telugu small |
| `telugu-tts/` | TTS source | ~139 MB | Used only to build the micro TTS |

Micro total ≈ **67 MB** (under 100 MB). Rebuild with `scripts/compress_models.py`.
