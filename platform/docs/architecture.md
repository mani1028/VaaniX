# Bhasha Platform architecture

## Goal

Open-weight alternative to Sarvam-style Indic speech products:

- Speech to text  
- Text to speech  
- Text translation  
- Speech-to-speech cascade  

Inference is local. Lite Telugu embed lives in `../lite/` and is not replaced by this platform.

## Data flow

```text
WAV ──► ASR (Telugu micro / IndicConformer)
          │
          ▼
        text ──► IndicTrans2 ──► text'
          │                         │
          └──────────► TTS ◄────────┘
                        │
                        ▼
                       WAV
```

## Components

| Layer | Package | Notes |
| --- | --- | --- |
| Lab STT/TTS | `src/telugu_speech` | Shared audio, metrics, micro models |
| Platform facades | `platform/src/bhasha_platform` | ASR / TTS / translate / cascade |
| Registry | `platform/configs/models.yaml` | Open model IDs and licenses |
| Lite aside | `lite/` | ≤100 MB Telugu-only copy |

## API

Mounted on the lab server when `BHASHA_ENABLE_PLATFORM=1` (default):

| Method | Path | Role |
| --- | --- | --- |
| GET | `/platform/` | Dashboard |
| GET | `/platform/v1/health` | Health |
| GET | `/platform/v1/catalog` | Open model catalog |
| POST | `/platform/v1/asr/transcribe` | STT |
| POST | `/platform/v1/tts/synthesize` | TTS |
| POST | `/platform/v1/translate` | IndicTrans2 |
| POST | `/platform/v1/speech-to-speech` | Cascade |

Legacy Telugu lab routes under `/api/v1/*` remain.

## Non-goals (for now)

- Calling / telephony agents  
- Training Sarvam-competitive students without a frozen eval  
- Shipping CC-BY-NC MMS as a commercial default (document and replace with IndicF5)  
