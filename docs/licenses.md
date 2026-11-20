# Licenses

This repository's code is MIT. See `LICENSE`.

A model license is not the license of its training data. Before releasing weights or a hosted API, record the checkpoint, dataset cards, and any gated terms in the experiment log.

| Artifact | License as published | Use in this project |
| --- | --- | --- |
| IndicConformer Telugu hybrid CTC/RNN-T | MIT, gated download | ASR teacher and baseline |
| IndicConformer 600M multilingual | MIT, separate checkpoint | Do not treat it as the 120M Telugu model |
| `models/telugu-asr-micro` from `vasista22/whisper-telugu-tiny` | Apache-2.0 | Default compact STT (~42 MB int8). |
| ukta-app/whisper-telugu-ct2 / `models/telugu-asr` | Apache-2.0 conversion of `vasista22/whisper-telugu-small` | Optional larger STT (~241 MB). |
| facebook/mms-tts-tel / `models/telugu-tts` / `models/telugu-tts-micro` | CC-BY-NC-4.0 | Offline TTS (~139 MB full, ~25 MB int8-on-disk). Research/non-commercial until replaced. |
| IndicF5 | MIT, plus a voice-consent term on the model card | TTS quality reference. No cloning without permission |
| Indic-Speak | Indic Open Model License v1.0; Llama 3.2 community license also applies to the base | Listening reference only. Not the compact target. Hosted multi-tenant use needs the upstream terms |
| MMS ASR / MMS-TTS Telugu | Check each checkpoint. MMS-TTS is CC-BY-NC-4.0 | Research comparison only. Not shipped |
| Kathbath, Shrutilipi, IndicVoices-R | CC-BY-4.0 on the dataset cards reviewed for this plan | Training candidates after the card is re-read at download time |
| Common Voice Telugu | CC0 | Training candidate |

`configs/asr/finetune.yaml` and `configs/tts/fastpitch.yaml` do not start jobs.
