# Phase status

Measured on this Mac on 20 November 2026: Apple M5, 16 GB RAM, no NVIDIA GPU. Python 3.15 cannot install faster-whisper. The model run used `.venv311` (Python 3.11.15).

FLEURS `te_in` test clips are evaluation audio only. They were not used for training.

## Measured ASR

| Model | Clips | WER | CER | RTF | Model bytes | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| faster-whisper small, beam 5, temperature 0, language te | 6 FLEURS te_in test | 1.3580 | 1.1103 | 0.9921 | 972425620 cache bytes | 1026015232 |
| IndicConformer Telugu 120M | 0 | not measured | not measured | not measured | not downloaded | NeMo is not installed |
| Sarvam Saaras | 0 | not measured | not measured | not measured | hosted API, no local file | no API key in this run |

WER and CER use this repository's normalizer. A rate above 1 means the hypothesis has more edit operations than reference words. That is a measured failure, not a published score.

## Phase outcomes

| Phase | Outcome |
| --- | --- |
| 1.5 Real model integration | faster-whisper small ran on six licensed FLEURS Telugu test clips. IndicConformer did not run: `import nemo` raises `ModuleNotFoundError`. IndicF5 was not downloaded: the checkpoint is gated and this machine has no Hugging Face token. |
| 2 Dataset preparation | Manifest writer, speaker-disjoint splits, and `scripts/fetch_fleurs_sample.py` work. The local manifest is test-only. Kathbath, Shrutilipi, and Common Voice training audio are not on disk. |
| 3 Fine-tune IndicConformer | Blocked. The fine-tune config is valid and `train()` still refuses to launch a NeMo job. There is no teacher checkpoint and no training audio. |
| 4 Adapt Telugu TTS | Blocked. IndicF5 needs transformers, a gated download, and a consented reference recording. No TTS audio was generated. |
| 5 TeluguTinyASR | Not trained. Training on the six FLEURS test clips would leak the evaluation set. |
| 6 TeluguTinyTTS | Not trained. No consented single-speaker corpus is present. |
| 7 Offline and hosted deployment | The local API calls the same `FasterWhisperBackend.transcribe` function as the benchmark. Hosted deployment is specified in `docs/deployment.md` and is not live. |

## Telugu AI Engine surface

Additive routes on the same process: `POST /api/v1/detect`, `POST /api/v1/stt`, `WS /api/v1/stt/stream` (`streaming: false`), `GET /api/v1/tts/voices`, `POST /api/v1/tts/voices/preview`, `POST /api/v1/tts/voices/custom`. `/api/v1/asr/transcribe` is unchanged. Training still raises `TrainingDeferred`. Detection F1, IndicConformer, IndicF5 naturalness, and Sarvam are not measured. Install notes: `docs/teacher_install.md`.

## What a later run still needs

- A Python 3.11 environment, which is `.venv311` on this machine.
- Hugging Face access for `ai4bharat/indicconformer_stt_te_hybrid_ctc_rnnt_large` and `ai4bharat/IndicF5`.
- A NeMo build that imports on macOS, or a Linux GPU machine.
- Train splits from Kathbath, Shrutilipi, or Common Voice, kept separate from this FLEURS test manifest.
- A Sarvam API key before any Saaras row is added.
