# Build plan

The product is Telugu speech to text and Telugu text to speech. Both run from local files. No API key. Calling, agents, and other features wait until these two tasks are smaller and more accurate.

100% transcription accuracy is not a release claim. The local STT model on ten FLEURS test clips has word error 0.3669 and character error 0.2451. Text to speech has no weights on disk yet.

## What exists

- Phase 1 package, API, and tests.
- Python 3.11 environment `.venv311`. `.venv-test` is a link to it so the Phase 1 Python 3.15 environment stays in place.
- Ten-clip target: FLEURS `te_in` test audio, CC-BY-4.0, with published transcripts. These clips are evaluation only.
- faster-whisper `small` is the working teacher. IndicConformer, our student, our TTS, and Sarvam are not scored until each one actually runs.

## Test order

1. One Whisper size at a time: `small`, then `base`, then `tiny`. Do not load three models together.
2. WER, CER, and RTF on `data/manifests/eval.jsonl` with this repository's normalizer.
3. Saaras v4 on the same files only when `SARVAM_API_KEY` is set. Files longer than 30 seconds must use Sarvam's batch API. The current clips are under that limit.
4. TTS listening tests start after a consented single-speaker model exists. Bulbul v4 Flash uses the same Telugu sentences. Three listeners, 1–5 naturalness, is the gate. There is no TTS audio yet.
5. API checks stay on `/api/v1/asr/transcribe`, `/api/v1/tts/generate`, `/api/v1/health`, and `/`.

## Voice-agent path

A calling agent needs streaming partial transcripts, endpointing, barge-in, and a short Telugu reply. The current API transcribes a whole WAV and does not stream. That is the right first test. Streaming, turn-taking, and a telephony codec path come after the student ASR beats the Whisper teacher on the frozen Telugu set and a consented TTS voice exists.

Open weights to keep using as tools, not as the product name:

- faster-whisper for the noisy-speech teacher.
- AI4Bharat IndicConformer Telugu as the acoustic teacher, in the 3.11 environment, after the gated download.
- Silero VAD for endpointing experiments.
- A FastSpeech2 or Matcha plus HiFi-GAN training stack for the Telugu voice, after the IISc or IndicTTS terms are written into `docs/licenses.md`.

## Training stays closed until

- A train manifest exists that shares no speakers with `eval.jsonl`.
- The teacher checkpoint's real size is recorded.
- A GPU rental is tied to one resumable job.

## Commands

```bash
cd "/Users/visys/.cursor/tts sst"
source .venv-test/bin/activate
python scripts/test_asr.py data/raw/fleurs_te/fleurs_000.wav --model small
python scripts/evaluate_manifest.py --model small
python scripts/test_sarvam.py data/raw/fleurs_te/fleurs_000.wav
```

The Sarvam command prints `SARVAM_NOT_RUN` when the key is absent. It does not send audio.
