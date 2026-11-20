# Offline Telugu STT and TTS plan

**Goal:** The most accurate Telugu speech-to-text and text-to-speech that runs fully offline, with a small enough footprint for phones later.

**Hard limits:** No API keys at inference. No Sarvam for production. No calling-agent features until STT and TTS meet the gates below. Do not claim 100% accuracy. Word error 0 is a target to approach, not a release claim.

## Current baseline (measured)

| Piece | State |
| --- | --- |
| STT | Local int8 Telugu Whisper in `models/telugu-asr`, ~241 MB. On 10 FLEURS `te_in` test clips: WER 0.3669, CER 0.2451, RTF ~0.77–1.1, peak RAM ~956 MB |
| TTS | Not installed |
| Auth | Local files only. API can run with `TELUGU_SPEECH_AUTH=local_open` on 127.0.0.1 |
| Product scope | STT and TTS only |

This baseline is usable offline. It is not world-class yet. Multilingual Whisper small scored WER 1.27 on the same clips and is worse.

## What “world best” means here

Beat every open Telugu system on a **frozen, speaker-disjoint Telugu benchmark**, and beat Sarvam Saaras v4 / Bulbul v4 Flash on that same set when scored later as an external reference. Report separately:

- Clean read speech
- Conversational and regional (Telangana, Coastal Andhra, Rayalaseema)
- Telugu–English code-mix
- Phone 8 kHz and noise
- Size, peak RAM, and RTF on M5 CPU and on a mid-range Android later

One pooled WER is not enough.

## Architecture (STT + TTS only)

```text
Audio WAV  --> VAD --> Telugu STT (local) --> text
Telugu text --> normalize --> acoustic + vocoder (local) --> WAV
```

Shared: 16 kHz mono prep, Telugu normalizer (raw vs norm text), manifests, WER/CER/RTF/size reports, FastAPI that calls the same functions as the CLI.

## Roadmap

### Phase A — Freeze the test (1 week)

1. Keep the current 10 FLEURS clips as a smoke set. Do not train on them.
2. Grow eval to 100–200 licensed Telugu clips with verified transcripts (FLEURS holdout, Common Voice Telugu test, later own consent recordings).
3. Speaker-disjoint splits only. Manifest fields already exist: `utt_id`, `audio`, `raw_text`, `norm_text`, `speaker_id`, `region`, `condition`.
4. Publish the eval protocol so every later model is scored the same way.

**Exit:** Fixed `data/manifests/eval.jsonl` (and a larger private holdout) that never changes during training.

### Phase B — Stronger offline STT teacher (1–2 weeks)

1. Prefer Telugu-specialized open weights over multilingual Whisper. Current: `models/telugu-asr` (Apache-2.0 conversion of `vasista22/whisper-telugu-small`).
2. Add IndicConformer Telugu 120M in a Python 3.11 environment when NeMo installs. Score on the same eval. Keep as teacher, not the phone model.
3. Tune decoding for accuracy first (beam, VAD, no previous-text conditioning), then cut threads for battery.

**Exit:** Teacher WER clearly below 0.3669 on the frozen eval, still offline.

### Phase C — Own student STT (weeks 3–8)

1. Training data: Kathbath Telugu, Common Voice Telugu, OpenSLR SLR66, and later consented conversational speech. Record every license in `docs/licenses.md`.
2. Architecture: compact Conformer-CTC, start ~12 layers / width 256 / 4 heads (validate; do not lock until measured).
3. Train with ground-truth CTC loss, then distill from the best teacher. Student must run without the teacher.
4. Compress: INT8 first, then prune only if accuracy recovers. Target package **25–60 MB**, peak RAM under **300 MB** before claiming mobile readiness.
5. Distillation objective (tune λ experimentally): CTC + KD + augmentation consistency.

**Exit gates for STT release:**

| Gate | Condition |
| --- | --- |
| G1 | Offline only; no key |
| G2 | WER within 10% relative of best teacher on frozen eval |
| G3 | INT8 loss ≤ 5% relative WER vs FP16 student |
| G4 | RTF &lt; 1 on the declared CPU |
| G5 | Peak RAM within declared budget (phone tier separate) |
| G6 | Slice tables for region, noise, phone, code-mix |

### Phase D — Offline TTS (after STT G2)

1. One consented speaker first (IISc SYSPIN or IndicTTS after terms are recorded).
2. FastSpeech2 or Matcha-TTS + HiFi-GAN (or Vocos). Package target 20–60M / compact INT8 package.
3. Evaluate with native listeners (MOS 1–5), pronunciation checks, RTF, and size. Do not use ASR-only as TTS quality.
4. Same local API: `POST /api/v1/tts/generate` with a fixed consented reference or embedded single voice.

**Exit:** Playable Telugu WAV offline; listening scores recorded; no cloning of strangers.

### Phase E — Mobile and “any device”

1. Desktop/laptop CPU first (current path: CTranslate2 int8).
2. Export student to ONNX / ExecuTorch / ONNX Runtime Mobile only after G2–G5 pass.
3. Phone claim only after measured size, RAM, and RTF on a mid-range Android. The current 241 MB / ~1 GB RAM model is **not** phone-ready.
4. Keep one inference library; wrap for Android/iOS later. Do not fork logic.

## What we will not do now

- Streaming voice agents, barge-in, telephony codecs
- Multi-speaker cloning without consent
- Claiming Sarvam parity without scoring Saaras on the frozen set
- Training on the FLEURS eval clips
- Weakening WER/CER gates to look accurate

## Immediate next actions (this repo)

1. Keep `models/telugu-asr` as the offline default.
2. Expand eval to ≥100 licensed clips without touching training.
3. Install IndicConformer in `.venv311` when feasible; score vs the Telugu Whisper baseline.
4. Prepare Kathbath / Common Voice Telugu train manifests (speaker-disjoint from eval).
5. Only then rent a GPU for one resumable student training job.

## Success definition

Done when we ship **our own** Telugu STT weights and **our own** Telugu TTS weights that:

- run offline with no keys,
- beat open Telugu baselines on the frozen benchmark,
- meet size/RAM/RTF budgets for the declared device tier,
- and, only then, are compared to Sarvam on identical audio and text.
