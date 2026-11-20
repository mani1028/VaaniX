# Bhasha Platform — open-source Sarvam-style stack

Full **Indic speech + text** platform built only from open models. No paid API required for local inference.

This is the **complete alternative track**. The small Telugu-only embed lives aside in [`../lite/`](../lite/).

## Sarvam feature map → open models

| Sarvam product | Role | Open alternative (default path) | License notes |
| --- | --- | --- | --- |
| Saaras v4 | Speech → text | `lite` Telugu micro STT now; IndicConformer / Indic-Transcribe when NeMo or HF weights are present | MIT / Apache |
| Bulbul v4 | Text → speech | Compact MMS Telugu (interim); **IndicF5** (MIT) when gated access + consented reference exist | MMS is **CC-BY-NC**; prefer IndicF5 for commercial |
| Mayura / Sarvam-Translate | Text translation | **IndicTrans2** distilled (`ai4bharat/indictrans2-*-dist-*`) | MIT |
| Instant voice clone | Voice adaptation | IndicF5 reference-audio path only with **explicit consent** | MIT + terms |
| Speech → speech | Cascade | STT → IndicTrans2 → TTS | Same as parts |

We do **not** copy Sarvam weights or call Sarvam. Optional Sarvam scoring stays outside this platform.

## Capabilities

1. **ASR** — Telugu offline today; multi-Indic via IndicConformer adapters when installed.  
2. **TTS** — Local Telugu today; IndicF5 multi-Indic when configured.  
3. **Translate** — IndicTrans2 distilled (lazy load).  
4. **Cascade** — `speech_to_speech`: audio → transcript → translation → speech.  
5. **API** — versioned `/platform/v1/*` next to the existing Telugu lab API.

## Quick start

```bash
cd "/Users/visys/.cursor/tts sst"
source .venv311/bin/activate

# Catalog (no downloads)
python -m bhasha_platform.catalog

# Local API with platform routes
TELUGU_SPEECH_AUTH=local_open BHASHA_ENABLE_PLATFORM=1 .venv311/bin/python -m bhasha_platform.api
```

Open http://127.0.0.1:8000/platform/ and http://127.0.0.1:8000/docs

## Install heavier open stacks when you need them

```bash
# Translation (IndicTrans2 distilled ~200–320M)
pip install sentencepiece

# IndicF5 TTS (gated HF; consent for reference voices)
# Accept model terms, then: pip install git+https://github.com/ai4bharat/IndicF5.git

# IndicConformer (NeMo fork — separate env recommended)
# git clone https://github.com/AI4Bharat/NeMo.git && cd NeMo && git checkout nemo-v2 && bash reinstall.sh
```

## Layout

```
platform/
  README.md
  configs/models.yaml      # open model registry
  docs/architecture.md
  apps/index.html          # platform dashboard
  src/bhasha_platform/     # Python package
```

Add to `PYTHONPATH` or install editable from the repo root (see root README).

## Honest status

| Feature | Status on this Mac |
| --- | --- |
| Telugu ASR (micro) | Runs offline (~42 MB) |
| Telugu TTS (micro) | Runs offline (~25 MB), CC-BY-NC |
| IndicTrans2 | Adapter ready; downloads on first use if online |
| IndicF5 / IndicConformer | Adapters ready; need install + weights |
| Match Sarvam quality | Not claimed until frozen multi-Indic eval |

Roadmap detail: [`docs/architecture.md`](docs/architecture.md) and [`../docs/world_class_offline_plan.md`](../docs/world_class_offline_plan.md).
