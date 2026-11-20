# Opt-in teacher install

This is not part of `pytest`. Do not put these commands in the default test environment (Python 3.15 `.venv`). Speech experiments stay on Python 3.11 (`.venv311`). NeMo gets its own environment.

The 16 GB Mac can load the Telugu Whisper micro/small models and the 120M IndicConformer teacher if the checkpoint and NeMo import succeed. The 600M multilingual checkpoint is a GPU/cloud teacher. Do not make it the default local model.

## ASR teacher: IndicConformer Telugu 120M

Checkpoint: `ai4bharat/indicconformer_stt_te_hybrid_ctc_rnnt_large` (MIT, gated). Decoder comparison: CTC, then RNNT. Config: `configs/asr/indicconformer.yaml`.

```bash
python3.11 -m venv .venv-nemo
source .venv-nemo/bin/activate
pip install torch
git clone https://github.com/AI4Bharat/NeMo.git
cd NeMo && git checkout nemo-v2 && bash reinstall.sh
```

Accept the Hugging Face gate, then from the repo root with that environment:

```bash
export HF_TOKEN=...   # do not commit this
python -m telugu_speech.cli run --config configs/evaluation.yaml
```

`telugu-bench` writes `experiments/asr/baseline_report.md`. If NeMo or the checkpoint is missing, the IndicConformer rows say **not measured**. Copy only numbers that the command prints into `docs/benchmark_results.md`.

The 600M model id, for a GPU machine later, is `ai4bharat/indic-conformer-600m-multilingual`. Score it on the same `data/manifests/eval.jsonl` before any fine-tune. `train()` still refuses, and it rejects train rows whose `utt_id` appears in that eval file.

## TTS teacher: IndicF5

Checkpoint: `ai4bharat/IndicF5` (MIT, gated). It needs a reference clip and the reference transcript, plus consent.

```bash
source .venv311/bin/activate
pip install transformers
# After the HF gate is accepted and a consented reference is on disk:
# TELUGU_TTS_BACKEND=indicf5 is not enough by itself. The API custom-voice
# route returns model_unavailable until IndicF5 actually loads.
```

Do not publish a MOS number until listeners, or an agreed objective proxy, have been run. Male and female voices come from consented Rasa (or other licensed) references after `docs/dataset_cards.md` is checked. The installed interim voice, when `models/telugu-tts-micro` is present, is one MMS voice under CC-BY-NC-4.0.

## Sarvam comparison

```bash
export SARVAM_API_KEY=...   # do not commit this
python scripts/test_sarvam.py data/raw/fleurs_te/fleurs_000.wav --model saaras:v4
```

Without the key the script prints `not measured` for `transcribe` and `codemix` and sends nothing. API elapsed time is not local RTF.

## Still blocked

Fine-tuning, distillation, and INT8 of an unmeasured teacher stay blocked. ONNX export remains a fallback until the locked manifest shows numeric parity with the PyTorch or CTranslate2 runtime.
