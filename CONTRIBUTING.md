# Contributing to Vaani X

Thank you for helping make Telugu speech self-hostable. This project wins by being small, honest, and easy to try — every contribution should make one of those three things stronger.

## Ground rules

- **No invented numbers.** A benchmark, a RAM figure, or a latency claim needs a command, a dataset, a machine, and a date. Unmeasured = TBD, written as TBD.
- **No "100% accurate" anywhere.** The measured ASR position is WER 0.2899 / CER 0.0811 on 10 clean clips (20 Nov 2026) — about 71% of words right. That is the baseline to beat, honestly.
- **No new dependencies without a reason.** Low-resource is the product. A new dependency must justify its size in the PR description.
- **No mock data in the product.** Tests may use fake engines and tmp voice packs; the server never returns fabricated audio or transcripts.
- **Keep the old API working.** `/api/v1/*` routes are a compatibility promise. Add, don't break.

## Setup

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -e ".[host,dev]"
.venv311/bin/python -m pytest -q
```

Tests need no model weights. Install the bench extra only for the jiwer cross-check: `pip install -e ".[bench]"`.

## How to add a voice pack (no code)

Voices are folders with a `manifest.yaml` — the USB-connector contract. Full contract: `docs/VOICE_PACK.md`.

1. Make the folder: `mkdir -p voices/telugu_<name>/` and put the model + config files in it.
2. Write `manifest.yaml` in that folder: `id`, `name`, `gender`, `language: te`, `engine` (`hima-nano` | `piper` | `onnx-vits`), `model_path`, `config_path`, `sample_rate`, `licence`, `licence_url`, `sha256`, `version`, optional `tags`.
3. Hash the model: `sha256sum voices/telugu_<name>/model.onnx` (macOS: `shasum -a 256 …`) and paste the 64-hex hash into `sha256`. A wrong hash is a rejection, not a warning.
4. Check the licence: read the voice card at `licence_url` and write the real licence into `licence`. Do not ship XTTS v2 (CPML, non-commercial) or MMS (CC-BY-NC) as a commercial-facing voice. Piper voices each carry their own licence.
5. Reload: `registry.reload()` (restart the process until the reload endpoint lands). The voice appears in `GET /v1/audio/voices` immediately.
6. Add a test: a tmp-path pack in `tests/test_voice_registry.py` covering accept + rejection for your pack shape. Never commit real model weights.

Removing a voice is the reverse: delete (or move) the folder and reload.

## How to add a backend (one provider, no core changes)

1. Add one module under `src/telugu_speech/tts/providers/` implementing the engine protocol (`synthesize(text, voice, style) -> bytes`, `synthesize_stream(...) -> Iterator[bytes]`, `close()`; ASR: `transcribe(audio_path) -> {text, language, segments, duration}`). See `docs/ARCHITECTURE.md` §5.
2. Add one line registering it in `core/engine.py::_BACKENDS`.
3. Core, registry, and API do not change. A voice pack must never contain code the core executes — a pack is data (manifest + model + config). An engine that needs code is a provider, reviewed and shipped with the product.
4. Add tests with a fake engine (see `tests/test_engine_pool.py`) and document the engine's RAM, RTF, and licence in the PR — measured, or labelled TBD with the command to measure it.

## How to run the bench

Datasets are NOT in the repo. Prepare a frozen manifest first; the command names the exact path when one is missing:

```bash
.venv311/bin/python -m telugu_speech.bench --asr-model telugu-asr-micro --sets fleurs-te kathbath-te cv-te --n 200
.venv311/bin/python -m telugu_speech.bench --asr-model telugu-asr-micro --sets fleurs-te --n 10 --allow-small  # smoke test only
```

Rules: frozen sets, speaker-disjoint (n ≥ 200 clips per set, ≥ 20 speakers), report WER + CER + AER together, raw and normalized, with RTF, peak RAM, and model revision SHA. Results go to `bench/results/<model>/<date>.json` and are never hand-edited. Full harness rules: `docs/benchmark_results.md`.

## PR gate

- `pytest -q` must pass (or the failure must be explained in the PR).
- **Any model swap that worsens Telugu WER by more than 2 points absolute on FLEURS-te fails the PR.** The CI gate is documented in `.github/workflows/ci.yml` and `docs/benchmark_results.md`; until CI is enabled, run the bench command above and paste the result into the PR.
- The PR template asks for: what changed, what was measured (or TBD), the verification commands you ran, and what you did not claim.

## Code style

- Python 3.11, standard library first. Keep functions small and errors OpenAI-shaped on `/v1/*` (`{"error": {"message", "type", "code"}}`).
- Telugu text handling lives in the core, not in a voice, so every voice benefits. Never drop user-typed words silently — code-mixed English is kept in Latin and passed through.
- Do not commit secrets, real API keys, model weights, `data/`, virtualenvs, or generated WAV files.

## Reporting bugs and asking for voices

Use the issue templates in `.github/ISSUE_TEMPLATE/`: `bug_report.md` (include server version, hardware, exact request, and the error shape — never paste a real key), `voice_request.md` (language, engine, licence, and a link to the voice card), `feature_request.md`.

## Code of conduct and security

By contributing you agree to [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). To report a vulnerability, follow [SECURITY.md](SECURITY.md) — do not open a public issue for it.
