# Vaani X Core Engine v2 — Architecture (Part 1: CORE + USB-style voice plugins)

> Status: Part 1 built 2026-11-21 and deployed to the Mac the same day.
> The Mac went offline mid-task (~15:32–15:35 UTC) and recovered; the
> Part 1 files below were then read from the Mac (`api/main.py`,
> `tts/voices.py`, `tts/infer.py`, `tts/runtime.py`, `tts/model.py`,
> `text/normalization.py`, `text/unicode.py`, `text/g2p.py`,
> `text/tokenizer.py`, `asr/infer.py`, `docs/benchmark_results.md`,
> lite pack README) and the code was corrected against them before
> deployment — notably `backend_for_voice()` returns a backend object,
> number expansion delegates to the existing `normalization.py`
> (lakh/crore, ordinals, decimals), chunks are capped at 180 chars
> (Hima's `speech_pieces` limit), and the ASR backends return `str`.
> Anything marked **TBD** below is unmeasured — do not publish it as a
> fact until the command next to it is run.

## 1. The idea in one paragraph

Voices are USB connectors; the core is the port. A voice is a **folder
with a `manifest.yaml`** — nothing else. Plugging a new voice in means
dropping its folder into `voices/` and reloading. The core never imports
a voice engine at startup, never hard-codes a voice, and never loads a
model per request. Telugu text handling lives in the core, not in a
voice, so every voice benefits from it.

## 2. What the #1 claim is (and is not)

**Winnable claim:** smallest, cheapest, genuinely self-hostable Telugu
voice API that is OpenAI-compatible. **Not** "best Telugu speech,
period" and **not** "100% accurate". Measured ASR on the current product:
WER 0.2899 / CER 0.0811 on 10 clean clips (~71% of words exactly right).
That is a prototype result, not a benchmark — see §7.

## 3. Layout (Part 1 scope only)

```
src/telugu_speech/
├── core/                     # NEW — this part. No model, no FastAPI, no voice engine.
│   ├── __init__.py           # public surface only
│   ├── registry.py           # VoicePack + VoiceRegistry: discovery, validation, reload
│   ├── engine.py             # EnginePool: lazy load, slots, LRU cache, idle unload
│   ├── text.py               # Telugu-first pipeline: NFC, split, expand, chunk
│   ├── errors.py             # VaaniError family, OpenAI-shaped
│   └── config.py             # Settings from env vars, safe low-resource defaults
├── tts/
│   ├── voices.py             # REFACTORED: delegates to registry; backend_for_voice() kept
│   └── providers/            # NEW — one module per engine, same interface
│       ├── base.py           # BaseTTSBackend: synthesize / synthesize_stream / close
│       ├── hima_nano.py      # adapter over legacy tts.infer / tts.runtime (shipped default)
│       ├── piper.py          # Piper ONNX VITS (recommended upgrade path)
│       └── onnx_vits.py      # generic VITS ONNX packs
├── api/                      # NOT touched in Part 1 (other agent owns routes)
├── asr/                      # NOT touched in Part 1 (engine resolves it lazily by name)
└── text/                     # NOT touched in Part 1 (core/text.py is the new pipeline;
                              #  migration of callers to it is a later part)
configs/voices/               # NEW — source manifests: srinivas.yaml, sravani.yaml
docs/ARCHITECTURE.md          # this file
docs/VOICE_PACK.md            # the manifest contract + how to add a voice
```

**Dependency direction (strict, one-way):**

```
API  ->  core (registry, engine pool, text, errors, config)
core ->  tts.providers.*   (lazy, by engine name, on first use only)
core -/->  any specific voice engine at import time
tts.providers.hima_nano -> legacy tts.infer / tts.runtime (lazy, in __init__)
```

If you draw an arrow from `core/` to a model file, a voice, or FastAPI,
the design is broken.

## 4. Voice registry — how a voice is found and trusted

See `docs/VOICE_PACK.md` for the full contract. Summary:

- Search roots, in order: `VAANI_VOICES_DIR` (colon-separated) →
  `<project>/voices/` → `<project>/lite/models/`.
- A pack is a folder with `manifest.yaml`. Required fields: `id, name,
  gender, language, engine, model_path, config_path, sample_rate,
  licence, licence_url, sha256, version`; optional `tags`.
- Validation checks: field presence, id format, gender/language/engine
  enums, `sample_rate` int > 0, sha256 format, **both files exist**,
  **SHA-256 of the model file matches the manifest**.
- A bad pack is recorded in `registry.rejected` with the exact reason
  and the server keeps running. A duplicate id is rejected (first match
  wins, and the shadowing is logged).
- `registry.reload()` re-scans without a restart. The API reload
  endpoint is Part 2 work, not Part 1.

## 5. Engine pool — why it stays small

| Rule | Why |
|---|---|
| Lazy load on first use | Server starts fast; an unused voice costs 0 MB |
| ONE TTS engine per loaded voice | Re-loading a 260–470 MB engine per request would OOM the box |
| ONE shared ASR instance | ASR is ~42 MB (FasterWhisper tiny INT8, per brief) — one copy is enough |
| Slot semaphore, default **2** | **TBD / brief-supplied sizing:** 1 engine ~260–470 MB, 2 engines ~640–720 MB, so 8 concurrent engines would need ~1.5–1.8 GB. The old default of 8 was dangerous on a 1–2 GB VPS. Measure peak RSS before raising it (see §7) |
| Threads pinned, default 2, cap 4 | `TELUGU_CPU_THREADS` → `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `ONNX_NUM_THREADS`, set **before** the first ONNX session is created |
| Idle unload after `VAANI_VOICE_IDLE_SECS` (default 1800 s) | A voice nobody uses frees its RAM without a restart |
| Bounded LRU audio cache | `VAANI_AUDIO_CACHE_ENTRIES` (128) + `VAANI_AUDIO_CACHE_MAX_MB` (256); key is `(voice, style, sha256(text)[:16])` |

### Engine protocol (the contract providers implement)

```python
class TTSBackend(Protocol):
    def synthesize(self, text: str, voice: VoicePack, style: str = "conversational") -> bytes: ...
    def synthesize_stream(self, text: str, voice: VoicePack, style: str = "conversational") -> Iterator[bytes]: ...
    def close(self) -> None: ...

class ASRBackend(Protocol):
    def transcribe(self, audio_path: str | Path) -> dict: ...
    # returns {"text": ..., "language": "te", "segments": [...], "duration": ...}
```

Adding an engine = one provider module + one line in
`core/engine.py::_BACKENDS`. Core, registry and API do not change.

## 6. Telugu-first text pipeline (`core/text.py`)

Order: NFC normalise → validate (empty / > `VAANI_MAX_TEXT_CHARS`,
default 4096) → sentence split (Telugu danda `।` + Latin `. ! ?`) →
number/date/currency expansion in Telugu words → chunking (default
≤500 chars, always on a sentence boundary).

- Numbers: `21` → `ఇరవై ఒకటి`, `2026` → `రెండు వేలు ఇరవై ఆరు`;
  dates `21/11/2026` → `ఇరవై ఒకటి నవంబర్ రెండు వేలు ఇరవై ఆరు`;
  `₹1,250` → `ఒకటి వేలు రెండు వందల యాభై రూపాయలు`; times `14:30` →
  `పద్నాలుగు గంటల ముప్పై నిమిషాలు`.
- **Code-mixed English is kept in Latin and passed through.** The old
  product dropped English words; dropping user-typed words is data loss.
- Errors are `TextValidationError` (HTTP 400, OpenAI-shaped
  `{"error": {"message", "type", "code"}}`), never a bare traceback.

## 7. Measured numbers (read from the Mac 2026-11-21) and what is still TBD

Measured — from `docs/benchmark_results.md` (20 Nov 2026, Apple M5, 16 GB,
FLEURS `te_in` test, 10 clips, faster-whisper small, beam 5, temp 0):

| Model | WER | CER | RTF | Model bytes | Peak RSS |
|---|---|---|---|---|---|
| `models/telugu-asr-micro` (tiny int8) | 0.4379 | 0.2760 | 0.1095 | 44,255,137 | 361,594,880 (~345 MB) |
| `models/telugu-asr` (small int8) | 0.3669 | 0.2451 | 0.7738 | 253,164,606 | 956,170,240 (~912 MB) |
| faster-whisper multilingual small | 1.2722 | 0.9797 | 0.5785 | 972,425,620 | 1,051,033,600 |
| Live recogniser, later the same day (VAD, logprob filter) | **0.2899** | **0.0811** | 0.18 | — | — |

Code-mix is a measured gap, not a suspicion: on 16 Andhra prompted
clips, English-word recall was **0 of 53** — the recogniser drops Latin
code-mixed words entirely (mixed WER 0.5458, Telugu-only WER 0.4394).
That is why `core/text.py` keeps Latin words and why the ASR upgrade
ladder (vasista22-small, IndicConformer) matters.

Hima pack (from `lite/models/hima-tts-nano-te/README.md`, pack's own
measurements, Apple M4 Pro, 48 sentences): int8 is **15.0× real-time**
at 4 threads, 10.5× at 1 thread; model files for one voice set 84 MB
int8 vs 273 MB fp32; letters-wrong 3.39% int8 vs 3.15% fp32 (Sarvam
saaras v4, built-in voices, 1,202 sentences).

Slots: `TELUGU_SPEECH_SLOTS=2 TELUGU_CPU_THREADS=2` is already the
bench suite's own convention (`scripts/run_benchmark_suite.py`) — the
core default of 2 aligns with it; the legacy `runtime.py` default of 8
remains for the old path until the API migrates to the pool (Part 2+).

| Still TBD | Command to measure it |
|---|---|
| Engine RAM 260–470 MB / 2 engines 640–720 MB | **TBD — brief-supplied, unmeasured**; `/usr/bin/time -v python -m telugu_speech.cli ...`, read Maximum RSS |
| TTS/ASR RTF on 2 vCPU (not M4/M5) | bench harness: audio duration vs wall time, p50/p95 |
| Docker image size | `docker images vaani-x` after build |
| Voice-pack `sha256` (step_int8.onnx) | `shasum -a 256 lite/models/hima-tts-nano-te/onnx/voices/step_int8.onnx` on the Mac — no shell access in Part 1, so manifests ship with a zero hash and the registry rejects them until filled |
| Piper Telugu voice licence | Read the voice card at the manifest's `licence_url` |
| IndicConformer / Sarvam parity | **Not measured** in this checkout (NeMo not installed, no Sarvam key) — see `docs/benchmark_results.md` |

## 8. Environment variables (all optional, all with safe defaults)

| Variable | Default | Notes |
|---|---|---|
| `TELUGU_CPU_THREADS` | 2 (cap 4) | ONNX/OMP threads |
| `VAANI_TTS_SLOTS` | 2 (cap 4) | concurrent syntheses; was 8 — dangerous, see §5 |
| `VAANI_ASR_SLOTS` | 2 | ASR queue depth (single shared instance) |
| `VAANI_VOICES_DIR` | — | extra voice-pack dirs, colon-separated |
| `VAANI_VOICE_IDLE_SECS` | 1800 (0 = never unload) | idle unload |
| `VAANI_MAX_TEXT_CHARS` | 4096 | TTS input limit |
| `VAANI_MAX_CHUNK_CHARS` | 500 | synthesis chunk size |
| `VAANI_AUDIO_CACHE_ENTRIES` / `VAANI_AUDIO_CACHE_MAX_MB` | 128 / 256 | LRU audio cache |
| `VAANI_MAX_AUDIO_MB` | 25 | ASR upload limit |
| `VAANI_DEFAULT_VOICE` | `telugu_srinivas` | default voice id |
