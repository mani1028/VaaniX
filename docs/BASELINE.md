# Vaani X — true baseline (measured 22 Nov 2026)

Recorded by the MEASURE worker before any accuracy work. Every number
below was read from the code, configs, or files on this Mac on
22 Nov 2026, or is quoted from `docs/benchmark_results.md` /
`docs/ACCURACY.md` with its date, set, and clip count attached. Nothing
here is a target, a vendor figure, or a guess. Anything not yet measured
is marked **UNMEASURED** with the exact command Mani should run.

No claim in this file says the system is 100% accurate. It is not.

## Runtime

| Item | Value | Source |
| --- | --- | --- |
| Python | 3.11.15 (Homebrew `python@3.11`) | `.venv311/pyvenv.cfg` |
| Venv | `.venv311/` exists (`bin/`, `lib/`, `pyvenv.cfg`) | directory listing |
| Package | `telugu-speech` 0.2.0, Telugu only | `pyproject.toml` |
| API | FastAPI, OpenAI-compatible `/v1/audio/*`, auth-required | `src/telugu_speech/api/` (`main.py`, `routes_openai.py`, `auth.py`) |
| Config | `configs/vaani.yaml` v2.1, 26 keys, env `VAANI_*` > file > defaults | `configs/vaani.yaml` |
| ASR chain | `[telugu-asr-micro]` (faster-whisper only) | `configs/vaani.yaml` |
| TTS chain | `[hima-nano]` (Hima ONNX int8 only) | `configs/vaani.yaml` |
| Default voice | `telugu_srinivas` | `configs/vaani.yaml` |
| Slots / threads | TTS 2, ASR 2, CPU threads 2 | `configs/vaani.yaml` |
| Text caps | max 2000 chars, max chunk 180 chars, upload 25 MB | `configs/vaani.yaml` |
| Machine for the 20 Nov 2026 numbers | Apple M5, 16 GB RAM, CPU inference | `docs/benchmark_results.md` |
| Current Mac | macvisys3 (Mac17,3); RAM **UNMEASURED** this session | — |

## Dependencies (pins from `pyproject.toml`)

Core (always installed): `fastapi>=0.115,<1`, `numpy>=1.26,<3`,
`pydantic>=2.6,<3`, `python-multipart>=0.0.9,<1`, `pyyaml>=6.0,<7`,
`uvicorn>=0.30,<1`.

Optional extras: `asr` = `faster-whisper>=1.0,<2`; `host` = faster-whisper
+ `onnxruntime>=1.17,<2` + `safetensors>=0.4,<1` + `sentencepiece>=0.2,<1`;
`dev` = `httpx>=0.27,<1`, `pytest>=8,<9`; `bench` = pytest + `jiwer>=3,<4`
(WER cross-check only, not needed to run the server).

Unpinned but imported by scripts (not in `pyproject.toml`): `soundfile`
(`scripts/fetch_fleurs_sample.py`), `datasets` + `huggingface_hub`
(fetch scripts). The Hima pack's own `requirements.txt` lists
`onnxruntime, numpy, safetensors, sentencepiece, soundfile, pyyaml`
(unpinned).

Installed versions and installed sizes: **UNMEASURED**. Run:

```bash
.venv311/bin/pip list
.venv311/bin/pip freeze > /tmp/vaani-pip-freeze.txt
du -sh .venv311
```

There is no `requirements*.txt` at the repo root; `pyproject.toml` is the
only dependency declaration.

## Models on disk (verified file-by-file, 22 Nov 2026)

Sizes are bytes on disk, summed from the file listings. 1 MiB = 1,048,576 B.
Revisions / SHA-256 are **TBD** for every ASR model in `configs/models.yaml`
— no model hash is pinned anywhere except the Hima voice pack (below).

### ASR — faster-whisper, CPU, `compute_type=int8`, language locked to `te`

| Model | Contents | Total bytes | MiB |
| --- | --- | ---: | ---: |
| `models/telugu-asr-micro` (tiny int8, **default**) | `model.bin` 40,520,007; `tokenizer.json` 2,480,466; `vocabulary.json` 1,068,103; `preprocessor_config.json` 184,990; `config.json` 1,571 | 44,255,137 | 42.2 |
| `models/telugu-asr` (small int8) | `model.bin` 248,203,797; `tokenizer.json` 3,653,236; `vocabulary.json` 1,119,969; `preprocessor_config.json` 184,990; `config.json` 2,614 | 253,164,606 | 241.4 |
| `models/whisper-multilingual-tiny` (vanilla multilingual tiny, **not** the Telugu fine-tune) | `model.bin` 75,538,270; `tokenizer.json` 2,203,239; `vocabulary.txt` 459,861; `config.json` 2,249; `README.md` 1,991 | 78,205,610 | 74.6 |

`lite/models/telugu-asr-micro/` is a second copy of the micro model on
disk (same five files, same byte sizes). Whether it is a copy or a hard link was not verified through the file API.

Not on disk: `vasista22-whisper-telugu-small/base` (download-required,
`configs/models.yaml`), IndicConformer (any size), the faster-whisper
multilingual **small** behind the 972,425,620 B row in
`docs/benchmark_results.md` (that file is not in `models/`).

### TTS

| Model | Contents | Total bytes | MiB |
| --- | --- | ---: | ---: |
| `models/telugu-tts` (Meta MMS Telugu VITS, FP32, **CC-BY-NC-4.0**) | `model.safetensors` 145,251,448 + tokenizer/config files | 145,254,657 | 138.5 |
| `models/telugu-tts-micro` (MMS INT8-on-disk, **CC-BY-NC-4.0**) | `weights.int8.npz` 26,207,151; `scales.json` 58,612 + tokenizer/config files | 26,269,200 | 25.1 |
| `lite/models/hima-tts-nano-te` (Hima ONNX int8, 62M params, **default TTS engine**) | `onnx/voices/step_int8.onnx` 48,565,039; `onnx/voices/frame_int8.onnx` 11,622,260; `onnx/decoder_int8.onnx` 23,482,392; `prefix.safetensors` 18,020,039; `tokenizer.model` 618,690; `voices/voices.safetensors` 50,688; `voices/voices.json` 349; `config.yaml` 676; `README.md` 3,848; `requirements.txt` 61; plus a second `onnx/base/step_int8.onnx` 48,565,039 (same size) | 150,929,081 inventoried (plus `hima_onnx/` + `text/` code dirs, not inventoried) | 143.9 |

One-voice-set int8 weights (`step` + `frame` + `decoder_int8`) =
83,669,691 B (~84 MB decimal), matching the upstream pack README's
"84 MB" figure. Upstream README also reports int8 ≈ fp32 accuracy on
1,202 sentences (letters wrong 3.39% vs 3.15% on Sarvam saaras v4) — that
is the upstream pack's figure, not a measurement by this repo.

Hima voices (from `src/telugu_speech/tts/voices.py` + voice manifests):
`telugu_srinivas` male, ~100–115 Hz; `telugu_sravani` female, ~185–225 Hz;
both seed 0, 24 kHz, style `conversational` only. Pack SHA-256
(`onnx/voices/step_int8.onnx`):
`a2d0dbbac3cd39f86d7b3125c405418a02d0cf1132b9e3cf64546ca4f3487f22`
— filled in `configs/voices/srinivas.yaml` and deployed identically in
`voices/telugu_srinivas/manifest.yaml` (verified same content 22 Nov 2026).

Micro STT+TTS package (micro ASR + micro MMS TTS):
44,255,137 + 26,269,200 = **70,524,337 B (~67.3 MiB)**, matching
`models/README.md`.

All of `models/` together: **547,149,697 B (~521.8 MiB)**.

### ASR decoding (the live decoder — `src/telugu_speech/asr/decode.py` + `infer.py`)

Language locked `te` (auto-detect refused). Beam 5, temperature 0.0,
VAD on (threshold 0.5, min speech 250 ms, min silence 400 ms, pad
160 ms), no-speech threshold 0.6, logprob threshold −1.0,
compression-ratio threshold 2.4, word-probability threshold 0.6,
hallucination-silence threshold 1.0 s, `condition_on_previous_text`
False, CPU threads 2, `compute_type` int8. Word-level filtering drops
low-probability words but never Latin words as a class; repeated phrases
are collapsed. Audio prep (`audio/preprocessing.py`): resample to
16 kHz, peak-normalise to 0.95, no silence trim by default.

## Bench numbers (all from 20 Nov 2026 — see `docs/benchmark_results.md`)

Normalizer: this repo, digit policy spoken. AER was not recorded for
the 10-clip runs below (the harness records it now; re-run to get it).

| Model / set | n | WER | CER | AER | RTF | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Live decoder, 10 clean FLEURS `te_in` clips | 10 | **0.2899** | **0.0811** | UNMEASURED | 0.18 | UNMEASURED |
| Tiny int8 (text-only bench), same 10 clips | 10 | 0.4379 | 0.2760 | UNMEASURED | 0.1095 | 361,594,880 B (~345 MB) |
| Small int8, same 10 clips | 10 | 0.3669 | 0.2451 | UNMEASURED | 0.7738 | 956,170,240 B (~912 MB) |
| Faster-whisper multilingual small, same 10 clips | 10 | 1.2722 | 0.9797 | UNMEASURED | 0.5785 | 1,051,033,600 B |
| Andhra prompted speech (Vaani sample data, Annamaya, 15 speakers, 106 s, 251 words) | 16 clips | 0.5458 | 0.3546 | UNMEASURED | UNMEASURED | UNMEASURED |
| Andhra, Telugu-only words | 16 clips | 0.4394 | 0.1966 | UNMEASURED | — | — |
| Andhra, English-word recall | 16 clips | **0 of 53** marked English words appeared as a Latin token | — | — | — | — |
| Telangana news, 6 matching clips, 55 words | 6 | 0.4909 | 0.1614 | UNMEASURED | ~0.11 | UNMEASURED |

Notes and discrepancies, stated plainly:

- The 10-clip rows do **not** meet the repo's own quotable gate
  (n ≥ 200 clips/set, ≥ 20 speakers, speaker-disjoint). They are smoke
  numbers, not results.
- The same-day suite re-run scored the small model at WER **0.3491**,
  not 0.3669, on the same 10 clips — two runs disagree, so a paired
  re-run is needed before quoting either.
- The live-decoder 0.2899 vs text-only 0.4379 gap is the decoding path
  (segment/word filtering, silence cuts); the bench table keeps the
  earlier text-only row. A paired re-run of the current `transcribe()`
  path on the same 10 clips is still owed.
- Andhra error counts (mixed text): 56 substitutions, 78 deletions,
  3 insertions. VAD-off and auto-language both stayed at WER 0.5458.
- TTS intelligibility (synthesise → ASR back-transcription):
  **not measured** — the opt-in `--tts-voice`/`--tts-sentences` bench
  has never been run. MOS: **none** — `configs/mos_listeners.csv` is a
  header row only (verified 22 Nov 2026).
- Detection macro-F1: **not evaluable** (no `lid_label` on any clip).
- Sarvam Saaras / IndicConformer: **not measured** (no key, NeMo not
  installed, checkpoint not downloaded).
- External figures in `docs/benchmark_results.md` (IndicWhisper 28.8%,
  IndicConformer-600M 22.09% on Kathbath, Sarvam Saaras v3 19.31%
  pooled) are **not reproduced** here and must not be compared with
  the rows above as if they were the same test set.

Targets (NOT results): WER ≤ 20% / CER ≤ 5% / AER ≤ 15% on FLEURS-te +
Kathbath-te, n ≥ 200/set, ≥ 20 speakers, speaker-disjoint; stretch
WER ≤ 35% conversational / code-mixed; RTF ≤ 0.5 on 2 vCPU; TTS
intelligibility WER ≤ 15% / CER ≤ 6%.

## Datasets on disk (verified 22 Nov 2026)

| Set | Location | Contents |
| --- | --- | --- |
| FLEURS `te_in` test sample | `data/raw/fleurs_te/` + `data/manifests/eval.jsonl` | 10 WAVs, 6,628,420 B total, 103.56 s, 10 speakers, region `unspecified`, condition `clean`, durations 5.40–17.46 s |
| Andhra prompted sample | `data/raw/research/andhra/` + `data/manifests/andhra_sample.jsonl` | 16 WAVs, 15 speakers |
| Telangana news sample | `data/raw/telangana_news/audio/` + `metadata.jsonl` (205,792 B) | 10 clips downloaded; only 6 matched their captions (see benchmark doc) |
| Vaani Andhra | `data/raw/research/vaani-andhra/audio/` | Present; clip count not inventoried this session |
| Frozen bench sets | `data/benchmarks/<set>/manifest.jsonl` | **DOES NOT EXIST** — `fleurs-te`, `kathbath-te`, `cv-te` manifests are all missing, so `python -m telugu_speech.bench` exits 2 today. Kathbath-te and Common Voice te are not on disk in any form. |

## Harness (what exists before this job)

- Frozen-set CLI: `src/telugu_speech/bench.py` (`python -m telugu_speech.bench`),
  scoring via `src/telugu_speech/evaluation/benchmark.py` (`run_benchmark`)
  with WER+CER+AER, raw + normalized, per-region / per-condition /
  code-mix slices, RTF, peak RSS, model bytes, revision SHA.
- Metrics: `src/telugu_speech/evaluation/asr_metrics.py` (word/character/
  akshara error rates, micro-average, numeral/Latin segment scores).
- Suite: `scripts/run_benchmark_suite.py` → `experiments/benchmarks/suite/`
  (`results.json`, `report.html`, `stt.csv`, `tts.csv`, per-model CSVs;
  20 Nov 2026 run present).
- Research diagnostics: `scripts/score_research_sets.py` (Andhra code-mix),
  `src/telugu_speech/evaluation/error_analysis.py`.
- Real-world harness (new, this job):
  `src/telugu_speech/evaluation/realworld.py` + `scripts/record_clips.py`
  + `docs/REALWORLD.md` — six conditions, per-condition table, no
  blended headline. See `docs/REALWORLD.md`.

## UNMEASURED — commands for Mani

```bash
cd "/Users/visys/.cursor/tts sst"
# Installed dependency versions + venv size
.venv311/bin/pip list
du -sh .venv311
du -sh . models lite/models data
# Machine RAM / CPU (for the baseline header)
system_profiler SPHardwareDataType
# Cold start: time to import + first transcription (tiny model, 1 clip)
time .venv311/bin/python -c "from telugu_speech.asr.infer import FasterWhisperBackend; b=FasterWhisperBackend('models/telugu-asr-micro'); print(b.transcribe(__import__('pathlib').Path('data/raw/fleurs_te/fleurs_000.wav')))"
# Paired re-run of the current decoder on the same 10 clean clips (settles 0.2899 vs 0.4379 + the 0.3669 vs 0.3491 small-model discrepancy; also records AER)
.venv311/bin/python -m telugu_speech.bench --asr-model telugu-asr-micro --sets fleurs-te --n 10 --allow-small
# Full test suite
.venv311/bin/python -m pytest -q
```

Note: the 10-clip bench above needs a `data/benchmarks/fleurs-te/`
manifest first (it does not exist yet — see `docs/REALWORLD.md` for
the fetch/prepare commands). Until then, the `--n 10 --allow-small`
re-run must use a manifest built from `data/manifests/eval.jsonl`
copied to `data/benchmarks/fleurs-te/manifest.jsonl`.
