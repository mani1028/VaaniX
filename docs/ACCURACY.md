# Vaani X — accuracy (measured 20–22 Nov 2026)

This file is the honest accuracy record. Every number carries its test
set, clip count, and conditions. Nothing here claims the system is 100%
accurate. It is not.

The quotable gate (from `docs/BASELINE.md`): a result may only be quoted
when it comes from a frozen set with **n ≥ 200 clips, ≥ 20 speakers,
speaker-disjoint**. None of the rows below meets that gate — they are
smoke numbers, not results. The frozen manifests do not exist yet
(`data/benchmarks/fleurs-te/`, `kathbath-te/`, `cv-te/` are all missing,
so `python -m telugu_speech.bench` exits 2 until a manifest is built;
see §8).

## 1. Baseline (measured 20 Nov 2026, Apple M5, 16 GB RAM, CPU inference)

Source: `docs/BASELINE.md` (which quotes `docs/benchmark_results.md`).
Normalizer: this repo, digit policy spoken. AER was not recorded for the
10-clip runs.

| Model / set | n | WER | CER | RTF | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Live decoder, 10 clean FLEURS `te_in` clips** | 10 | **0.2899** | **0.0811** | 0.18 | UNMEASURED |
| Tiny int8, text-only bench, same 10 clips | 10 | 0.4379 | 0.2760 | 0.1095 | ~362 MB |
| Small int8, text-only bench, same 10 clips | 10 | 0.3669 | 0.2451 | 0.7738 | ~956 MB |
| Faster-whisper multilingual small, same 10 clips | 10 | 1.2722 | 0.9797 | 0.5785 | ~1,051 MB |
| Andhra prompted speech (Vaani sample data, 15 speakers, 106 s, 251 words) | 16 clips | 0.5458 | 0.3546 | UNMEASURED | UNMEASURED |
| Andhra, Telugu-only words | 16 clips | 0.4394 | 0.1966 | — | — |
| Andhra, English-word recall | 16 clips | **0 of 53** marked English words appeared as a Latin token | — | — | — |
| Telangana news, 6 matching clips, 55 words | 6 | 0.4909 | 0.1614 | ~0.11 | UNMEASURED |

Discrepancies, stated plainly:

- **Small-model re-run disagrees with the table.** The same-day suite
  re-run scored the small model at WER **0.3491**, not 0.3669, on the
  same 10 clips. Two runs disagree, so a paired re-run is needed
  before quoting either number.
- **Live decoder vs text-only gap is unexplained.** The live decoder
  scores 0.2899 on the same 10 clips where the text-only bench scores
  the tiny model at 0.4379. The gap is attributed to the decoding path
  (segment/word filtering, silence cuts), but no paired re-run of the
  current `transcribe()` path on the same 10 clips has been done. That
  re-run is still owed (see §8).
- **Andhra error counts (mixed text):** 56 substitutions, 78 deletions,
  3 insertions. VAD-off and auto-language both stayed at WER 0.5458.
- **External figures are not comparable.** IndicWhisper 28.8%,
  IndicConformer-600M 22.09% on Kathbath, and Sarvam Saaras v3 19.31%
  pooled (all quoted in `docs/benchmark_results.md`) were measured on
  different test sets. Do not compare them with the rows above as if
  they were the same test.

Targets (NOT results): WER ≤ 20% / CER ≤ 5% / AER ≤ 15% on FLEURS-te +
Kathbath-te, n ≥ 200/set, ≥ 20 speakers, speaker-disjoint; stretch
WER ≤ 35% conversational / code-mixed; RTF ≤ 0.5 on 2 vCPU; TTS
intelligibility WER ≤ 15% / CER ≤ 6%.

## 2. Model decision: default stays `telugu-asr-micro`

- The live decoder with the micro model scores **0.2899 WER** on the
  10 clean FLEURS clips; the small model scores **0.3669** (text-only
  bench) on the same clips. The default does not change on the basis
  of a model that measures worse on the only paired comparison we
  have — and the 0.3669 figure itself is disputed (§1).
- The small model costs far more for that worse number: RTF **0.77**
  vs **0.11** (text-only bench) and peak RSS **~956 MB** vs **~362 MB**.
  On the 1–2 GB RAM target that cost alone disqualifies it as a
  default.
- The small model stays **opt-in**: `--asr-model telugu-asr` on the
  bench, or `asr.chain` in `configs/vaani.yaml`.

Two naming facts that cause real confusion:

- The on-disk "small" model is **`telugu-asr`** (`models/telugu-asr`,
  241.4 MiB), **not** `vasista22-small`. The vasista22 fine-tune
  (`vasista22-whisper-telugu-small/base`) is **not installed**; it is
  listed in `configs/models.yaml` as download-required.
- The id **`telugu-asr-small` means three different things in three
  places, and two of them are wrong.** The API model name
  `vaani-asr-small` is an alias for **`telugu-asr-micro`**
  (`core/fallback.py`: `ASR_MODEL_ALIASES`) — it does not select the
  small model. The bench resolves `--asr-model` as a folder under
  `models/` (`bench.py`: `resolve_asr_model`), so `--asr-model
  telugu-asr-small` exits with "asr model not found" — there is no
  `models/telugu-asr-small` directory. The small model is reached only
  as `--asr-model telugu-asr`. Fix the ids before quoting any
  "small-model" number; until then, say which id you used.

## 3. What changed in this job (accuracy work, 22 Nov 2026)

### `clean_transcript()` — `src/telugu_speech/text/postprocess.py`

The only rewriting the live transcribe path does. Deliberately small:

- NFC normalisation (a decomposed vowel sign from the decoder becomes
  its composed form).
- Per-line whitespace collapse, empty-line removal. Sentence-per-line
  structure from `accepted_transcript` is kept; lines are joined with
  `\n`, never flattened.
- Digits are kept exactly as spoken (`5` stays `5` — the bench
  normalizer expands digits to words for **scoring**, but the API must
  return what was said, not an expansion).
- Latin code-mixed words are kept with their casing; punctuation is
  kept.
- A transcript that is already clean comes back unchanged: the
  function is **idempotent** (covered by `tests/test_asr_postprocess.py`).

### Cleaning on all three transcript exits — `src/telugu_speech/asr/infer.py`

`clean_transcript()` now runs on every path that returns a transcript:

1. `FasterWhisperBackend.transcribe()` — the bench and legacy
   `/api/v1/asr/transcribe` path.
2. `accepted_transcript()` — the `/v1/audio/transcriptions` path
   (`routes_openai.py`: `_transcribe_file`).
3. `IndicConformerBackend.transcribe()`.

### Decode unification — no more hard-coded decode literals in the routes

`routes_openai.py`: `_transcribe_file()` now reads the decode settings
from `backend.decoding` — the single source of truth, shared with
`asr/infer.py`. The old literals had already drifted: they omitted the
no-speech and logprob thresholds, so the two live paths could decode
differently. They cannot drift now.

### `decoding_from_env()` actually reaches the live paths

`VAANI_ASR_BEAM`, `VAANI_ASR_TEMPERATURE`, `VAANI_ASR_VAD`, and
`VAANI_ASR_NO_SPEECH_THRESHOLD` were **dead code before this job**:
the server built its backend with the defaults unconditionally, so
setting them changed nothing. Now `api/main.py`: `configure()` passes
`decoding=decoding_from_env()` into `FasterWhisperBackend`, and
`core/fallback.py`: `load_asr_backend()` does the same for chain-loaded
backends. A bad env value raises at backend construction and is
recorded as a failed chain attempt naming the value — never silently
ignored. Covered by `tests/test_asr_postprocess.py`
(`test_decoding_from_env_*`).

### Language `te` lock — verified in all 5 places

1. `DecodingConfig` validation rejects any language other than `te`
   (`en` and `auto` both raise).
2. `FasterWhisperBackend.transcribe_with_segments()` passes
   `language=config.language` to faster-whisper.
3. `routes_openai.py`: `_transcribe_file()` passes `config.language`.
4. `routes_openai.py`: `/v1/audio/transcriptions` returns 400
   `unsupported_language` for any `language` other than `te`.
5. The legacy `/api/v1/asr/transcribe` route inherits the lock from
   the backend (`backend.transcribe()`), which only ever decodes `te`.

### Logprob / no-speech thresholds now explicit — no behaviour change

The thresholds the decoder already applied are now stated in one
place (`DecodingConfig` measured defaults, pinned by
`test_decoding_config_measured_defaults`): logprob threshold **−1.0**,
no-speech threshold **0.6**, compression-ratio threshold **2.4**,
word-probability threshold **0.6**, hallucination-silence threshold
**1.0 s**, `condition_on_previous_text` **False**. These are the same
values faster-whisper uses by default; making them explicit changed
no behaviour. The only behavioural difference from the old route
literals is that the `/v1` path now applies the no-speech/logprob
thresholds it previously omitted.

## 4. Tried and rejected

- **Beam / VAD / temperature changes — rejected, defaults kept.**
  Measured defaults stay: beam **5**, temperature **0.0**, VAD **on**
  (threshold 0.5, min speech 250 ms, min silence 400 ms, pad 160 ms).
  There is no paired evidence for any other value: no A/B run has
  scored beam 3, VAD off, or temperature > 0 on the same clips. The
  env knobs exist for a one-variable-at-a-time A/B (see §8); until
  one is run, changing a default is a guess.
- **Correction table — stays empty.** `CORRECTIONS` in
  `text/postprocess.py` ships empty by design. No entry has bench
  evidence behind it (an entry requires a source like "fleurs-te
  n=200: 14 occurrences, 12 fixed, 0 broken"). The `EXAMPLE_CORRECTIONS`
  in the module are shape examples only — not validated, not applied.

<!-- PART2 -->
