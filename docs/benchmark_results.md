# Benchmark results

## Measured in this checkout

Date: 20 November 2026. Machine: Apple M5, 16 GB RAM, CPU inference. Audio: ten `google/fleurs` `te_in` **test** clips (CC-BY-4.0), saved under `data/raw/fleurs_te/`. Decoder: faster-whisper `small`, `language=te`, `beam_size=5`, `temperature=0`. Normalizer: this repository, digit policy spoken.

| Model | Clips | WER | CER | RTF | Model bytes | Peak RSS bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `models/telugu-asr-micro` (Telugu tiny int8) | 10 | 0.4379 | 0.2760 | 0.1095 | 44255137 | 361594880 |
| `models/telugu-asr` (Telugu small int8) | 10 | 0.3669 | 0.2451 | 0.7738 | 253164606 | 956170240 |
| faster-whisper multilingual small | 10 | 1.2722 | 0.9797 | 0.5785 | 972425620 | 1051033600 |
| Micro TTS | e2e | n/a | n/a | offline | 26269200 | CC-BY-NC MMS int8 disk |
| **Micro STT+TTS package** | — | — | — | — | **70524337 (~67 MB)** | under 100 MB |
| IndicConformer Telugu | 0 | not measured | not measured | not measured | not downloaded | NeMo is not installed |
| Sarvam | 0 | not measured | not measured | not measured | API only | no key configured |

The Whisper hypotheses include long repeated syllables. A WER above 1 means the edit count exceeds the reference length. An earlier six-clip run on the same decoder scored WER 1.3580. The ten-clip row replaces it as the current measurement.

These six clips are not a full Telugu benchmark. Region is `unspecified` because FLEURS does not label Telangana, Coastal Andhra, or Rayalaseema.

## External figures, not reproduced

| Source | Setup | Telugu figure | Status |
| --- | --- | --- | --- |
| IndicWhisper / Vistaar issue table | Average of Kathbath, Kathbath-hard, FLEURS, IndicTTS, MUCS | 28.8% WER | external, not reproduced |
| IndicASR on IndicVoices | Natural speech, secondary citation | 26.8% WER | external, not reproduced |
| Whisper on IndicVoices Telugu | Secondary citation | about 151.9% WER | external, not reproduced |
| Community IndicConformer-600M | 100 Kathbath validation clips | 22.09% WER | external, not reproduced |
| Sarvam Saaras v3 | IndicVoices, 10 languages pooled | 19.31% WER | external, not a Telugu-only number |

Do not compare the 1.2722 WER above with those percentages as if they were the same test set. Saaras v4 has no Telugu-only number copied here.

## Slices this project will report

- Region: Telangana, Coastal Andhra, Rayalaseema
- Telugu-English code-mix
- Condition: clean, noisy, phone 8 kHz, conversational

One pooled WER is not a release metric.

## Release gate for TeluguTinyASR

On the frozen local bench, after it exists:

- Clean read-speech WER within 0.02 absolute of the 120M Telugu teacher.
- Weight file at least 3 times smaller than that teacher.
- Slice tables for region, noise, 8 kHz, and code-mix.

A "better than Sarvam" statement requires Saaras scored with this normalizer on the same clips.

## Harness (0.2.0) — how a number becomes quotable

```bash
.venv311/bin/python -m telugu_speech.bench --asr-model telugu-asr-micro --sets fleurs-te kathbath-te cv-te --n 200
```

- **Frozen sets, speaker-disjoint.** Each set is a manifest at `data/benchmarks/<set>/manifest.jsonl` (FLEURS-te, Kathbath-te, Common Voice te). The test split must not share speakers with train; pass `--train-manifest` to check. Datasets are NOT in this repo — the command exits 2 and names the exact manifest path when one is missing. No data is invented.
- **Minimum size.** n ≥ 200 clips per set from ≥ 20 speakers. The ten-clip rows above do not meet it and stay labelled as what they are. `--allow-small` is a smoke-test escape only.
- **Three metrics, always together.** WER + CER + AER (akshara error rate). Telugu is agglutinative — case, plural, and postpositions fuse into one long word, so one wrong matra voids the whole word and WER runs roughly 3–4x the akshara error rate on the same transcript. Report all three, raw and normalized (digit policy `spoken`, punctuation stripped for scoring). Numbers and Latin code-mix words are scored separately (numeral WER, Latin WER) because a model can be good at Telugu words and still drop every English word — the Andhra check recalled 0 of 53 marked English words.
- **RTF + peak RAM + model revision SHA** are recorded with every run. A result without a revision is not reproducible; omit the SHA only when it is genuinely unknown, and say so.
- **Output.** JSON goes to `bench/results/<model>/<date>.json` (plus a Markdown report beside it). Results are never hand-edited; re-run the bench to change a number.
- **TTS intelligibility (opt-in).** `--tts-voice telugu_srinivas --tts-sentences <file>` synthesizes each sentence, transcribes it back with the ASR model, and scores the round trip. Without those flags the TTS section is `not_measured`. No MOS claim exists until native listeners fill `configs/mos_listeners.csv`.
- **Cross-check.** `--crosscheck-jiwer` (needs `pip install -e ".[bench]"`) recomputes WER with jiwer on the same clips; a mismatch with the repo's WER means a normalizer bug.

## v1.0 targets (targets, NOT results)

| Metric | Target (clean read speech) | Stretch (conversational / code-mixed) |
| --- | --- | --- |
| ASR WER (FLEURS-te + Kathbath-te, n ≥ 200/set) | ≤ 20% | ≤ 35% |
| ASR CER | ≤ 5% | ≤ 8% |
| ASR AER | ≤ 15% | report only |
| TTS intelligibility (synthesized → ASR back-transcription) | WER ≤ 15%, CER ≤ 6% | — |
| ASR RTF (2 vCPU) | ≤ 0.5 | — |
| TTS RTF (2 vCPU) | ≤ 0.5 | — |

Current measured position for comparison: live decoder WER 0.2899 / CER 0.0811 on 10 clean clips (20 Nov 2026). Every target above is still **TBD** until a frozen-set run measures it.

## CI gate (documented, not yet wired)

Add this when CI is enabled — do not claim CI exists until Mani enables it. Any model swap that worsens Telugu WER by more than 2 points absolute on FLEURS-te fails the PR:

```yaml
# .github/workflows/bench-gate.yml (proposed, not yet added)
- name: Benchmark gate
  run: |
    .venv311/bin/python -m telugu_speech.bench --asr-model telugu-asr-micro --sets fleurs-te --n 200
    .venv311/bin/python scripts/check_bench_gate.py --baseline bench/results/telugu-asr-micro/baseline.json --max-wer-regression 0.02
```

`scripts/check_bench_gate.py` does not exist yet — it is part of the gate work, not assumed here.

## Detection and Sarvam rows (not scores)

The 10 FLEURS rows have no `lid_label` and no non-Telugu negatives. Detection macro F1 is **not evaluable**. `scripts/test_sarvam.py` prints `not measured` for Saaras v4 `transcribe` and `codemix` when `SARVAM_API_KEY` is unset and does not send audio. IndicConformer remains **not measured** until NeMo and the gated checkpoint are installed. No row in this file claims parity with Saaras or Bulbul.

## Same-machine suite

```bash
TELUGU_SPEECH_SLOTS=2 TELUGU_CPU_THREADS=2 .venv311/bin/python scripts/run_benchmark_suite.py
```

The run writes `experiments/benchmarks/suite/results.json`, CSV files, and `report.html`. Models that are not on disk stay `not_installed`. MOS is left empty. The 20 November 2026 suite did not replace the table above: the tiny recognizer again scored WER 0.4379, and the 241 MB recognizer scored WER 0.3491 in that process. Those ten clips are still the only shared set.

The live recognizer was changed later the same day. It keeps word timing, treats silence after 400 ms as a cut, and drops a piece whose average log probability is below −1.0 or whose sentence repeats. On the same ten clean clips that setting scored WER 0.2899, CER 0.0811, and a real-time factor of 0.18. The table above stays the earlier text-only measurement. A paired rerun of the 16 Andhra clips moved only slightly and still recalled 0 of 53 marked English words.

## Andhra prompted speech, 20 November 2026

Sixteen clips, 15 speakers, 106 seconds, from `SujithPulikodan/Vaani-sample-data` (Project Vaani, CC-BY-4.0). District Annamaya, Andhra Pradesh. Each clip is a picture description with a background-noise tag and at least one English word marked in braces. The shipped recognizer, Telugu forced, beam 5, voice-activity detection on, scored WER 0.5458 and CER 0.3546 on 251 words. Turning detection off, and leaving the language automatic, both stayed at WER 0.5458. The clean ten-clip check in the same process was 0.4235 with detection on and 0.4353 with it off, so those two settings were not adopted. On the first clip the reference ends “లైట్లు light కూడా వెలుగుతూ ఉన్నాయి సోఫా sofa” and the hypothesis stops at “లైట్లు కూడా వెలుగుతూ ఉన్నాయి”. English words are dropped. This is not a conversation set and not a listener score.

The same 16 clips were split by the brace-marked English words. Mixed-sentence WER stays 0.5458. Telugu-only WER is 0.4394 and Telugu-only CER is 0.1966. English-word recall is 0 of 53: none of the marked English words appeared as a Latin token. Error counts on the mixed text are 56 substitutions, 78 deletions and 3 insertions. VAD off and automatic language both stay at mixed WER 0.5458 and English recall 0. Of 44 unique English words, 40 are already a whole token in `vocabulary.json`. The four that are not are `a_k`, `lorry`, `sandles` and `t_v`. A missing whole token is not proof of an acoustic miss, and a present token that is never emitted is not a tokenizer gap. The per-clip rows are in `experiments/benchmarks/suite/codemix_diagnostic.json`.

Repeat with `.venv311/bin/python scripts/score_research_sets.py` after the sample files are present.

## Telangana news sample, 20 November 2026

Ten MIT-licensed clips were downloaded from `nickthelegend/Telengana-Speech-Dataset`. They are Telangana news speech, not conversation, not Andhra speech, and not street noise. Adjacent clips overlap, so a score against one caption is not valid. Six clips whose audio matches the words inside them scored WER 0.4909 and CER 0.1614 on 55 reference words. Real-time factor was about 0.11. The recognizer was not changed. English words in these clips are written in Telugu script, so this is not a Latin code-mix test. No listener scored the voices.

Competitor rows are located, not downloaded, by:

```bash
.venv311/bin/python scripts/run_benchmark_suite.py --locate-only
.venv311/bin/python scripts/run_benchmark_suite.py --download sherpa-onnx
```

`--download` only prints whether the fetch is allowed. It does not download. Laya is refused because its card says the project is private for now. Chatterbox Telugu is refused until its size is known and a consented reference recording exists. The production recognizer, voices, thread count, and two-slot limit are unchanged.
