# Real-world ASR bench — score the model on real recordings

The frozen-set bench (`python -m telugu_speech.bench`) scores curated
sets with an n ≥ 200 / ≥ 20-speaker gate. This harness scores whatever
real recordings exist — your own voice, background noise, phone-quality
audio, code-mixed speech, fast speech, several speakers — with no gate
and no invented data. A small set is reported as what it is: per-condition
numbers with n and speaker counts. Never quote a single blended number
from a real-world run; quote the per-condition rows.

## The six conditions

| Condition | What it means | How to get clips |
| --- | --- | --- |
| `clean` | Quiet room, read Telugu, 16 kHz | Read a sentence aloud, or use the FLEURS sample (see below) |
| `noise` | Real background noise (street, fan, TV, market) | Record in the noisy place itself — do not add fake noise to a clean clip and call it real |
| `phone_8k` | 8 kHz audio (phone call / 8 kHz export) | Record at 8 kHz, or export a call recording at 8 kHz |
| `code_mix` | Telugu-English mixed speech | Speak naturally with English words in ("... light కూడా ..."), write the Latin words exactly as said |
| `fast` | Fast / hurried speech | Read the same sentences at natural fast speed |
| `multi_speaker` | More than one speaker in the clip | Two people talking, or a conversation recording |

Unknown conditions are rejected. Each condition is a row in the output
table with its own WER / CER / AER / numeral-WER / Latin-WER / RTF,
clip count, speaker count, and audio seconds. The pooled row is printed
last and labelled pooled — it is not the result.

## Quick start (smoke test, uses clips already on disk)

```bash
cd "/Users/visys/.cursor/tts sst"
.venv311/bin/python -m telugu_speech.evaluation.realworld --asr-model telugu-asr-micro --manifest data/realworld/sample_manifest.csv
```

The sample manifest has 2 clean FLEURS clips (real audio + real
references from `data/manifests/eval.jsonl`). Expect a small-n warning —
2 clips is a smoke test, not a result.

## Add your own clips

### Option A — record with the script (needs a mic + a recorder library)

```bash
# One clip:
.venv311/bin/python scripts/record_clips.py --prompt "నమస్కారం, ఎలా ఉన్నారు?" --condition clean --speaker mani
# Several prompts from a file (one Telugu sentence per line):
.venv311/bin/python scripts/record_clips.py --prompts-file prompts.txt --condition noise --speaker mani
# Phone-quality records at 8 kHz automatically:
.venv311/bin/python scripts/record_clips.py --prompt "రేపు ఉదయం కాల్ చేయండి" --condition phone_8k --speaker mani
# See what would be recorded, without touching the mic:
.venv311/bin/python scripts/record_clips.py --prompts-file prompts.txt --condition fast --speaker mani --dry-run
```

Say **exactly** the prompt text — the prompt IS the reference transcript.
If you stumble or say something different, delete that WAV and its
manifest row and re-record; a wrong reference poisons the score.

If `sounddevice` / `pyaudio` is not installed the script refuses to
record and tells you how to install it — it never writes silence.

```bash
brew install portaudio
.venv311/bin/pip install sounddevice
```

### Option B — record without the script

1. Record with Voice Memos or QuickTime Player (File > New Audio Recording).
2. Convert to mono WAV — 16 kHz, or 8 kHz for `phone_8k`:

```bash
ffmpeg -i in.m4a -ar 16000 -ac 1 data/realworld/clips/mani_clean_000.wav
ffmpeg -i in.wav -ar 8000 -ac 1 data/realworld/clips/mani_phone_8k_000.wav
```

3. Append a row to `data/realworld/manifest.csv` (header:
   `path,reference_text,condition,speaker,region,duration_s`):

```csv
data/realworld/clips/mani_clean_000.wav,నమస్కారం ఎలా ఉన్నారు,clean,mani,unspecified,
```

Only record yourself or someone who agreed to be recorded for this
project. Clips stay on this Mac; nothing is uploaded.

## Manifest format

CSV (shown above) or JSONL with the same fields. Required:
`path`, `reference_text`, `condition`. Optional: `speaker`
(default `unknown`), `region` (default `unspecified`), `duration_s`
(measured from the WAV when omitted), `utt_id` (defaults to the file
stem). JSONL also accepts `audio` for `path` and `raw_text` for
`reference_text`. Audio paths resolve against the working directory
first, then the manifest's folder — run from the repo root.

A missing audio file fails the run and names the path. Clips are never
silently skipped. An unknown condition fails the run and lists the six
allowed ones.

## Score a manifest

```bash
.venv311/bin/python -m telugu_speech.evaluation.realworld --asr-model telugu-asr-micro --manifest data/realworld/manifest.csv
# Other model, custom output, smoke cap of 3 clips per condition:
.venv311/bin/python -m telugu_speech.evaluation.realworld --asr-model telugu-asr --manifest data/realworld/manifest.csv --out bench/results/telugu-asr/realworld.json --max-per-condition 3
# List the six conditions:
.venv311/bin/python -m telugu_speech.evaluation.realworld --list-conditions
```

Output: a per-condition table on stdout, JSON at
`bench/results/<model>-realworld/<date>.json` (or `--out`), and a
Markdown report beside it. The JSON records model id, model bytes,
revision SHA when passed (`--revision`, never guessed), normalizer,
peak RSS, per-condition aggregates, and per-clip rows with hypothesis
text, RTF, and clipping fraction. Results are never hand-edited;
re-run the bench to change a number.

Scoring reuses the frozen bench's pieces — same normalizer (digit
policy spoken), same WER/CER/AER, same 16 kHz `prepare_utterance`
(resample + peak-normalise 0.95), same RTF / peak-RSS measurement — so
a real-world `clean` row is comparable with a frozen-set clean row
modulo the different clips.

## Frozen sets: FLEURS / Kathbath / Common Voice te

The frozen bench needs `data/benchmarks/<set>/manifest.jsonl` for
`fleurs-te`, `kathbath-te`, `cv-te`. None of those manifests exists yet
(verified 22 Nov 2026), so the frozen bench exits 2 today. No data is
faked here — prepare each set first:

```bash
# FLEURS te_in test sample (10 clips; the fetch script caps at 20 by design).
# Already on disk: data/raw/fleurs_te/ + data/manifests/eval.jsonl.
.venv311/bin/python scripts/fetch_fleurs_sample.py --count 10
# A full 200-clip FLEURS fetch is NOT scripted yet — do not quote n=200 until it lands.

# Kathbath Telugu: download the audio yourself (Hugging Face ai4bharat/Kathbath,
# CC-BY-4.0 packaging — verify the card's terms first), place WAVs + one .txt
# transcript per clip in a folder, then build a manifest (no download in this step):
.venv311/bin/python scripts/prepare_kathbath.py --source <folder-with-wavs> --output data/benchmarks/kathbath-te/manifest.jsonl --region unspecified --condition clean

# Common Voice Telugu: download the validated set from commonvoice.mozilla.org
# (CC0), then build a manifest with the same columns as the real-world manifest
# (path, reference_text, condition, speaker) and place it at
# data/benchmarks/cv-te/manifest.jsonl. No verified one-command fetch exists
# in this repo for Common Voice — do not invent one.
```

Then run the frozen bench:

```bash
.venv311/bin/python -m telugu_speech.bench --asr-model telugu-asr-micro --sets fleurs-te kathbath-te cv-te --n 200
```

## What this harness does NOT do

- No synthetic noise, no resampling tricks presented as recordings: a
  `noise` clip must be recorded in noise; a `phone_8k` clip must be
  8 kHz audio. (`prepare_utterance` resamples everything to 16 kHz for
  the model, so the 8 kHz condition measures information loss, not
  sample-rate handling.)
- No MOS / listener scores — `configs/mos_listeners.csv` is header-only.
- No TTS scoring — TTS intelligibility stays the frozen bench's opt-in
  `--tts-voice` / `--tts-sentences` path, still unmeasured.
- No speaker-disjointness check against train: real-world clips are a
  test-only convenience set. If a clip's speaker is in any train set,
  say so in the report; do not quote it as held-out.
