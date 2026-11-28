# Launch kit — Vaani X

How to make Vaani X the project Telugu developers reach for first, without claiming anything that is not yet true. Launch is a checklist, not a date on a calendar: every item below is either measured or explicitly not launched yet.

## 1. Measure BEFORE launching (do not skip)

| # | Gate | Command | Pass means |
| --- | --- | --- | --- |
| 1 | Voice-pack hash filled | `shasum -a 256 lite/models/hima-tts-nano-te/onnx/voices/step_int8.onnx`, paste into both `configs/voices/*.yaml`, copy to `voices/*/manifest.yaml` | Registry accepts both packs; `GET /v1/audio/voices` lists them from the registry, not the fallback |
| 2 | Tests green | `.venv311/bin/python -m pytest -q` | Exit 0 |
| 3 | Smoke curls pass | Old `/api/v1/*` routes + new `/v1/*` routes (README Checks section) | Both surfaces answer correctly |
| 4 | Bench, frozen set | `python -m telugu_speech.bench --asr-model telugu-asr-micro --sets fleurs-te kathbath-te cv-te --n 200` | n ≥ 200/set, ≥ 20 speakers, WER+CER+AER + RTF + RAM + SHA recorded |
| 5 | Docker size measured | `docker images vaani-x` after `docker build -t vaani-x:0.2.0-slim .` | A real number to put in the README (today: TBD) |
| 6 | Tiny VPS smoke | `docker compose up` on a 1–2 GB VPS, one TTS + one STT call | Works inside the RAM budget, slots 1–2 |
| 7 | Prices re-checked | Open Sarvam, Google, OpenAI pricing pages | README price table matches the live pages (Nov 2026 snapshot today) |
| 8 | Piper voice licence checked | Read the voice card at its `licence_url` | Licence box in README is correct for every listed voice |

If any gate fails, launch later. A launch with a failing gate is how a project gets its first comment debunking it.

## 2. What NOT to claim until then

- "100% accurate" — never. Measured: WER 0.2899 / CER 0.0811 on 10 clean clips, ~71% of words right.
- "Better than Sarvam" — not measured. Needs Sarvam scored with this normalizer on the same clips.
- A Docker image size, a Pi 4 RTF, or a VPS RTF — all TBD until measured.
- "CI green" — `.github/workflows/ci.yml` is a file, not live CI, until enabled in GitHub.
- That `vaani-tts-piper` is Piper — it is an alias for the installed Hima engine. Piper is a future pack.
- Stars, users, or quotes — none exist yet. Do not invent them.

## 3. README checklist (the Kitten TTS playbook)

Kitten TTS hit #1 on Hacker News with one number in the title ("25MB CPU-Only"), a live demo, and `pip install`. Vaani X's version:

- [ ] One number in the title: "Telugu voice API in 60 MB, runs on a $5-class VPS, OpenAI-compatible" — with the footnote that 60 MB is the Piper-class voice size, not a measured Docker size.
- [ ] 60-second quickstart at the top (README has it).
- [ ] OpenAI SDK snippet showing only `base_url` changes.
- [ ] Benchmark table vs paid APIs on the same test set (or, until the frozen bench exists, the honest 10-clip table labelled as what it is).
- [ ] Docker one-liner + `pip install`.
- [ ] Hardware table (Pi / 2 GB VPS / GPU) with measured RTF — TBDs labelled as TBD.
- [ ] Licence box per model (README has it).
- [ ] Limitations section (README has it — this builds the trust that gets upvotes).
- [ ] Working examples in `examples/` (5 files, all against real endpoints).

## 4. Show HN title options

Pick one. All three carry one number, one price-class, and one compatibility claim — and none claims accuracy that is not measured.

1. "Vaani X — Telugu voice API in 60 MB, runs on a $5 VPS, OpenAI-compatible"
2. "Show HN: Self-host Telugu TTS + STT in 60 MB — drop-in OpenAI replacement"
3. "Vaani X: a Telugu voice API you can self-host on a 1 GB VPS (OpenAI-compatible)"

Post body: one-line claim + the 60-second quickstart + the honest accuracy box + the benchmark table + the limitations + a link to the Hugging Face Space (below). Answer the first hard comment with a measurement, not an adjective.

## 5. Hugging Face Space (10-second listen, no install)

Goal: a visitor hears Telugu in 10 seconds without installing anything.

1. Create a Hugging Face org/space for Vaani X (Gradio or a static demo page).
2. One text box (pre-filled with a Telugu sentence), two voice buttons (Srinivas / Sravani), one Play button, and one upload box for STT.
3. The Space calls the hosted demo server — or, if the demo server is not yet up, say so plainly and link the Docker one-liner instead. Do not fake audio.
4. Link the Space from the README, from the Show HN post, and from `index.html`.
5. The Space is a demo, not a benchmark: keep the same honest accuracy box on it.

## 6. Where to post (in order)

1. GitHub — repo public, README + examples + templates + CI file in place, issues labelled (good first issue: a voice pack, a dialect test set).
2. Hugging Face — Space live, model cards per model (not one mega-repo).
3. Show HN — only after gates 1–6 above pass.
4. Telugu-first communities — Swecha, AI4Bharat, Bhashini channels, and Telugu developer groups. Cross-post the bench table, not the slogan.
5. VoiceFlow users — the `examples/voiceflow_integration.py` flow is the bridge: Vaani X as the Telugu voice behind a broadcast.

## 7. After launch

- Answer every issue with a measurement or a TBD, never a slogan.
- Publish every bench run to `bench/results/` — including the bad numbers. A project that publishes where it struggles gets trusted.
- The next voice pack (Piper Telugu) is the next launch beat: one folder + manifest + sha256, announced as "voices are USB connectors — this one just plugged in".
