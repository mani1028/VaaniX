## What changed

## Why

## What was measured (or TBD)

Paste the command, dataset, machine, and date for every number you claim. Unmeasured = TBD, written as TBD. Do not claim Sarvam parity, a Docker size, or an RTF you did not measure.

- [ ] `pytest -q` passes (paste the result, or explain the failure)
- [ ] Bench run attached (if this touches ASR/TTS quality), or N/A with a reason

## PR gate

- [ ] No WER regression over 2 points absolute on FLEURS-te (bench command in CONTRIBUTING.md; until CI is enabled, paste the bench result here)
- [ ] Old `/api/v1/*` routes still work (smoke curls in README)
- [ ] No new dependency without a reason in the description
- [ ] No secrets, real keys, model weights, or generated WAV files committed
- [ ] Docs updated if behaviour changed (README / docs/API_V2.md / docs/ARCHITECTURE.md as needed)

## What I did NOT claim

Write here, in one line, what this PR does not prove (for example: "Docker size not measured", "Pi 4 RTF not measured").
