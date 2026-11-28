---
name: Voice request
about: Ask for a new Telugu voice pack (or offer one)
title: "[voice] "
labels: voice
---

## Voice

- Name / speaker:
- Gender (male / female / neutral):
- Language: te (Telugu only in v2)
- Engine (hima-nano / piper / onnx-vits / other):

## Source

- Link to the voice card / model page:
- Licence (SPDX id or short name):
- Licence URL:

A voice is a folder with a `manifest.yaml` — see `docs/VOICE_PACK.md`. The licence is the operator's attestation; the core trusts it but does not verify it against the voice card.

## Licence check (required)

- [ ] I have read the voice card and the licence above is what it says.
- [ ] It is NOT XTTS v2 (CPML, non-commercial) or MMS (CC-BY-NC) offered as a commercial-facing voice.
- [ ] The model files are publicly downloadable, or I can provide them.

## Why this voice?

What does it add — a dialect (Telangana / Coastal Andhra / Rayalaseema), a style, a quality step up from the shipped Hima voices?

## Offer to contribute

- [ ] I can provide the pack (folder + manifest + sha256) as a PR, with no weights committed (weights stay out of git — see CONTRIBUTING.md).
