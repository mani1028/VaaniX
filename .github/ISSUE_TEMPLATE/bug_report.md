---
name: Bug report
about: Something in Vaani X does not work as documented
title: "[bug] "
labels: bug
---

## What happened

A clear description of the bug. What did you do, and what did you expect?

## Server version and hardware

- Vaani X version (e.g. 0.2.0):
- How you run it (pip / Docker / compose):
- Hardware (e.g. Tiny VPS 2 GB, Apple M5, Pi 4):
- Python version:

## Exact request

Paste the exact request (curl, SDK call, or example file). **Never paste a real API key.**

```
# your request here
```

## What you got

Paste the response or error — including the full error shape (`{"error": {...}}` on `/v1/*`).

```
# response / logs here
```

## What you expected

## Did you check?

- [ ] The voice pack is on disk (`models/telugu-asr-micro`, `lite/models/hima-tts-nano-te`)
- [ ] The voice-pack `sha256` is filled in (see README blocker section)
- [ ] `GET /health` returns `ok`
- [ ] The same request works against the other surface (`/api/v1/*` vs `/v1/*`)
