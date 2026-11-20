# Engineering policy

Policy version: 2.0 of the SaaS Engineering Rulebook at `/Users/visys/.saas-engineering/SaaS_Engineering_Rulebook.md`.

Project: Telugu AI Speech Lab. Risk: moderate. The API accepts audio uploads and can call large models, but Phase 1 has no accounts, database, or multi-tenant store.

Owners: the repository maintainer covers technical, security, data, and operations roles until a second maintainer exists. High-risk release still needs a reviewer who is not the author. This file does not record that review.

## Stack

Python 3.11+, FastAPI, NumPy, PyYAML. Optional NeMo and faster-whisper. No database.

## Identity

Bearer tokens from `TELUGU_API_KEYS` when `TELUGU_SPEECH_AUTH=required` (the default). Missing keys fail closed. `local_open` is an explicit developer override.

## Applicable deviations

- No database, so migration and tenant-isolation rules are not applicable. Manifest speaker splits are the data invariant that is tested.
- No web UI in Phase 1, so WCAG checks are not applicable yet. `apps/web-dashboard` is a placeholder note, not a site.
- Training is intentionally refused. That is the phase boundary, not a silent success path.

## Checks

`.engineering/quality-gates.json` runs compile, AST checks, pytest, a secret pattern scan, and dependency bound checks. Passing those checks is not a production-readiness claim.
