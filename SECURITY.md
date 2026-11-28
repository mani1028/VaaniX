# Security Policy

## Reporting a vulnerability

**Do not open a public issue for a security problem.** Email the maintainer (see the repository owner) with a description, the affected version, and steps to reproduce. You will get a reply; if the report is confirmed, a fix is prepared before any public disclosure.

## What counts

- Authentication bypass (for example, calling a keyed route without a key when `TELUGU_SPEECH_AUTH=required`).
- Path traversal or arbitrary file read/write through uploads, voice-pack paths, or `audio_url` handling.
- A voice pack executing code (packs are data — manifest + model + config — and must never run code).
- Denial of service through unbounded uploads, unbounded text, or unbounded engine loading (the slot semaphore and upload caps exist for this reason).

## What does not count

- `local_open` mode skipping the key on loopback. That is documented, development-only behaviour — do not publish that mode.
- The process not terminating TLS and not quota-ing callers. That is by design: put TLS and a rate limit on a proxy (see README).

## Supported versions

Only the latest release (currently 0.2.0) receives security fixes. Older versions are not patched.

## Handling secrets

Never commit real API keys, `.env` files, or credentials. Keys are passed through `TELUGU_API_KEYS` at runtime only. If a key is ever committed by mistake, rotate it immediately and tell the maintainer.
