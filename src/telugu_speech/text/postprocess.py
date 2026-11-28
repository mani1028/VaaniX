"""Telugu post-processing, applied AFTER ASR and BEFORE TTS input validation.

Two rules govern everything here:

1. Nothing is silently rewritten. ``postprocess()`` always returns both
   ``raw_text`` (exactly what the recogniser / caller gave, NFC-normalised
   only) and ``corrected_text``. A caller that wants the raw transcript
   uses ``raw_text``; a caller that wants the cleaned one uses
   ``corrected_text`` and can diff the two.
2. No invented corrections. The correction table ships EMPTY. A
   correction may only be added after it is counted on bench data
   (see ``docs/ACCURACY.md``) and the entry records its source.
   ``EXAMPLE_CORRECTIONS`` below are examples of the *shape* of an
   entry, clearly marked as NOT validated and NOT applied by default.

What post-processing does do by default (all measurable by the bench):
NFC normalisation, numeral/punctuation normalisation for scoring (via
``text.normalization.normalize_text``), and preserving Latin code-mixed
words instead of dropping them — the 20 Nov 2026 Andhra check recalled
0 of 53 marked English words, and dropping them is the single biggest
measured error class after script errors.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from telugu_speech.text.normalization import (
    CodeMixSpan,
    NormalizationConfig,
    find_code_mix_spans,
    normalize_text,
    prepare_tts_text,
)
from telugu_speech.text.unicode import is_telugu_char, to_nfc

MAX_TTS_CHARS = 2000
MAX_TTS_CHUNK_CHARS = 180

# Telugu sentence-enders. A chunk boundary may only fall on one of these
# (or on a hard length cap, in which case it falls on a word boundary).
_SENTENCE_END = re.compile(r"[।.?!\u0964]")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Correction:
    """One explicit, bench-validated script correction.

    ``source`` must name the bench set + clip count that justified it,
    e.g. ``"fleurs-te n=200: 14 occurrences, 12 fixed, 0 broken"``.
    An entry without a source is not a correction, it is a guess.
    """

    wrong: str
    right: str
    source: str


# The live correction table. EMPTY by design — see the module docstring.
# Add entries only after counting them on bench data.
CORRECTIONS: tuple[Correction, ...] = ()

# Shape examples only. NOT validated, NOT applied by default. Do not
# move these into CORRECTIONS without bench evidence per entry.
EXAMPLE_CORRECTIONS: tuple[Correction, ...] = (
    Correction(wrong="EXAMPLE", right="EXAMPLE", source="EXAMPLE — not validated, do not apply"),
    Correction(wrong="EXAMPLE", right="EXAMPLE", source="EXAMPLE — not validated, do not apply"),
    Correction(wrong="EXAMPLE", right="EXAMPLE", source="EXAMPLE — not validated, do not apply"),
)


@dataclass(frozen=True)
class PostProcessResult:
    raw_text: str
    corrected_text: str
    normalized_text: str
    code_mix_spans: tuple[CodeMixSpan, ...] = ()
    corrections_applied: tuple[str, ...] = ()

    @property
    def has_code_mix(self) -> bool:
        return bool(self.code_mix_spans)


def apply_corrections(text: str, corrections: tuple[Correction, ...] = CORRECTIONS) -> tuple[str, tuple[str, ...]]:
    """Apply whole-token corrections only. Returns (text, applied wrong-forms)."""
    applied: list[str] = []
    result = text
    for correction in corrections:
        if not correction.source or correction.source.startswith("EXAMPLE"):
            continue
        pattern = re.compile(rf"(?<!\S){re.escape(correction.wrong)}(?!\S)")
        if pattern.search(result):
            result = pattern.sub(correction.right, result)
            applied.append(correction.wrong)
    return result, tuple(applied)


def clean_transcript(text: str, corrections: tuple[Correction, ...] = CORRECTIONS) -> str:
    """Clean one ASR transcript for the API response (and the bench).

    This is the ONLY rewriting the live transcribe path does, and it is
    deliberately small: NFC normalisation (a decomposed vowel sign from
    the decoder becomes its composed form), per-line whitespace collapse,
    empty-line removal, and the (empty-by-default) correction table.
    Sentence-per-line structure from ``accepted_transcript`` is kept:
    lines are joined with ``\n``, never flattened to one line.

    What it does NOT do, on purpose: digits are kept exactly as spoken
    (``5`` stays ``5`` — the bench normalizer expands digits to words
    for SCORING, but the API must return what was said, not an
    expansion); Latin code-mixed words are kept with their casing;
    punctuation is kept. A transcript that is already clean comes back
    unchanged, and the function is idempotent.
    """
    lines: list[str] = []
    for line in to_nfc(text).split("\n"):
        cleaned, _applied = apply_corrections(_WHITESPACE.sub(" ", line).strip(), corrections)
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def postprocess(text: str, corrections: tuple[Correction, ...] = CORRECTIONS) -> PostProcessResult:
    """NFC-normalise, apply the (empty-by-default) correction table, score-normalise.

    Latin tokens are preserved exactly — they are never dropped,
    transliterated, or lower-cased here (scoring lower-cases via the
    normalizer; the raw/corrected text keeps the caller's casing).
    """
    raw = _WHITESPACE.sub(" ", to_nfc(text)).strip()
    corrected, applied = apply_corrections(raw, corrections)
    _raw_norm, normalized = normalize_text(corrected, NormalizationConfig())
    return PostProcessResult(
        raw_text=raw,
        corrected_text=corrected,
        normalized_text=normalized,
        code_mix_spans=find_code_mix_spans(corrected),
        corrections_applied=applied,
    )


@dataclass(frozen=True)
class TextSpan:
    text: str
    script: str  # "telugu" | "latin" | "other"
    start: int
    end: int


def split_telugu_latin_spans(text: str) -> tuple[TextSpan, ...]:
    """Split text into Telugu / Latin / other spans for TTS.

    Latin words are spoken with a Latin fallback instead of being
    mangled by the Telugu voice or dropped. Honest limit: no English
    TTS voice is installed, so the fallback is the same Telugu voice
    reading Latin letters — documented in docs/ACCURACY.md, not hidden.
    The split itself is exact and lossless: concatenating the spans
    reproduces the input.
    """
    spans: list[TextSpan] = []
    current_script: str | None = None
    start = 0
    for index, char in enumerate(text):
        if "A" <= char <= "Z" or "a" <= char <= "z":
            script = "latin"
        elif is_telugu_char(char):
            script = "telugu"
        elif char.isspace() or char.isdigit() or char in "।.?!" :
            script = current_script or "other"
        else:
            script = "other"
        if current_script is None:
            current_script, start = script, index
        elif script != current_script:
            spans.append(TextSpan(text[start:index], current_script, start, index))
            current_script, start = script, index
    if current_script is not None:
        spans.append(TextSpan(text[start:], current_script, start, len(text)))
    return tuple(spans)


def chunk_for_tts(text: str, max_chars: int = MAX_TTS_CHUNK_CHARS) -> list[str]:
    """Split spoken Telugu text into <=180-char chunks.

    Boundaries fall on Telugu sentence-enders first, then on word
    boundaries — never mid-word. Numbers/dates/money are expanded via
    the existing ``prepare_tts_text`` (lakh/crore, ₹, ordinals); that
    logic is delegated, not re-implemented here. Raises ValueError on
    empty input and on text over 2000 chars, naming the limit.
    """
    if not text or not text.strip():
        raise ValueError("empty text: nothing to speak")
    if len(text) > MAX_TTS_CHARS:
        raise ValueError(f"text is {len(text)} chars, max is {MAX_TTS_CHARS}")
    if max_chars < 1 or max_chars > MAX_TTS_CHARS:
        raise ValueError(f"max_chars must be 1..{MAX_TTS_CHARS}")
    spoken = prepare_tts_text(text)
    chunks: list[str] = []
    current = ""
    for sentence in _split_sentences(spoken):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip() if current else sentence
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(sentence) <= max_chars:
            current = sentence
            continue
        # One long sentence: split on word boundaries only.
        for word in sentence.split(" "):
            if not word:
                continue
            if len(current) + len(word) + 1 <= max_chars:
                current = f"{current} {word}".strip() if current else word
            else:
                if current:
                    chunks.append(current)
                current = word
    if current:
        chunks.append(current)
    return chunks


def _split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        parts.append(text[start : match.end()])
        start = match.end()
    if start < len(text):
        parts.append(text[start:])
    return parts
