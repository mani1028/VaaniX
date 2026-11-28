"""Vaani X core text pipeline — Telugu-first, code-mixed aware.

Order of operations (do not reorder — each step assumes the previous):
  1. NFC normalisation (``unicodedata.normalize``)
  2. Validation: empty / whitespace-only / oversize -> OpenAI-shaped error
  3. Telugu punctuation normalisation (danda, commas, quotes)
  4. Sentence splitting on Telugu + Latin sentence terminators
  5. Number / date / currency expansion, in Telugu words
  6. Chunking for long text, always on a sentence boundary
  7. English code-mixed words are KEPT in Latin script and passed through
     unchanged — the previous product dropped them; dropping a word the
     user typed is a data-loss bug, not a normalisation choice.

No step here needs a model, a network call, or a voice engine. This module
is pure stdlib and is unit-testable without any model files present.

Numbers: Telugu has its own digits (౦-౯) and the expansion below handles
both Telugu and ASCII digits. Expansion is deliberately simple and
deterministic (no LLM, no locale library): cardinals up to 999,999,
years, dates (DD/MM/YYYY and DD-MM-YYYY), times (HH:MM), and ₹ / Rs.
amounts. Anything it cannot expand safely is left as-is, never dropped.
"""

from __future__ import annotations

import re
import unicodedata

from .config import settings
from .errors import TextValidationError

# ---------------------------------------------------------------------------
# Telugu digits and number words
# ---------------------------------------------------------------------------

_TELUGU_DIGITS = "౦౧౨౩౪౫౬౭౮౯"
_ASCII_TO_TELUGU = str.maketrans("0123456789", _TELUGU_DIGITS)

_ONES = [
    "", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు",
    "ఎనిమిది", "తొమ్మిది", "పది", "పదకొండు", "పన్నెండు", "పదమూడు",
    "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు", "పద్దెనిమిది", "పంతొమ్మిది",
]
_TENS = [
    "", "", "ఇరవై", "ముప్పై", "నలభై", "యాభై", "అరవై", "డెబ్బై",
    "ఎనభై", "తొంభై",
]


def _under_100(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (" " + _ONES[ones] if ones else "")


def _under_1000(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    out = f"{_ONES[hundreds]} వందలు" if hundreds else ""
    if rest:
        out = (out + " " if out else "") + _under_100(rest)
    return out or "సున్నా"


def _legacy_speak_integer(n: int):
    """The project's own number speaker, if importable (preferred).

    ``telugu_speech.text.normalization.speak_integer`` is the authority:
    it says వంద for 100 (not "ఒకటి వందలు"), వెయ్యి for 1,000, and
    handles lakh / crore up to 99,99,99,999. The local fallback below
    exists only so this module stays importable without the text package.
    Returns None when the legacy module is not importable.
    """
    try:
        from telugu_speech.text.normalization import speak_integer
    except ImportError:
        return None
    return speak_integer(n)


def number_to_telugu(n: int) -> str:
    """Cardinal number -> Telugu words.

    Delegates to the project's ``speak_integer`` when available:
    0 -> సున్నా, 5 -> ఐదు, 21 -> ఇరవై ఒకటి, 100 -> వంద,
    1000 -> వెయ్యి, 2026 -> రెండు వేలు ఇరవై ఆరు.
    Out-of-range values are returned as Telugu digits, never dropped.
    """
    legacy = _legacy_speak_integer(n) if n >= 0 else None
    if legacy is not None:
        return legacy
    if n == 0:
        return "సున్నా"
    if n < 0 or n > 999_999:
        return str(abs(n)).translate(_ASCII_TO_TELUGU)
    thousands, rest = divmod(n, 1000)
    parts: list[str] = []
    if thousands:
        parts.append(f"{_under_1000(thousands)} వేలు")
    if rest:
        parts.append(_under_1000(rest))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

# Telugu danda (। U+0964) and double danda (॥ U+0965) are sentence ends in
# Telugu text; normalise the ASCII pipe misuse and keep the danda.
_DANDA = "।"
_SENTENCE_END = re.compile(r"([।.!?]+)\s+")
# Telugu + Latin sentence terminators, for splitting.
_SPLIT_RE = re.compile(r"(?<=[।.!?])\s+(?=[\u0c00-\u0c7fA-Za-z0-9\"'“‘(])")

# Dates: 21/11/2026, 21-11-2026, 21.11.2026
_DATE_RE = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b")
# Times: 14:30, 9:05
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
# Currency: ₹1,250 / ₹1250 / Rs. 1250 / రూ. 1250
_CURRENCY_RE = re.compile(r"(?:₹|Rs\.?|రూ\.?)\s*([\d,]+(?:\.\d{1,2})?)")
# Plain integers (ASCII or Telugu digits), 1-6 digits, not part of a date/time.
_NUMBER_RE = re.compile(r"(?<![\d౦-౯:./-])(\d{1,6})(?![\d౦-౯:./-])")

_TELUGU_MONTHS = [
    "", "జనవరి", "ఫిబ్రవరి", "మార్చి", "ఏప్రిల్", "మే", "జూన్",
    "జూలై", "ఆగస్టు", "సెప్టెంబర్", "అక్టోబర్", "నవంబర్", "డిసెంబర్",
]


def normalize_text(text: str) -> str:
    """NFC-normalise and tidy whitespace/punctuation. Does not drop content."""
    if text is None:
        raise TextValidationError("text must be a non-empty string")
    text = unicodedata.normalize("NFC", str(text))
    # Collapse runs of whitespace (including newlines) to a single space,
    # but keep sentence boundaries intact.
    text = re.sub(r"\s+", " ", text).strip()
    # Normalise curly quotes to straight ones the TTS front-end handles.
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return text


def validate_text(text: str, *, max_chars: int | None = None) -> str:
    """Validate + normalise. Returns the normalised text.

    Raises TextValidationError (OpenAI-shaped, HTTP 400) for:
      * empty / whitespace-only text
      * text longer than ``max_chars`` (default VAANI_MAX_TEXT_CHARS)
    """
    limit = settings.max_text_chars if max_chars is None else max_chars
    cleaned = normalize_text(text)
    if not cleaned:
        raise TextValidationError("text must not be empty")
    if len(cleaned) > limit:
        raise TextValidationError(
            f"text is {len(cleaned)} characters, over the limit of {limit}; "
            "split it into smaller requests or use streaming"
        )
    return cleaned


def split_sentences(text: str) -> list[str]:
    """Split normalised text into sentences on Telugu/Latin terminators.

    Keeps the terminator attached to its sentence. A text with no
    terminator is returned as a single sentence. Never returns empty strings.
    """
    text = normalize_text(text)
    if not text:
        return []
    parts = _SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Number / date / currency expansion
# ---------------------------------------------------------------------------

def _expand_date(match: re.Match) -> str:
    day, month, year = (int(match.group(i)) for i in (1, 2, 3))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return match.group(0)  # not a real date — leave it alone
    return (
        f"{number_to_telugu(day)} {_TELUGU_MONTHS[month]} "
        f"{number_to_telugu(year)}"
    )


def _expand_time(match: re.Match) -> str:
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23 or minute > 59:
        return match.group(0)
    if minute == 0:
        return f"{number_to_telugu(hour)} గంటలు"
    return f"{number_to_telugu(hour)} గంటల {number_to_telugu(minute)} నిమిషాలు"


def _expand_currency(match: re.Match) -> str:
    raw = match.group(1).replace(",", "")
    try:
        amount = float(raw)
    except ValueError:
        return match.group(0)
    whole = int(amount)
    words = f"{number_to_telugu(whole)} రూపాయలు"
    paise = round((amount - whole) * 100)
    if paise:
        words += f" {number_to_telugu(paise)} పైసలు"
    return words


def _expand_number(match: re.Match) -> str:
    return number_to_telugu(int(match.group(1)))


def expand_numbers(text: str) -> str:
    """Expand dates, times, currency and plain integers to Telugu words.

    Delegates to the project's ``prepare_tts_text`` when importable —
    it already handles dates, times, ₹ amounts (incl. Indian grouped
    commas), decimals, percent, ordinals, signed numbers and lakh/crore,
    and it keeps Latin code-mixed words in Latin. The local regex path
    below is only a fallback for when the text package is unavailable.
    English words are untouched either way. Unparseable tokens are left
    as-is, never dropped.
    """
    try:
        from telugu_speech.text.normalization import prepare_tts_text

        return prepare_tts_text(text)
    except ImportError:
        pass
    text = _DATE_RE.sub(_expand_date, text)
    text = _TIME_RE.sub(_expand_time, text)
    text = _CURRENCY_RE.sub(_expand_currency, text)
    text = _NUMBER_RE.sub(_expand_number, text)
    return text


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, *, max_chars: int | None = None) -> list[str]:
    """Split validated text into synthesis chunks.

    Chunks break on sentence boundaries first; a single sentence longer
    than ``max_chars`` is split on commas, then on spaces, as a last
    resort. No chunk exceeds ``max_chars`` unless a single word does.
    Returns at least one chunk for non-empty input.
    """
    limit = settings.max_chunk_chars if max_chars is None else max_chars
    cleaned = validate_text(text)
    expanded = expand_numbers(cleaned)
    sentences = split_sentences(expanded)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        # Sentence itself is too long: split it greedily on commas/spaces.
        while len(sentence) > limit:
            cut = sentence.rfind(",", 0, limit)
            if cut <= 0:
                cut = sentence.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            chunks.append(sentence[:cut].strip(" ,"))
            sentence = sentence[cut:].strip(" ,")
        current = sentence
    if current:
        chunks.append(current)
    return chunks


def prepare_text(text: str) -> list[str]:
    """The single entry point the TTS path uses: validate -> expand -> chunk."""
    return chunk_text(text)


def contains_telugu(text: str) -> bool:
    """True if the text contains any Telugu-script character (U+0C00-U+0C7F)."""
    return any("\u0c00" <= ch <= "\u0c7f" for ch in text)


def contains_latin(text: str) -> bool:
    """True if the text contains any Latin letter (code-mixed English)."""
    return any("a" <= ch.lower() <= "z" for ch in text if ch.isalpha())
