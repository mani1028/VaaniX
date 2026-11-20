"""Telugu text normalization.

Raw text keeps punctuation. Normalized text is the evaluation string.
Latin spans stay in Latin; they are tagged, not transliterated.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from telugu_speech.text.unicode import is_telugu_char, telugu_digits_to_ascii, to_nfc

DIGIT_POLICIES = ("keep", "ascii", "spoken")

_ONES = (
    "సున్నా",
    "ఒకటి",
    "రెండు",
    "మూడు",
    "నాలుగు",
    "ఐదు",
    "ఆరు",
    "ఏడు",
    "ఎనిమిది",
    "తొమ్మిది",
)
_TEENS = {
    10: "పది",
    11: "పదకొండు",
    12: "పన్నెండు",
    13: "పదమూడు",
    14: "పద్నాలుగు",
    15: "పదిహేను",
    16: "పదహారు",
    17: "పదిహేడు",
    18: "పద్దెనిమిది",
    19: "పందొమ్మిది",
}
_TENS = {
    2: "ఇరవై",
    3: "ముప్పై",
    4: "నలభై",
    5: "యాభై",
    6: "అరవై",
    7: "డెబ్బై",
    8: "ఎనభై",
    9: "తొంభై",
}
_LATIN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_INTEGER = re.compile(r"\d+")
_PUNCT = re.compile(r"[^\w\s\u0c00-\u0c7f]", re.UNICODE)


@dataclass(frozen=True)
class NormalizationConfig:
    digit_policy: str = "spoken"
    lowercase_latin: bool = True
    strip_punctuation_in_norm: bool = True

    def __post_init__(self) -> None:
        if self.digit_policy not in DIGIT_POLICIES:
            raise ValueError(f"digit_policy must be one of {DIGIT_POLICIES}")


@dataclass(frozen=True)
class CodeMixSpan:
    start: int
    end: int
    text: str


_LARGEST_WORD = 99_99_99_999
_ORDINALS = {
    1: "మొదటి",
    2: "రెండవ",
    3: "మూడవ",
    4: "నాలుగవ",
    5: "ఐదవ",
    6: "ఆరవ",
    7: "ఏడవ",
    8: "ఎనిమిదవ",
    9: "తొమ్మిదవ",
    10: "పదవ",
    11: "పదకొండవ",
    12: "పన్నెండవ",
    13: "పదమూడవ",
    14: "పద్నాలుగవ",
    15: "పదిహేనవ",
    16: "పదహారవ",
    17: "పదిహేడవ",
    18: "పద్దెనిమిదవ",
    19: "పందొమ్మిదవ",
    20: "ఇరవయ్యవ",
}
_MONTHS = (
    "జనవరి",
    "ఫిబ్రవరి",
    "మార్చి",
    "ఏప్రిల్",
    "మే",
    "జూన్",
    "జూలై",
    "ఆగస్టు",
    "సెప్టెంబర్",
    "అక్టోబర్",
    "నవంబర్",
    "డిసెంబర్",
)


def _below_hundred(value: int) -> str:
    if value < 10:
        return _ONES[value]
    if value < 20:
        return _TEENS[value]
    tens, ones = divmod(value, 10)
    if ones == 0:
        return _TENS[tens]
    return f"{_TENS[tens]} {_ONES[ones]}"


def _below_thousand(value: int) -> str:
    if value < 100:
        return _below_hundred(value)
    hundreds, rest = divmod(value, 100)
    head = "వంద" if hundreds == 1 else f"{_ONES[hundreds]} వందలు"
    if rest == 0:
        return head
    return f"{head} {_below_hundred(rest)}"


def speak_digits(value: int) -> str:
    return " ".join(_ONES[int(digit)] for digit in str(value))


def speak_integer(value: int) -> str:
    """Speak 0 through 99,99,99,999 with lakh and crore. Longer values are digit by digit."""
    if value < 0:
        raise ValueError("negative integers are not expanded")
    if value > _LARGEST_WORD:
        return speak_digits(value)
    crores, rest = divmod(value, 10_000_000)
    lakhs, rest = divmod(rest, 100_000)
    thousands, rest = divmod(rest, 1_000)
    parts: list[str] = []
    if crores:
        parts.append("కోటి" if crores == 1 else f"{_below_thousand(crores)} కోట్లు")
    if lakhs:
        parts.append("లక్ష" if lakhs == 1 else f"{_below_thousand(lakhs)} లక్షలు")
    if thousands:
        parts.append("వెయ్యి" if thousands == 1 else f"{_below_thousand(thousands)} వేలు")
    if rest or not parts:
        parts.append(_below_thousand(rest))
    return " ".join(parts)


def _apply_digits(text: str, policy: str) -> str:
    if policy == "keep":
        return text
    ascii_text = telugu_digits_to_ascii(text)
    if policy == "ascii":
        return ascii_text

    def replace(match: re.Match[str]) -> str:
        return speak_integer(int(match.group(0)))

    return _INTEGER.sub(replace, ascii_text)


def normalize_text(text: str, config: NormalizationConfig | None = None) -> tuple[str, str]:
    """Return (raw_text, norm_text)."""
    chosen = config or NormalizationConfig()
    raw = re.sub(r"\s+", " ", to_nfc(text)).strip()
    norm = _apply_digits(raw, chosen.digit_policy)
    if chosen.lowercase_latin:
        norm = "".join(char.lower() if "A" <= char <= "Z" else char for char in norm)
    if chosen.strip_punctuation_in_norm:
        norm = _PUNCT.sub(" ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    return raw, norm


def find_code_mix_spans(text: str) -> tuple[CodeMixSpan, ...]:
    return tuple(
        CodeMixSpan(match.start(), match.end(), match.group(0))
        for match in _LATIN.finditer(text)
    )


def has_code_mix(text: str) -> bool:
    return bool(find_code_mix_spans(text))


_DATE = re.compile(r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?!\d)")
_TIME = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")
_RUPEE = re.compile(r"₹\s*(\d{1,3}(?:,\d{2})*,\d{3}|\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?")
_PERCENT = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)\s*%")
_ORDINAL = re.compile(r"(?<!\d)(\d{1,2})వ")
_GROUPED = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{2})*,\d{3}|\d{1,3}(?:,\d{3})+)(?!\d)")
_DECIMAL = re.compile(r"(?<!\d)(\d+)\.(\d+)(?!\d)")
_SIGNED = re.compile(r"(?<![\d.])-(\d+)(?!\d)")
_PLAIN_INTEGER = re.compile(r"(?<!\d)\d+(?!\d)")


def _grouped_int(token: str) -> int:
    return int(token.replace(",", ""))


def _speak_decimal(whole: str, fraction: str) -> str:
    digits = " ".join(_ONES[int(digit)] for digit in fraction)
    return f"{speak_integer(int(whole))} పాయింట్ {digits}"


def _speak_money(amount: int, fraction: str | None) -> str:
    paise = None
    if fraction is not None:
        paise = int(fraction) * 10 if len(fraction) == 1 else int(fraction)
    parts: list[str] = []
    if amount or paise in {None, 0}:
        if amount == 1:
            parts.append("ఒక రూపాయి")
        else:
            parts.append(f"{speak_integer(amount)} రూపాయలు")
    if paise:
        unit = "పైసా" if paise == 1 else "పైసలు"
        parts.append(f"{speak_integer(paise)} {unit}")
    return " ".join(parts)


def _with_suffix(match: re.Match[str], spoken: str) -> str:
    following = match.string[match.end():]
    if following and is_telugu_char(following[0]):
        return f"{spoken} "
    return spoken


def _replace_date(match: re.Match[str]) -> str:
    day, month, year = (int(match.group(index)) for index in (1, 2, 3))
    if not (1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2099):
        return match.group(0)
    spoken = f"{speak_integer(day)} {_MONTHS[month - 1]} {speak_integer(year)}"
    return _with_suffix(match, spoken)


def _replace_time(match: re.Match[str]) -> str:
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23 or minute > 59:
        return match.group(0)
    parts: list[str] = []
    if hour == 1:
        parts.append("ఒక గంట")
    elif hour:
        parts.append(f"{speak_integer(hour)} గంటలు")
    if minute == 1:
        parts.append("ఒక నిమిషం")
    elif minute:
        parts.append(f"{speak_integer(minute)} నిమిషాలు")
    return _with_suffix(match, " ".join(parts) or "సున్నా గంటలు")


def prepare_tts_text(text: str) -> str:
    """Read numbers, money, dates, times, and ordinals as Telugu words. Latin words stay Latin."""
    spoken = telugu_digits_to_ascii(to_nfc(text))
    spoken = _DATE.sub(_replace_date, spoken)
    spoken = _TIME.sub(_replace_time, spoken)

    def rupee(match: re.Match[str]) -> str:
        return _with_suffix(match, _speak_money(_grouped_int(match.group(1)), match.group(2)))

    def percent(match: re.Match[str]) -> str:
        token = match.group(1)
        if "." in token:
            whole, fraction = token.split(".", 1)
            amount = _speak_decimal(whole, fraction)
        else:
            amount = speak_integer(int(token))
        return _with_suffix(match, f"{amount} శాతం")

    def ordinal(match: re.Match[str]) -> str:
        value = int(match.group(1))
        spoken = _ORDINALS.get(value, f"{speak_integer(value)}వ")
        return _with_suffix(match, spoken)

    spoken = _RUPEE.sub(rupee, spoken)
    spoken = _PERCENT.sub(percent, spoken)
    spoken = _ORDINAL.sub(ordinal, spoken)
    spoken = _GROUPED.sub(lambda match: _with_suffix(match, speak_integer(_grouped_int(match.group(1)))), spoken)
    spoken = _DECIMAL.sub(lambda match: _with_suffix(match, _speak_decimal(match.group(1), match.group(2))), spoken)
    spoken = _SIGNED.sub(lambda match: _with_suffix(match, f"మైనస్ {speak_integer(int(match.group(1)))}"), spoken)
    spoken = _PLAIN_INTEGER.sub(lambda match: _with_suffix(match, speak_integer(int(match.group(0)))), spoken)
    return re.sub(r"\s+", " ", spoken).strip()


def contains_telugu(text: str) -> bool:
    return any(is_telugu_char(char) for char in text)


def tokens(text: str) -> list[str]:
    return [part for part in text.split(" ") if part]


def iter_config_fields(config: NormalizationConfig) -> Iterable[tuple[str, object]]:
    return (
        ("digit_policy", config.digit_policy),
        ("lowercase_latin", config.lowercase_latin),
        ("strip_punctuation_in_norm", config.strip_punctuation_in_norm),
    )
