"""Telugu Unicode helpers."""

from __future__ import annotations

import unicodedata

TELUGU_BLOCK = range(0x0C00, 0x0C80)
TELUGU_DIGITS = {chr(0x0C66 + value): str(value) for value in range(10)}
VIRAMA = "\u0c4d"


def to_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def telugu_digits_to_ascii(text: str) -> str:
    return "".join(TELUGU_DIGITS.get(char, char) for char in text)


def is_telugu_char(char: str) -> bool:
    return len(char) == 1 and ord(char) in TELUGU_BLOCK
