"""Turn a research transcript into words that can be scored."""

from __future__ import annotations

import re

from telugu_speech.text.unicode import is_telugu_char

_TAGS = re.compile(
    r"</?(?:noise|pause|talking|static_noise|child_yelling|phone_ringing)>",
    re.IGNORECASE,
)
_BRACKETS = re.compile(r"\[[^\]]*\]")
_BRACES = re.compile(r"\{([^}]*)\}")
_TOKEN_EDGE = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$")


def research_reference(text: str) -> str:
    """Drop annotator tags. Keep an English word that was marked in braces."""
    cleaned = _TAGS.sub(" ", text)
    cleaned = _BRACKETS.sub(" ", cleaned)
    cleaned = _BRACES.sub(r"\1", cleaned)
    cleaned = cleaned.replace("--", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def marked_english_words(transcript: str) -> list[str]:
    """English words the annotator placed in braces, in order."""
    words: list[str] = []
    for match in _BRACES.finditer(transcript):
        for part in match.group(1).split():
            cleaned = _TOKEN_EDGE.sub("", part).lower()
            if cleaned:
                words.append(cleaned)
    return words


def telugu_only(text: str) -> str:
    """Keep tokens that contain at least one Telugu letter."""
    return " ".join(token for token in text.split() if any(is_telugu_char(char) for char in token))


def english_hits(expected: list[str], hypothesis: str) -> int:
    """Count marked English words that appear as whole tokens in the hypothesis."""
    available = [token.lower() for token in hypothesis.split()]
    hits = 0
    for word in expected:
        if word in available:
            available.remove(word)
            hits += 1
    return hits


def whole_token_present(word: str, vocabulary: set[str]) -> bool:
    """A whole-token hit. A missing word can still be built from smaller pieces."""
    return any(form in vocabulary for form in (word, f" {word}", f"Ġ{word}", f"▁{word}"))
