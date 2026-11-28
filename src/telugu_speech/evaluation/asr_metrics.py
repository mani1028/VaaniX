"""Word, character, and akshara error rates.

Telugu is agglutinative: case, plural, and postpositions fuse into one long
word, so one wrong matra voids the whole word. That is why WER for Telugu
runs roughly 3-4x the akshara error rate on the same transcript. Report
WER + CER + AER together; CER/AER are the more informative headlines for
Telugu, and WER alone understates how readable a transcript is.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from telugu_speech.text.normalization import tokens


@dataclass(frozen=True)
class ErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def rate(self) -> float:
        if self.reference_length == 0:
            return 0.0 if self.insertions == 0 else 1.0
        return (self.substitutions + self.deletions + self.insertions) / self.reference_length


def edit_counts(reference: list[str], hypothesis: list[str]) -> ErrorCounts:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    distance = [[0] * cols for _ in range(rows)]
    operations = [[(0, 0, 0)] * cols for _ in range(rows)]
    for i in range(1, rows):
        distance[i][0] = i
        operations[i][0] = (0, i, 0)
    for j in range(1, cols):
        distance[0][j] = j
        operations[0][j] = (0, 0, j)
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
                operations[i][j] = operations[i - 1][j - 1]
                continue
            candidates = (
                (distance[i - 1][j - 1] + 1, _add(operations[i - 1][j - 1], 1, 0, 0)),
                (distance[i - 1][j] + 1, _add(operations[i - 1][j], 0, 1, 0)),
                (distance[i][j - 1] + 1, _add(operations[i][j - 1], 0, 0, 1)),
            )
            distance[i][j], operations[i][j] = min(candidates, key=lambda item: item[0])
    substitutions, deletions, insertions = operations[-1][-1]
    return ErrorCounts(substitutions, deletions, insertions, len(reference))


def _add(counts: tuple[int, int, int], substitutions: int, deletions: int, insertions: int) -> tuple[int, int, int]:
    return (
        counts[0] + substitutions,
        counts[1] + deletions,
        counts[2] + insertions,
    )


def word_error_rate(reference: str, hypothesis: str) -> ErrorCounts:
    return edit_counts(tokens(reference), tokens(hypothesis))


def character_error_rate(reference: str, hypothesis: str) -> ErrorCounts:
    ref_chars = [char for char in reference if not char.isspace()]
    hyp_chars = [char for char in hypothesis if not char.isspace()]
    return edit_counts(ref_chars, hyp_chars)


_VIRAMA = "\u0c4d"
_DIGIT = re.compile(r"[0-9\u0c66-\u0c6f]")
_LATIN_TOKEN = re.compile(r"[A-Za-z]")


def akshara_units(text: str) -> list[str]:
    """Split text into aksharas (grapheme clusters).

    A cluster is a base character plus its following combining marks, and a
    virama-joined consonant stays in the same cluster (e.g. ``క్ష`` is one
    akshara, not three characters). Non-Telugu characters are one unit each.
    Whitespace is dropped, matching :func:`character_error_rate`.
    """
    units: list[str] = []
    current = ""
    join_next = False
    for char in text:
        if char.isspace():
            continue
        if not current:
            current = char
            join_next = char == _VIRAMA
            continue
        if join_next or unicodedata.category(char).startswith("M"):
            current += char
            join_next = char == _VIRAMA
            continue
        if char == _VIRAMA:
            current += char
            join_next = True
            continue
        units.append(current)
        current = char
        join_next = False
    if current:
        units.append(current)
    return units


def akshara_error_rate(reference: str, hypothesis: str) -> ErrorCounts:
    """Akshara error rate: edit distance over akshara units, not characters."""
    return edit_counts(akshara_units(reference), akshara_units(hypothesis))


def micro_average(counts: list[ErrorCounts]) -> float | None:
    """Micro-average rate over reference units. None when there is nothing."""
    if not counts:
        return None
    total_errors = sum(item.substitutions + item.deletions + item.insertions for item in counts)
    total_ref = sum(item.reference_length for item in counts)
    if total_ref == 0:
        return 0.0 if total_errors == 0 else 1.0
    return total_errors / total_ref


@dataclass(frozen=True)
class SegmentScores:
    """Word error split by token class, so one bad class cannot hide.

    ``numerals`` and ``latin`` are None when neither side has such a token.
    Numbers and Latin code-mix words are scored separately because a model
    can be good at Telugu words and still drop every English word (the
    20 Nov 2026 Andhra check recalled 0 of 53 marked English words).
    """

    words: ErrorCounts
    numerals: ErrorCounts | None
    latin: ErrorCounts | None


def _segment_counts(reference: list[str], hypothesis: list[str], pattern: re.Pattern[str]) -> ErrorCounts | None:
    ref = [token for token in reference if pattern.search(token)]
    hyp = [token for token in hypothesis if pattern.search(token)]
    if not ref and not hyp:
        return None
    return edit_counts(ref, hyp)


def segment_scores(reference: str, hypothesis: str) -> SegmentScores:
    ref_tokens = tokens(reference)
    hyp_tokens = tokens(hypothesis)
    return SegmentScores(
        words=edit_counts(ref_tokens, hyp_tokens),
        numerals=_segment_counts(ref_tokens, hyp_tokens, _DIGIT),
        latin=_segment_counts(ref_tokens, hyp_tokens, _LATIN_TOKEN),
    )
