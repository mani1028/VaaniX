"""Word and character error rates."""

from __future__ import annotations

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
