"""Greedy CTC collapse."""

from __future__ import annotations


def greedy_ctc(token_ids: list[int], blank_id: int) -> list[int]:
    collapsed: list[int] = []
    previous: int | None = None
    for token_id in token_ids:
        if token_id == blank_id:
            previous = token_id
            continue
        if token_id != previous:
            collapsed.append(token_id)
        previous = token_id
    return collapsed
