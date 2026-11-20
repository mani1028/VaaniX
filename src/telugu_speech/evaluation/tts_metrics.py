"""Runtime measurements shared by ASR and TTS reports."""

from __future__ import annotations

from pathlib import Path
import resource
import sys
import time


def peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(usage)
    return int(usage) * 1024


def file_size_bytes(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    return path.stat().st_size


class Stopwatch:
    def __init__(self) -> None:
        self.elapsed_s = 0.0

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed_s = time.perf_counter() - self._start


def real_time_factor(elapsed_s: float, audio_duration_s: float) -> float | None:
    if audio_duration_s <= 0:
        return None
    return elapsed_s / audio_duration_s
