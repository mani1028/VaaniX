"""Loaded speech engines.

One voice pack is resident at a time. While requests overlap, that pack can
grow to TELUGU_SPEECH_SLOTS exclusive copies so they do not share one ONNX
runtime. When a request finishes and another copy is already idle, the extra
copy is dropped. A failed load is not stored.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import gc
import os
import queue
import threading

_ENGINES: dict[tuple[object, ...], object] = {}
_LOCK = threading.Condition()
_VOICE_PACKS = frozenset({"hima-int8", "indic-tiny"})
_POOLS: dict[tuple[object, ...], _Pool] = {}


class _Pool:
    def __init__(self) -> None:
        self.engines: list[object] = []
        self.free: queue.Queue[object] = queue.Queue()
        self.busy = 0
        self.building = 0


def speech_slots() -> int:
    raw = os.environ.get("TELUGU_SPEECH_SLOTS", "2")
    try:
        slots = int(raw)
    except ValueError:
        return 2
    return min(4, max(1, slots))


def shared_engine(key: tuple[object, ...], loader: Callable[[], object]) -> object:
    """Return the single engine for key. Used by tests and by models that stay one copy."""
    with _LOCK:
        engine = _ENGINES.get(key)
        if engine is not None:
            return engine
        if key and key[0] in _VOICE_PACKS:
            for old in [item for item in _ENGINES if item and item[0] in _VOICE_PACKS]:
                del _ENGINES[old]
            gc.collect()
        engine = loader()
        _ENGINES[key] = engine
        return engine


@contextmanager
def borrow_engine(key: tuple[object, ...], loader: Callable[[], object]) -> Iterator[object]:
    """Exclusive engine for one request. A second request gets another copy, up to the slot cap."""
    engine = _acquire(key, loader)
    try:
        yield engine
    finally:
        _release(key, engine)


def _acquire(key: tuple[object, ...], loader: Callable[[], object]) -> object:
    pack = key[0] if key else None
    while True:
        with _LOCK:
            if pack in _VOICE_PACKS:
                others = [
                    slot
                    for other, slot in _POOLS.items()
                    if other and other[0] in _VOICE_PACKS and other[0] != pack and (slot.busy or slot.building)
                ]
                if others:
                    _LOCK.wait(timeout=0.05)
                    continue
                for other in [item for item in _POOLS if item and item[0] in _VOICE_PACKS and item[0] != pack]:
                    del _POOLS[other]
                gc.collect()
            pool = _POOLS.setdefault(key, _Pool())
            try:
                engine = pool.free.get_nowait()
            except queue.Empty:
                engine = None
            if engine is not None:
                pool.busy += 1
                return engine
            if len(pool.engines) + pool.building < speech_slots():
                pool.building += 1
            else:
                _LOCK.wait(timeout=0.05)
                continue
        try:
            engine = loader()
        except Exception:
            with _LOCK:
                pool.building -= 1
                _LOCK.notify_all()
            raise
        with _LOCK:
            pool.building -= 1
            pool.engines.append(engine)
            pool.busy += 1
            return engine


def _release(key: tuple[object, ...], engine: object) -> None:
    dropped = False
    with _LOCK:
        pool = _POOLS.get(key)
        if pool is None:
            return
        pool.busy -= 1
        if pool.free.empty():
            pool.free.put(engine)
        else:
            pool.engines.remove(engine)
            dropped = True
        _LOCK.notify_all()
    if dropped:
        del engine
        gc.collect()


def clear_shared_engines() -> None:
    with _LOCK:
        _ENGINES.clear()
        _POOLS.clear()
        _LOCK.notify_all()
