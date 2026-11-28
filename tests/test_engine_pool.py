"""Engine pool: one resident engine, exclusive copies while calls overlap.

Tests the real pool in src/telugu_speech/tts/runtime.py with fake engines
(objects, not ONNX) plus the HTTP surface with a tmp voice pack. No model
weights are loaded anywhere in this file.

Measured memory this pool is built around (20 Nov 2026, Apple M5 16 GB,
2 threads, short Telugu lines — see README.md): 1 engine 260-470 MB,
2 engines 640-720 MB, recognizer +350-540 MB. Those numbers are NOT
re-asserted here; a unit test cannot measure them.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from telugu_speech.api.auth import Settings
from telugu_speech.api.main import app, configure
from telugu_speech.tts.runtime import (
    borrow_engine,
    clear_shared_engines,
    shared_engine,
    speech_slots,
)
from telugu_speech.tts.voices import backend_for_voice


@pytest.fixture(autouse=True)
def _clean_pool() -> None:
    clear_shared_engines()
    yield
    clear_shared_engines()


def _counter_loader() -> object:
    calls = {"n": 0}

    def load() -> object:
        calls["n"] += 1
        return object()

    load.calls = calls  # type: ignore[attr-defined]
    return load


def test_shared_engine_loads_once() -> None:
    loader = _counter_loader()
    first = shared_engine(("test-pack",), loader)
    second = shared_engine(("test-pack",), loader)
    assert first is second
    assert loader.calls["n"] == 1


def test_failed_load_is_not_stored() -> None:
    attempts = {"n": 0}

    def flaky() -> object:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("weights missing")
        return object()

    with pytest.raises(RuntimeError):
        shared_engine(("test-pack",), flaky)
    engine = shared_engine(("test-pack",), flaky)
    assert engine is not None
    assert attempts["n"] == 2


def test_voice_pack_switch_evicts_the_old_pack() -> None:
    first = shared_engine(("hima-int8", "a"), lambda: object())
    second = shared_engine(("indic-tiny", "b"), lambda: object())
    assert first is not second
    # The old pack was evicted: loading it again builds a new engine.
    third = shared_engine(("hima-int8", "a"), lambda: object())
    assert third is not first


def test_borrow_gives_exclusive_engines_up_to_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELUGU_SPEECH_SLOTS", "2")
    assert speech_slots() == 2
    made: list[object] = []

    def load() -> object:
        engine = object()
        made.append(engine)
        return engine

    acquired: list[object] = []
    release = threading.Event()

    def hold() -> None:
        with borrow_engine(("test-pack",), load) as engine:
            acquired.append(engine)
            release.wait(timeout=5)

    threads = [threading.Thread(target=hold) for _ in range(2)]
    for thread in threads:
        thread.start()
    deadline = time.time() + 5
    while len(acquired) < 2 and time.time() < deadline:
        time.sleep(0.01)
    assert len(acquired) == 2
    assert acquired[0] is not acquired[1], "overlapping calls must not share one engine"
    assert len(made) == 2
    # A third caller waits: slots are full. It must get an engine only
    # after a holder releases — never a third copy.
    third_done = threading.Event()
    third_engine: list[object] = []

    def third() -> None:
        with borrow_engine(("test-pack",), load) as engine:
            third_engine.append(engine)
            third_done.set()

    worker = threading.Thread(target=third)
    worker.start()
    time.sleep(0.2)
    # threading.Event has is_set(), not is_not_set().
    assert not third_done.is_set(), "third caller acquired an engine past the slot cap"
    assert len(made) == 2
    release.set()
    assert third_done.wait(timeout=5)
    assert third_engine[0] in acquired, "released engine is reused, not rebuilt"
    assert len(made) == 2
    for thread in threads:
        thread.join(timeout=5)
    worker.join(timeout=5)


def test_slots_are_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELUGU_SPEECH_SLOTS", "99")
    assert speech_slots() == 4
    monkeypatch.setenv("TELUGU_SPEECH_SLOTS", "0")
    assert speech_slots() == 1
    monkeypatch.setenv("TELUGU_SPEECH_SLOTS", "not-a-number")
    assert speech_slots() == 2
    monkeypatch.delenv("TELUGU_SPEECH_SLOTS", raising=False)
    assert speech_slots() == 2


def test_tmp_voice_pack_backend_is_lazy_and_lists_over_http(tmp_path: Path) -> None:
    pack = tmp_path / "lite/models/hima-tts-nano-te"
    (pack / "onnx" / "voices").mkdir(parents=True)
    (pack / "onnx" / "voices" / "step_int8.onnx").write_bytes(b"fake-onnx")
    (pack / "voices").mkdir(parents=True, exist_ok=True)
    (pack / "voices" / "voices.json").write_text("{}", encoding="utf-8")
    backend = backend_for_voice("telugu_srinivas", tmp_path)
    assert backend is not None
    # Building the backend object loads no weights; the pool stays empty.
    from telugu_speech.tts import runtime

    assert runtime._ENGINES == {} and runtime._POOLS == {}
    configure(Settings("local_open", frozenset(), 20000000, "indicconformer", "ctc", "small", "indicf5"))
    client = TestClient(app)
    voices = client.get("/api/v1/tts/voices")
    assert voices.status_code == 200
    # GET /api/v1/tts/voices returns a dict envelope; the voice list is
    # under "voices" (each entry a dict with "id" — see routes_tts.voices).
    # The route was correct; this test was iterating the envelope's keys.
    ids = {voice["id"] for voice in voices.json()["voices"]}
    assert {"telugu_srinivas", "telugu_sravani"} <= ids
