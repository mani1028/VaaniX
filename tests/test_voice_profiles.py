from pathlib import Path

from telugu_speech.tts.runtime import borrow_engine, clear_shared_engines, shared_engine
from telugu_speech.tts.voices import VOICES, backend_for_voice, list_installed_voices, sample_names


def test_listed_voices_are_spread_across_one_pack() -> None:
    catalog = {item["id"]: item for item in list_installed_voices(Path("."))}
    assert set(VOICES) == set(catalog)
    assert "telugu_bhavana" not in catalog
    assert "telugu_karthik" not in catalog
    assert "te_female_01" not in catalog
    assert "te_male_01" not in catalog
    assert {item["download_id"] for item in catalog.values()} == {"hima-tts-nano-te"}
    assert sample_names() == {spec["sample"] for spec in VOICES.values()}
    assert catalog["telugu_srinivas"]["gender"] == "male"
    assert catalog["telugu_srinivas"]["language"] == "te"
    assert catalog["telugu_srinivas"]["compatibility"] == ["hima-onnx-int8"]
    assert catalog["telugu_sravani"]["compatibility"] == ["hima-onnx-int8"]
    assert catalog["telugu_srinivas"]["shared_weights"] is True
    assert catalog["telugu_sravani"]["gender"] == "female"
    assert catalog["telugu_srinivas"]["available"] is (
        Path("lite/models/hima-tts-nano-te/onnx/voices/step_int8.onnx").is_file()
    )
    assert not Path("pro").exists()
    default = backend_for_voice("telugu_default")
    assert default is not None
    assert default.speaker == "Srinivas"
    assert default.requires_reference is False


def test_one_pack_loads_its_engine_once() -> None:
    clear_shared_engines()
    calls = {"n": 0}

    def load() -> dict[str, int]:
        calls["n"] += 1
        return {"n": calls["n"]}

    first = shared_engine(("hima-int8", "pack"), load)
    second = shared_engine(("hima-int8", "pack"), load)
    assert first is second
    assert calls["n"] == 1

    def fail() -> object:
        raise RuntimeError("missing")

    try:
        shared_engine(("missing",), fail)
    except RuntimeError:
        recovered = shared_engine(("missing",), lambda: "ready")
    else:
        raise AssertionError("a failed load must not be stored")
    assert recovered == "ready"
    clear_shared_engines()


def test_second_voice_pack_releases_the_first() -> None:
    clear_shared_engines()
    calls = {"hima": 0}

    def load_hima() -> dict[str, int]:
        calls["hima"] += 1
        return {"n": calls["hima"]}

    shared_engine(("hima-int8", "pack"), load_hima)
    shared_engine(("indic-tiny", "pack"), lambda: {"pack": "tiny"})
    reloaded = shared_engine(("hima-int8", "pack"), load_hima)
    assert calls["hima"] == 2
    assert reloaded["n"] == 2
    clear_shared_engines()


def test_two_requests_get_two_engines() -> None:
    import threading

    import time

    clear_shared_engines()
    created = {"n": 0}
    inside = {"n": 0, "peak": 0}
    gate = threading.Lock()
    seen: list[int] = []

    def load() -> dict[str, int]:
        with gate:
            created["n"] += 1
            return {"id": created["n"]}

    def work() -> None:
        with borrow_engine(("hima-int8", "pack"), load) as engine:
            with gate:
                inside["n"] += 1
                inside["peak"] = max(inside["peak"], inside["n"])
                seen.append(engine["id"])
            time.sleep(0.05)
            with gate:
                inside["n"] -= 1

    threads = [threading.Thread(target=work) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    assert inside["peak"] == 2
    assert sorted(seen) == [1, 2]
    clear_shared_engines()
