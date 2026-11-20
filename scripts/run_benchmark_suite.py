"""Score the engines that are installed. Do not invent the rest.

Usage, from the repository root, with the Python 3.11 speech environment:

    TELUGU_SPEECH_SLOTS=2 .venv311/bin/python scripts/run_benchmark_suite.py
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telugu_speech.asr.dataset import read_manifest
from telugu_speech.evaluation.asr_metrics import character_error_rate, word_error_rate
from telugu_speech.evaluation.competitors import adapter_status, download_decision, locate_competitors
from telugu_speech.evaluation.dataset_gate import dataset_counts, release_decision
from telugu_speech.evaluation.suite import model_catalog, silence_stats, write_report
from telugu_speech.evaluation.tts_metrics import real_time_factor
from telugu_speech.text.normalization import NormalizationConfig, normalize_text

SENTENCES = (
    "నమస్కారం. ఈ రోజు వాతావరణం బాగుంది.",
    "రేపు ఉదయం 10:30కి కాల్ చేయండి.",
    "మీ ఆర్డర్ ₹500. డెలివరీ రేపు.",
)


def main() -> None:
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--locate-only", action="store_true")
    parser.add_argument("--download", default="", help="Print whether a competitor may be fetched. This does not download it.")
    parser.add_argument("--allow-large", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("TELUGU_SPEECH_SLOTS", "2")
    os.environ.setdefault("TELUGU_CPU_THREADS", "2")
    if args.download:
        free = shutil.disk_usage(ROOT).free
        print(json.dumps(download_decision(args.download, free, args.allow_large)))
        return
    if args.locate_only:
        report = _existing_report()
        _attach_phase2(report)
        destination = ROOT / "experiments" / "benchmarks" / "suite"
        write_report(destination, report)
        print(destination / "report.html")
        return
    report: dict[str, object] = {
        "hardware": _hardware(),
        "models": model_catalog(ROOT),
        "tts": _tts(),
        "concurrency": _concurrency(),
        "stt": _stt(),
        "combined": _combined(),
        "subjective": {
            "mos": None,
            "speaker_similarity": None,
            "pronunciation": None,
            "note": "No native-speaker panel was run. ASR error is not a naturalness score.",
        },
        "missing_sets": [
            "Telangana and Andhra labeled speech is not in data/manifests/eval.jsonl.",
            "Noisy and phone-call sets are not on disk.",
            "Telugu-English code-mix audio is not a separate scored set.",
            "First partial STT latency is not measured. The recognizer returns the whole clip.",
        ],
    }
    _attach_phase2(report)
    destination = ROOT / "experiments" / "benchmarks" / "suite"
    write_report(destination, report)
    print(destination / "report.html")


def _existing_report() -> dict[str, object]:
    path = ROOT / "experiments" / "benchmarks" / "suite" / "results.json"
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return loaded
    return {"hardware": _hardware(), "models": model_catalog(ROOT), "stt": [], "tts": [], "concurrency": []}


def _attach_phase2(report: dict[str, object]) -> None:
    competitors = locate_competitors(ROOT)
    for row in competitors:
        probe = adapter_status(ROOT, str(row["model"]))
        row["runnable"] = probe["runnable"]
        row["adapter"] = probe["reason"]
    report["competitors"] = competitors
    report["dataset"] = dataset_counts(ROOT / "data/manifests/competitive.jsonl")
    report["decision"] = release_decision(
        {"status": "measured", "clips": 10, "wer": None},
        {"status": "not_measured"},
    )
    report["decision"]["reason"] = (
        "Production stays on the 42 MB recognizer and Hima at two threads. "
        "No competitor has a score on this machine."
    )


def _hardware() -> dict[str, object]:
    ort = None
    try:
        import onnxruntime

        ort = onnxruntime.__version__
    except Exception:
        ort = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "onnxruntime": ort,
        "threads": os.environ.get("TELUGU_CPU_THREADS", "2"),
        "slots": os.environ.get("TELUGU_SPEECH_SLOTS", "2"),
        "precision": "int8",
        "device": "cpu",
    }


def _stt() -> list[dict[str, object]]:
    import gc

    from telugu_speech.tts.runtime import clear_shared_engines

    clear_shared_engines()
    gc.collect()
    rows = []
    manifest = ROOT / "data/manifests/eval.jsonl"
    for name, folder in (("telugu-asr-micro", "models/telugu-asr-micro"), ("telugu-asr-small", "models/telugu-asr")):
        path = ROOT / folder
        if not (path / "model.bin").is_file() or not manifest.is_file():
            rows.append({"model": name, "status": "not_measured", "reason": "weights or eval manifest missing"})
            continue
        from telugu_speech.asr.infer import FasterWhisperBackend

        backend = FasterWhisperBackend(str(path))
        utterances = read_manifest(manifest)
        load_started = time.perf_counter()
        first = utterances[0]
        backend.transcribe(ROOT / first.audio)
        load_s = time.perf_counter() - load_started
        elapsed = 0.0
        audio_s = 0.0
        word_errors = 0
        word_refs = 0
        char_errors = 0
        char_refs = 0
        config = NormalizationConfig()
        finals = []
        peak = _rss_mb()
        for utterance in utterances:
            audio = ROOT / utterance.audio
            started = time.perf_counter()
            hypothesis = backend.transcribe(audio)
            took = time.perf_counter() - started
            finals.append(took)
            elapsed += took
            audio_s += float(utterance.duration_s or 0)
            _raw, normalized = normalize_text(hypothesis, config)
            words = word_error_rate(utterance.norm_text, normalized)
            chars = character_error_rate(utterance.norm_text, normalized)
            word_errors += words.substitutions + words.deletions + words.insertions
            word_refs += words.reference_length
            char_errors += chars.substitutions + chars.deletions + chars.insertions
            char_refs += chars.reference_length
            peak = max(peak, _rss_mb())
        rows.append(
            {
                "model": name,
                "status": "measured",
                "clips": len(utterances),
                "wer": _rate(word_errors, word_refs),
                "cer": _rate(char_errors, char_refs),
                "rtf": real_time_factor(elapsed, audio_s),
                "load_s": round(load_s, 3),
                "final_p50_s": _round(percentile_local(finals, 50)),
                "peak_rss_mb": round(peak, 1),
                "cpu_percent_avg": _cpu_during(lambda: backend.transcribe(ROOT / first.audio)),
                "first_partial_s": None,
                "noise_wer": None,
                "code_mix_wer": None,
                "note": "Same process already held an earlier recognizer." if rows else "",
            }
        )
    return rows


def _tts() -> list[dict[str, object]]:
    pack = ROOT / "lite/models/hima-tts-nano-te/onnx/voices/step_int8.onnx"
    if not pack.is_file():
        return [{"voice": "hima-int8", "status": "not_measured", "reason": "voice pack missing"}]
    from telugu_speech.tts.runtime import clear_shared_engines
    from telugu_speech.tts.voices import backend_for_voice

    rows = []
    for threads in ("1", "2", "4"):
        os.environ["TELUGU_CPU_THREADS"] = threads
        clear_shared_engines()
        for voice_id in ("telugu_srinivas", "telugu_sravani"):
            backend = backend_for_voice(voice_id)
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "speech.wav"
                started = time.perf_counter()
                backend.synthesize(SENTENCES[0], None, "", output)
                cold_s = time.perf_counter() - started
                audio_s = _wav_seconds(output)
                warm_elapsed = 0.0
                warm_audio = 0.0
                first_audio = None
                for sentence in SENTENCES:
                    piece_started = time.perf_counter()
                    next(iter(backend.iter_pcm(sentence)))
                    if first_audio is None:
                        first_audio = time.perf_counter() - piece_started
                    started = time.perf_counter()
                    backend.synthesize(sentence, None, "", output)
                    warm_elapsed += time.perf_counter() - started
                    warm_audio += _wav_seconds(output)
                samples = _wav_samples(output)
                stats = silence_stats(samples, 24000)
            rows.append(
                {
                    "voice": voice_id,
                    "threads": int(threads),
                    "status": "measured",
                    "cold_s": round(cold_s, 3),
                    "rtf": real_time_factor(warm_elapsed, warm_audio),
                    "first_audio_s": _round(first_audio),
                    "audio_s": round(audio_s, 3),
                    "peak_rss_mb": round(_rss_mb(), 1),
                    "silence_fraction": _round(stats["silence_fraction"]),
                    "longest_silence_s": _round(stats["longest_silence_s"]),
                    "mos": None,
                }
            )
    os.environ["TELUGU_CPU_THREADS"] = "2"
    clear_shared_engines()
    switch = _switch_time()
    if rows:
        rows[0]["voice_switch_s"] = switch
    return rows


def _switch_time() -> float | None:
    from telugu_speech.tts.voices import backend_for_voice

    male = backend_for_voice("telugu_srinivas")
    female = backend_for_voice("telugu_sravani")
    next(iter(male.iter_pcm("నమస్కారం.")))
    started = time.perf_counter()
    next(iter(female.iter_pcm("నమస్కారం.")))
    return round(time.perf_counter() - started, 3)


def _concurrency() -> list[dict[str, object]]:
    pack = ROOT / "lite/models/hima-tts-nano-te/onnx/voices/step_int8.onnx"
    if not pack.is_file():
        return []
    from telugu_speech.tts.voices import backend_for_voice

    os.environ["TELUGU_CPU_THREADS"] = "2"
    os.environ["TELUGU_SPEECH_SLOTS"] = os.environ.get("TELUGU_SPEECH_SLOTS", "2")
    slots = int(os.environ["TELUGU_SPEECH_SLOTS"])
    backend = backend_for_voice("telugu_srinivas")
    with tempfile.TemporaryDirectory() as directory:
        backend.synthesize("నమస్కారం.", None, "", Path(directory) / "warm.wav")
    rows = []
    for workers in (1, 2, 3, 4, 8, 16):
        latencies: list[float] = []
        errors = 0

        def once() -> None:
            nonlocal errors
            started = time.perf_counter()
            try:
                with tempfile.TemporaryDirectory() as directory:
                    backend_for_voice("telugu_srinivas").synthesize(
                        "నమస్కారం.", None, "", Path(directory) / "speech.wav"
                    )
            except Exception:
                errors += 1
                return
            latencies.append(time.perf_counter() - started)

        wall = time.perf_counter()
        cpu = _cpu_during(lambda: _run_workers(workers, once))
        wall_s = time.perf_counter() - wall
        summary = {
            "kind": "tts",
            "workers": workers,
            "slots": slots,
            "p50_s": _round(percentile_local(latencies, 50)),
            "p95_s": _round(percentile_local(latencies, 95)),
            "p99_s": _round(percentile_local(latencies, 99)),
            "errors": errors,
            "wall_s": round(wall_s, 3),
            "cpu_percent_avg": cpu,
            "peak_rss_mb": round(_rss_mb(), 1),
        }
        rows.append(summary)
    return rows


def _combined() -> dict[str, object]:
    audio = ROOT / "data/raw/fleurs_te/fleurs_000.wav"
    pack = ROOT / "lite/models/hima-tts-nano-te/onnx/voices/step_int8.onnx"
    if not audio.is_file() or not pack.is_file() or not (ROOT / "models/telugu-asr-micro/model.bin").is_file():
        return {"status": "not_measured"}
    from telugu_speech.asr.infer import FasterWhisperBackend
    from telugu_speech.tts.voices import backend_for_voice

    recognizer = FasterWhisperBackend(str(ROOT / "models/telugu-asr-micro"))
    recognizer.transcribe(audio)
    speaker = backend_for_voice("telugu_srinivas")
    with tempfile.TemporaryDirectory() as directory:
        speaker.synthesize("నమస్కారం.", None, "", Path(directory) / "warm.wav")

    def listen() -> None:
        recognizer.transcribe(audio)

    def speak() -> None:
        with tempfile.TemporaryDirectory() as directory:
            speaker.synthesize(SENTENCES[0], None, "", Path(directory) / "speech.wav")

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(listen)
        second = pool.submit(speak)
        first.result()
        second.result()
    return {
        "status": "measured",
        "wall_s": round(time.perf_counter() - started, 3),
        "peak_rss_mb": round(_rss_mb(), 1),
        "note": "One warm recognition and one warm Srinivas line at the same time.",
    }


def _run_workers(workers: int, task) -> None:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda _index: task(), range(workers)))


def _cpu_during(task) -> float | None:
    samples: list[float] = []
    stop = False

    def watch() -> None:
        while not stop:
            samples.append(_cpu_percent())
            time.sleep(0.2)

    import threading

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        task()
    finally:
        stop = True
        thread.join(timeout=1)
    if not samples:
        return None
    return round(sum(samples) / len(samples), 1)


def _rss_mb() -> float:
    try:
        text = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return int(text.strip() or 0) / 1024
    except (subprocess.SubprocessError, ValueError):
        return 0.0


def _cpu_percent() -> float:
    try:
        text = subprocess.check_output(["ps", "-o", "%cpu=", "-p", str(os.getpid())], text=True)
        return float(text.strip() or 0)
    except (subprocess.SubprocessError, ValueError):
        return 0.0


def _wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def _wav_samples(path: Path) -> list[float]:
    with wave.open(str(path), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
        width = handle.getsampwidth()
    if width != 2:
        return []
    unpacked = raw
    count = len(unpacked) // 2
    step = max(1, count // 8000)
    values = []
    for index in range(0, count, step):
        sample = int.from_bytes(unpacked[index * 2 : index * 2 + 2], "little", signed=True)
        values.append(sample / 32768.0)
    return values


def _rate(errors: int, refs: int) -> float | None:
    if refs <= 0:
        return None
    return round(errors / refs, 4)


def percentile_local(values: list[float], pct: float) -> float | None:
    from telugu_speech.evaluation.suite import percentile

    return percentile(values, pct)


def _round(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value), 4)


if __name__ == "__main__":
    main()
