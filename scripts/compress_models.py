"""Build the under-100MB offline package: Telugu Whisper tiny INT8 + MMS TTS INT8 NPZ."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import torch
from ctranslate2.converters import TransformersConverter
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, VitsModel

ASR_OUT = Path("models/telugu-asr-micro")
TTS_OUT = Path("models/telugu-tts-micro")
TTS_SRC = Path("models/telugu-tts")
LIMIT = 100 * 1024 * 1024


def build_asr() -> int:
    if ASR_OUT.exists():
        shutil.rmtree(ASR_OUT)
    print("converting vasista22/whisper-telugu-tiny -> int8")
    TransformersConverter("vasista22/whisper-telugu-tiny").convert(
        str(ASR_OUT), quantization="int8", force=True
    )
    for name in ("tokenizer.json", "preprocessor_config.json"):
        try:
            src = hf_hub_download("vasista22/whisper-telugu-tiny", name)
        except Exception:
            src = hf_hub_download("openai/whisper-tiny", name)
        shutil.copy2(src, ASR_OUT / name)
    return sum(path.stat().st_size for path in ASR_OUT.rglob("*") if path.is_file())


def build_tts() -> int:
    if not TTS_SRC.is_dir():
        raise SystemExit("models/telugu-tts missing; run scripts/fetch_mms_tts.py first")
    TTS_OUT.mkdir(parents=True, exist_ok=True)
    model = VitsModel.from_pretrained(str(TTS_SRC), local_files_only=True)
    payload = {}
    scales = {}
    for key, tensor in model.state_dict().items():
        array = tensor.detach().cpu()
        if array.is_floating_point() and array.numel() > 0:
            values = array.float()
            max_abs = float(values.abs().max()) if values.numel() else 1.0
            scale = max_abs / 127.0 if max_abs > 0 else 1.0
            quantized = torch.clamp(torch.round(values / scale), -128, 127).to(torch.int8)
            payload[key] = quantized.numpy()
            scales[key] = scale
        else:
            payload[key] = array.numpy()
            scales[key] = None
    np.savez_compressed(TTS_OUT / "weights.int8.npz", **payload)
    (TTS_OUT / "scales.json").write_text(json.dumps(scales), encoding="utf-8")
    AutoTokenizer.from_pretrained(str(TTS_SRC), local_files_only=True).save_pretrained(TTS_OUT)
    shutil.copy2(TTS_SRC / "config.json", TTS_OUT / "config.json")
    for leftover in ("weights.fp16.pt", "model.safetensors"):
        path = TTS_OUT / leftover
        if path.exists():
            path.unlink()
    return sum(path.stat().st_size for path in TTS_OUT.rglob("*") if path.is_file())


def main() -> None:
    asr_bytes = build_asr()
    tts_bytes = build_tts()
    total = asr_bytes + tts_bytes
    print("stt_bytes", asr_bytes)
    print("tts_bytes", tts_bytes)
    print("total_bytes", total)
    print("under_100mb", total < LIMIT)
    if total >= LIMIT:
        raise SystemExit("compressed package still exceeds 100 MB")


if __name__ == "__main__":
    main()
