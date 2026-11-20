"""Local TTS backends. Heavy libraries load only on use."""

from __future__ import annotations

import json
from pathlib import Path

from telugu_speech.asr.infer import ModelUnavailable


class IndicF5Backend:
    def __init__(self, checkpoint: str = "ai4bharat/IndicF5") -> None:
        self.name = "indicf5"
        self.checkpoint = checkpoint
        self.requires_reference = True
        self.sample_rate_hz = 24000
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from transformers import AutoModel
        except Exception as exc:
            raise ModelUnavailable("transformers is not installed for IndicF5") from exc
        self._model = AutoModel.from_pretrained(self.checkpoint, trust_remote_code=True, local_files_only=True)
        return self._model

    def synthesize(self, text: str, reference_audio: Path | None, reference_text: str, output_path: Path) -> Path:
        if reference_audio is None or not reference_audio.is_file():
            raise FileNotFoundError("reference audio is required and must be a local file")
        model = self._load()
        audio = model(text, ref_audio_path=str(reference_audio), ref_text=reference_text)
        import numpy as np
        from telugu_speech.audio.resampling import write_wav

        array = np.asarray(audio)
        if np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float32) / 32768.0
        else:
            array = array.astype(np.float32)
        write_wav(output_path, array, self.sample_rate_hz)
        return output_path


class MmsTtsBackend:
    """Offline Meta MMS Telugu VITS. Full FP32 package. CC-BY-NC-4.0."""

    def __init__(self, checkpoint: str = "models/telugu-tts") -> None:
        self.name = "mms-tts-tel"
        self.checkpoint = checkpoint
        self.requires_reference = False
        self.sample_rate_hz = 16000
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return self._model, self._tokenizer
        path = Path(self.checkpoint)
        if not path.is_dir():
            raise ModelUnavailable(f"TTS checkpoint directory is missing: {self.checkpoint}")
        try:
            import torch
            from transformers import AutoTokenizer, VitsModel
        except Exception as exc:
            raise ModelUnavailable("torch and transformers are required for MMS Telugu TTS") from exc
        self._model = VitsModel.from_pretrained(str(path), local_files_only=True)
        self._tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self._model.eval()
        self._torch = torch
        return self._model, self._tokenizer

    def synthesize(self, text: str, reference_audio: Path | None, reference_text: str, output_path: Path) -> Path:
        del reference_audio, reference_text
        model, tokenizer = self._load()
        torch = self._torch
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            waveform = model(**inputs).waveform
        array = waveform.squeeze().detach().cpu().numpy()
        from telugu_speech.audio.resampling import write_wav

        write_wav(output_path, array.astype("float32"), self.sample_rate_hz)
        return output_path


class CompactMmsTtsBackend:
    """Disk-compressed INT8 MMS Telugu TTS. Dequantizes into RAM at load. CC-BY-NC-4.0."""

    def __init__(self, checkpoint: str = "models/telugu-tts-micro") -> None:
        self.name = "mms-tts-tel-micro"
        self.checkpoint = checkpoint
        self.requires_reference = False
        self.sample_rate_hz = 16000
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return self._model, self._tokenizer
        path = Path(self.checkpoint)
        weights = path / "weights.int8.npz"
        scales_path = path / "scales.json"
        if not weights.is_file() or not scales_path.is_file():
            raise ModelUnavailable(f"compact TTS weights missing under {self.checkpoint}")
        try:
            import numpy as np
            import torch
            from transformers import AutoTokenizer, VitsModel
        except Exception as exc:
            raise ModelUnavailable("torch, numpy, and transformers are required for compact TTS") from exc
        scales = json.loads(scales_path.read_text(encoding="utf-8"))
        packed = np.load(weights)
        state = {}
        for key in packed.files:
            array = packed[key]
            scale = scales.get(key)
            if scale is None:
                state[key] = torch.from_numpy(array)
            else:
                state[key] = torch.from_numpy(array.astype("float32") * float(scale))
        from transformers import VitsConfig

        config = VitsConfig.from_pretrained(str(path), local_files_only=True)
        model = VitsModel(config)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise ModelUnavailable(f"compact TTS missing weights: {missing[:5]}")
        del unexpected
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        return self._model, self._tokenizer

    def synthesize(self, text: str, reference_audio: Path | None, reference_text: str, output_path: Path) -> Path:
        del reference_audio, reference_text
        model, tokenizer = self._load()
        torch = self._torch
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            waveform = model(**inputs).waveform
        array = waveform.squeeze().detach().cpu().numpy()
        from telugu_speech.audio.resampling import write_wav

        write_wav(output_path, array.astype("float32"), self.sample_rate_hz)
        return output_path


def build_tts_backend(name: str, checkpoint: str | None = None):
    if name in {"mms-micro", "mms_tts_micro", "compact"}:
        return CompactMmsTtsBackend(checkpoint or "models/telugu-tts-micro")
    if name in {"mms", "mms-tts-tel", "mms_tts"}:
        path = checkpoint or "models/telugu-tts"
        if Path(path).joinpath("weights.int8.npz").is_file() or Path("models/telugu-tts-micro/weights.int8.npz").is_file():
            if checkpoint is None and not Path("models/telugu-tts").joinpath("model.safetensors").is_file():
                return CompactMmsTtsBackend("models/telugu-tts-micro")
        return MmsTtsBackend(path)
    if name == "indicf5":
        return IndicF5Backend(checkpoint or "ai4bharat/IndicF5")
    raise ValueError(f"unknown TTS backend {name}")
