"""Text to speech on the CPU with ONNX Runtime. No torch needed.

    from hima_onnx import OnnxTTS
    tts = OnnxTTS(".")                                      # the downloaded model folder
    tts = OnnxTTS(".", precision="int8")                    # smaller and faster
    audio = tts.say("...", voice="Bhavana")                 # built-in voice
    audio = tts.say("...", voice_file="me.safetensors")     # a saved voice (see the README)
    audio = tts.say("...")                                  # a random voice
    tts.save(audio, "out.wav")                              # float32 audio at 24 kHz
"""
import os
import sys
import json

import numpy as np
import onnxruntime as ort
import yaml
from safetensors import safe_open
from safetensors.numpy import load_file

SAMPLE_RATE = 24000
FRAMES_PER_SEC = 12.5   # one latent frame = 80 ms of audio
SEC_PER_CHAR = 0.092    # median speaking time per letter of good takes (Telugu 0.0896, Hindi 0.0933), for the dead-air guard
PREFIX_KEYS = ["text_embed.weight", "text_up.weight", "spk_in.weight", "spk_in.bias", "no_voice", "head_voice.weight",
               "head_no_voice", "bos_text", "bos_frame", "frame_in.weight", "ref_in.weight", "ref_tag", "emb_mean", "emb_std"]


def _load_normalize(folder):
    """The text normaliser from hima_tts/text, loaded by file path so torch is never imported."""
    import importlib.util
    d = os.path.join(folder, "text")
    if not os.path.exists(os.path.join(d, "normalize.py")):
        d = os.path.join(folder, "hima_tts", "text")
    if d not in sys.path:
        sys.path.append(d)  # normalize.py imports indic_languages by its bare name
    spec = importlib.util.spec_from_file_location("hima_text_normalize", os.path.join(d, "normalize.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.normalize


def _sentences(text, limit=220):
    """Same split as hima_tts: cut at sentence ends, pack pieces up to `limit` characters."""
    import re
    parts = [p.strip() for p in re.split(r"(?<=[.!?\u0964\u0965])\s+", text.strip()) if p.strip()]
    out = []
    for p in parts:
        if out and len(out[-1]) + len(p) + 1 <= limit:
            out[-1] += " " + p
        else:
            out.append(p)
    return out


class OnnxTTS:
    def __init__(self, model_dir=".", precision="fp32", threads=4, onnx_dir=None):
        """model_dir: the downloaded model folder. precision: 'fp32' or 'int8'. threads: CPU threads (4 is the sweet spot)."""
        onnx_dir = onnx_dir or os.path.join(model_dir, "onnx")
        if precision not in ("fp32", "int8"):
            raise ValueError(f"precision must be 'fp32' or 'int8', not {precision!r}")
        # int8: speech model with int8 weights (the flow head's input/output layers and the stop vote stay float32)
        # and a decoder whose attention weights are 8-bit; its convolutions stay float32 (int8 there costs sound quality)
        decoder = "int8" if precision == "int8" and os.path.exists(os.path.join(onnx_dir, "decoder_int8.onnx")) else "fp32"
        self.model_dir, self.onnx_dir = model_dir, onnx_dir
        self._normalize, self._sentences = _load_normalize(model_dir), _sentences
        import sentencepiece as spm
        cfg = yaml.safe_load(open(os.path.join(model_dir, "config.yaml")))
        self.language = cfg["language"]
        s = cfg.get("sampler", {})
        self.guidance = float(s.get("guidance", 1.25))
        self.max_frames = int(s.get("max_frames", 375))
        self.temperature = float(s.get("temperature", 0.3))
        self.flow_steps = int(s.get("flow_steps", 4))
        m = cfg["model"]
        self.vocab = m["vocab_size"]
        self.ref_frames = m["ref_frames"]
        self.n_layers = m["n_begin"] + m["n_middle"] * m["n_loops"] + m["n_end"]
        self.n_heads, self.d = m["n_heads"], m["d_model"]
        self.head_dim = self.d // self.n_heads
        self.sp = spm.SentencePieceProcessor(model_file=os.path.join(model_dir, "tokenizer.model"))
        self.voices = [v["name"] for v in json.load(open(os.path.join(model_dir, "voices", "voices.json")))["voices"]]
        self._voice_w = {}
        pf = os.path.join(model_dir, "prefix.safetensors")
        with safe_open(pf if os.path.exists(pf) else os.path.join(model_dir, "model.safetensors"), framework="np") as f:
            # the voices adapter does not touch these weights, so both graph sets share them
            self._pw = {k: np.asarray(f.get_tensor(k), dtype=np.float32) for k in PREFIX_KEYS}
        with safe_open(os.path.join(model_dir, "voices", "voices.safetensors"), framework="np") as f:
            self._presets = {k: np.asarray(f.get_tensor(k), dtype=np.float32) for k in f.keys()}
        self._threads = threads
        self._sess = {}
        self._decoder_path = os.path.join(onnx_dir, "decoder.onnx" if decoder == "fp32" else "decoder_int8.onnx")
        self._precision = precision
        self._dec = self._session(self._decoder_path)

    def _session(self, path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = self._threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

    def _graphs(self, wset):
        if wset not in self._sess:
            sub = os.path.join(self.onnx_dir, wset)
            suffix = "" if self._precision == "fp32" else "_int8"
            self._sess[wset] = (self._session(os.path.join(sub, f"step{suffix}.onnx")),
                                self._session(os.path.join(sub, f"frame{suffix}.onnx")))
        return self._sess[wset]

    def _prefix(self, W, ids, vec, ref):
        """Prefix rows for [text, blank text] -> [2, L, d] and the head add [d]. Same math as SpeechModel._prefix."""
        E, up = W["text_embed.weight"], W["text_up.weight"]
        t = E[ids] @ up.T
        tb = E[np.full_like(ids, self.vocab)] @ up.T
        if vec is not None:
            row = vec @ W["spk_in.weight"].T + W["spk_in.bias"]
            head = W["head_voice.weight"] @ vec
        else:
            row, head = W["no_voice"], W["head_no_voice"]
        parts_c = [W["bos_text"][None], t, row[None]]
        parts_u = [W["bos_text"][None], tb, row[None]]
        if ref is not None and len(ref) > 0:
            r = (ref[: self.ref_frames] - W["emb_mean"]) / W["emb_std"]
            rp = r @ W["ref_in.weight"].T + W["ref_tag"]
            parts_c.append(rp)
            parts_u.append(rp)
        bos = (W["bos_frame"] @ W["frame_in.weight"].T)[None]
        rows = np.stack([np.concatenate(parts_c + [bos]), np.concatenate(parts_u + [bos])]).astype(np.float32)
        return rows, head.astype(np.float32)

    def _frames(self, ids, vec, ref, wset, noises, state, seed_rng, limit=None):
        """Latent frames for one sentence. limit: stop after this many frames (the dead-air guard's length cap)."""
        W = self._pw
        step, frame = self._graphs(wset)
        rows, head = self._prefix(W, ids, vec, ref)
        empty = np.zeros((2, self.n_heads, 0, self.head_dim), np.float32)
        pasts = [empty] * (2 * self.n_layers)
        names = [f"past_{i}" for i in range(2 * self.n_layers)]
        outs = step.run(None, {"rows": rows, **dict(zip(names, pasts))})
        z, pasts = outs[0], outs[1:]
        g = np.array([self.guidance], np.float32)
        scale = np.float32(self.temperature ** 0.5)
        frames = []
        for _ in range(min(limit or self.max_frames, self.max_frames)):
            k = state["t"]; state["t"] += 1
            if noises is not None:
                eps = np.asarray(noises[k], np.float32)[None]
            else:
                eps = seed_rng.standard_normal((1, 32)).astype(np.float32)
            x, stop, nxt = frame.run(None, {"z": z, "head": head[None], "noise": scale * eps, "guidance": g})
            if stop[0, 0] > -1.0:
                break
            frames.append(x)
            rows1 = np.repeat(nxt[None], 2, axis=0)
            outs = step.run(None, {"rows": rows1, **dict(zip(names, pasts))})
            z, pasts = outs[0], outs[1:]
        if not frames:
            return np.zeros((0, 32), np.float32)
        return np.concatenate(frames, 0) * W["emb_std"] + W["emb_mean"]

    def frames(self, text, voice=None, voice_file=None, seed=None, noises=None):
        """Raw latent frames [n, 32] for one sentence (no codec)."""
        wset, vec, ref = self._select(voice, voice_file)
        ids = np.asarray(self.sp.encode(self._normalize(text, language=self.language)), np.int64)
        rng = np.random.default_rng(seed)
        return self._frames(ids, vec, ref, wset, noises, {"t": 0}, rng)

    def decode(self, frames):
        """Raw frames [n, 32] -> 24 kHz float32 audio via the codec graph."""
        if len(frames) == 0:
            return np.zeros(0, np.float32)
        return self._dec.run(None, {"frames": np.asarray(frames, np.float32)[None]})[0].reshape(-1)

    def _select(self, voice, voice_file):
        if voice is not None:
            if voice not in self.voices:
                raise ValueError(f"unknown voice {voice!r}; choose one of {self.voices}")
            return "voices", self._presets[f"vec_{voice}"], self._presets[f"ref_{voice}"]
        if voice_file is not None:
            from safetensors.numpy import load_file as lf
            d = lf(voice_file)
            return "base", np.asarray(d["vec"], np.float32).reshape(-1), np.asarray(d["ref"], np.float32)
        return "base", None, None

    @staticmethod
    def _longest_silence(audio):
        """Seconds of the longest run below -55 dBFS (100 ms windows) after the first sound; whole length if never loud."""
        w = SAMPLE_RATE // 10
        n = len(audio) // w
        if n == 0:
            return 0.0
        db = 20 * np.log10(np.sqrt((audio[: n * w].reshape(n, w) ** 2).mean(1)) + 1e-9)
        loud = np.where(db > -55)[0]
        if len(loud) == 0:
            return n / 10
        run = best = 0
        for v in db[loud[0]:]:
            run = run + 1 if v <= -55 else 0
            best = max(best, run)
        return best / 10

    def say(self, text, voice=None, voice_file=None, seed=None, noises=None, pause=0.15, retries=2):
        """Speech for `text` as float32 audio at 24 kHz. voice= a preset name; voice_file= .safetensors with vec [192] and ref [R, 32];
        neither = random voice. noises= optional [N, 32] standard-normal draws, consumed one per frame across sentences.
        retries: a sentence that goes silent for 3 s, or runs past twice the length its text needs, is generated again with new
        noise, up to this many times (the model sometimes says the first word and then falls silent). 0 turns the guard off."""
        wset, vec, ref = self._select(voice, voice_file)
        rng = np.random.default_rng(seed)
        state = {"t": 0}
        gap = np.zeros(int(pause * SAMPLE_RATE), np.float32)
        out = []
        random_voice = voice is None and voice_file is None
        self.last_retries = 0
        for part in self._sentences(text):
            norm = self._normalize(part, language=self.language)
            ids = np.asarray(self.sp.encode(norm), np.int64)
            chars = sum(ch.isalnum() or 0x0900 <= ord(ch) <= 0x0D7F for ch in norm)  # letters incl. Indic marks, no spaces/punct
            limit = int(np.ceil((2.0 * SEC_PER_CHAR * chars + 1.0) * FRAMES_PER_SEC)) if retries else None
            best = None
            for attempt in range(retries + 1):
                fr = self._frames(ids, vec, ref, wset, noises, state, rng, limit=limit)
                wav = self.decode(fr)
                hit_limit = limit is not None and len(fr) >= min(limit, self.max_frames)
                sil = self._longest_silence(wav)
                bad = hit_limit or sil >= 3.0
                if best is None or (not bad) or (best[3] and sil < best[2]):
                    best = (fr, wav, sil, bad)
                if not bad or not retries:
                    break
                self.last_retries += 1
            fr, wav = best[0], best[1]
            if random_voice and ref is None and len(fr) >= 12:
                ref = fr[: self.ref_frames]  # keep the random voice for the rest of the text
            if len(fr):
                out += [wav, gap]
        audio = np.concatenate(out[:-1]) if out else np.zeros(0, np.float32)
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        return (audio * (0.97 / peak) if peak > 0.97 else audio).astype(np.float32)

    @staticmethod
    def save(audio, path):
        import soundfile as sf
        sf.write(path, audio, SAMPLE_RATE)
