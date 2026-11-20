---
license: other
language: [te, en]
library_name: onnx
base_model: Prahaladha/hima-tts-nano-te
tags: [text-to-speech, onnx]
---
# hima-tts-nano-te-ONNX

[hima-tts-nano-te](https://huggingface.co/Prahaladha/hima-tts-nano-te) (Telugu text to speech, 62M parameters) converted to ONNX, for fast speech on any CPU (Windows, Linux, Mac), with no PyTorch.
Same voices, same sound; `int8` is 3.3x smaller and about 1.4x faster on a CPU, and reads as accurately as `fp32` (checked on 1,202 sentences, below).

Built-in voices: Bhavana (female), Lakshmi (female), Sravani (female), Srinivas (male), Karthik (male), Pranay (male). Leave the voice out for a new random voice.

## Install

```bash
pip install -U huggingface_hub
hf download Prahaladha/hima-tts-nano-te-ONNX --local-dir hima-tts-nano-te-ONNX
cd hima-tts-nano-te-ONNX
pip install -r requirements.txt
```

## Use it

```python
from hima_onnx import OnnxTTS

tts = OnnxTTS(".", precision="int8")      # or "fp32"
audio = tts.say("ఏమండీ, సాయంత్రం free గా ఉన్నారా? బయటికి వెళ్ళి coffee తాగుదాం.", voice="Bhavana")
audio = tts.say("ఏమండీ, సాయంత్రం free గా ఉన్నారా? బయటికి వెళ్ళి coffee తాగుదాం.")              # a random voice
tts.save(audio, "hello.wav")
```

`say` returns float32 audio at 24 kHz. Pass `seed=` for the same take every time.

## Your own voice

Copying a voice from a recording needs the main repo once (it uses PyTorch): `python -m hima_tts --save-voice me.safetensors --from me.wav`
inside [hima-tts-nano-te](https://huggingface.co/Prahaladha/hima-tts-nano-te). Then use the file here: `tts.say("...", voice_file="me.safetensors")`.
A random voice you like can be saved the same way (`--save-voice calm.safetensors --seed 7`).

## Speed

Apple M4 Pro CPU, 48 sentences:

| | `fp32` | `int8` |
|---|---|---|
| 4 threads | 10.7x real time | **15.0x** |
| 1 thread | 6.7x | **10.5x** |
| model files for one voice set | 273 MB | **84 MB** |

In a browser (onnxruntime-web 1.23, one WASM thread, same graphs): `fp32` 2.9x and `int8` 2.3x real time;
int8 matrix kernels are slower than float ones in WASM, so there `fp32` is faster and `int8` is smaller.

## How it reads

1,202 sentences (1,002 with the six voices, 200 with random voices), the same sentence and seed for both versions,
read back by two speech recognizers (Sarvam saaras v4 and Hima ASR). Letters wrong counts an English word written in Telugu script or
in Latin letters as the same word.

| Letters wrong (lower is better) | `fp32` | `int8` |
|---|---|---|
| Built-in voices, Sarvam saaras v4 | 3.15% | 3.39% |
| Built-in voices, Hima ASR | 4.15% | 4.07% |
| Random voices, Sarvam saaras v4 | 4.74% | 3.76% |
| Random voices, Hima ASR | 5.16% | 4.16% |

Line by line, `int8` and `fp32` differ by +0.04 points on Sarvam (95% interval -0.42 to
+0.49): the same accuracy.

What `int8` changes: the speech model's matrix weights are 8-bit, except a few small layers at the input and output of the
flow head, which stay float32; the audio decoder's attention weights are 8-bit and its convolutions stay float32.

## Files

| File | What it is |
|---|---|
| `onnx/base/`, `onnx/voices/` | the model as ONNX graphs, `fp32` and `int8`; `voices/` holds the version tuned for the six voices |
| `onnx/decoder.onnx` | turns the model's output into audio |
| `onnx/decoder_int8.onnx` | the same, used with `precision="int8"` |
| `prefix.safetensors` | the few weights the code uses before the graphs |
| `hima_onnx/` | the code |
| `voices/voices.safetensors`, `voices/voices.json` | the six voices' settings and names |
| `tokenizer.model`, `config.yaml` | text tokenizer, model and sampling settings |
| `text/` | turns numbers, dates and amounts into words |
