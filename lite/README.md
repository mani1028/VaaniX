# Telugu voice pack

The weights are `models/hima-tts-nano-te` (int8 ONNX, about 150 MB) and they ship with this folder.

The model's six published names are two pitches. On the same sentence the men are about 100–115 Hz and the women are about 185–195 Hz. A second sentence moves each group together, so Karthik is not a different person from Srinivas, and Bhavana is not a different person from Sravani.

| Voice id | What you hear | Where it comes from |
| --- | --- | --- |
| `telugu_srinivas` | deep male, about 100–115 Hz | trained speaker Srinivas |
| `telugu_sravani` | female, about 185–225 Hz | trained speaker Sravani |

Samples are in `voices/samples/`. Same sentence in every file, peaked to the same loudness. Call `POST /api/v1/tts/generate` with `"voice": "telugu_srinivas"` and Telugu text.

These are open Hima voices. They are not Grok voices and they are not Sarvam Bulbul voices.
