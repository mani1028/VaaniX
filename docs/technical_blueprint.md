# Telugu Speech Engine blueprint

This file records the locked design. It extends the Phase 1 package. It does not replace `src/telugu_speech`, the existing `/api/v1` routes, the MIT license, or the tests.

The folder on this computer is `/Users/visys/.cursor/tts sst`. The inspection and the faster-whisper measurement already happened. See `docs/benchmark_results.md` and `docs/phase_status.md`.

## Decisions

| Decision | Selection |
| --- | --- |
| Language | Telugu first |
| ASR | Compact Conformer, about 12 layers, width 256, 4 heads, depthwise convolution, CTC greedy first |
| ASR size target | 25–50M parameters |
| TTS | FastSpeech2 or Matcha-TTS, duration, pitch, energy, HiFi-GAN |
| TTS size target | 20–60M, one consented speaker first |
| Teacher | IndicConformer Telugu checkpoint after its real dimensions are verified |
| Extra teacher | faster-whisper |
| External reference | Sarvam Saaras v4 and Bulbul v4 Flash, same audio and same text |
| First compression | INT8, then pruning only if recovery data exists |
| Export | ONNX, with a Core ML measurement on the M5 rather than an assumption |
| API | Keep `/api/v1/*`. Add `/v1/*` later if a second contract is required |
| Hardware | M5, 16 GB, for prep and small inference. Rent a GPU only for a defined training job |

These sizes are targets. No student checkpoint has been trained.

## Data rules

- Common Voice Telugu, Kathbath Telugu, IndicVoices, OpenSLR SLR66, IISc SYSPIN, and IndicTTS are candidates only after each card is copied into `docs/licenses.md`.
- Speaker-disjoint manifests stay the split rule.
- The six FLEURS `te_in` test clips already downloaded are evaluation audio. They are not a training set.
- Sarvam text may be a reference score. It is not a training label unless the API terms allow that use.

## Order

1. Keep Phase 1 tests passing.
2. Grow the licensed evaluation manifest before any training.
3. Run IndicConformer in a Python 3.11 environment, not in the Python 3.15 venv.
4. Fine-tune, then distil a student that does not need the teacher at inference.
5. Train TTS on a consented corpus such as IISc after the terms are recorded.
6. Quantize, measure size, RTF, and RAM, and compare with Sarvam on the same items.

## Where to look

- Dashboard: `index.html` in the repository root, served at `/` by the existing API.
- Measured table: `docs/benchmark_results.md`.
- What is blocked: `docs/phase_status.md`.
