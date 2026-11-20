# Dataset cards

These are pointers for later downloads. Phase 1 does not fetch them.

| Dataset | Role | Terms to verify before training | Notes |
| --- | --- | --- | --- |
| Kathbath Telugu | ASR train | Hugging Face card `ai4bharat/Kathbath`, CC-BY-4.0 packaging; IndicSUPERB text provenance is separate | Read speech from many districts |
| Shrutilipi Telugu | ASR supplement | CC-BY-4.0 | All India Radio news. Not everyday conversation |
| Mozilla Common Voice Telugu | ASR train | CC0 | Crowdsourced read speech |
| FLEURS Telugu | ASR test only | Dataset card | Do not mix into training |
| MUCS Telugu | ASR test only | Obtain the original distribution | Use only if the license allows this benchmark |
| IndicTTS Telugu | TTS train and ASR read-speech test | IIT Madras IndicTTS terms | Studio recordings |
| IndicVoices-R Telugu | TTS after single-speaker quality | CC-BY-4.0, gated access | Accept the gate before download |
| Rasa | Male and female Telugu TTS only after the card is checked | CC BY 4.0 on the published card; store `consent_status` per speaker before training | About 27 h female and 25 h male in the dataset description. Not downloaded here |
| TeluguAccentBench | Own eval set | Consent for research and, separately, commercial voice use | Not an existing corpus. Regions: Telangana, Coastal Andhra, Rayalaseema, Telugu-English code-mix. Conditions: clean, phone, noisy, conversational |

Speaker ids must not appear in more than one of train, validation, and test. Augmented copies of test audio are not training data.

`data/manifests/eval.jsonl` is the locked FLEURS test set. `configs/asr/finetune.yaml` lists `fleurs_te` in `held_out` and points `held_out_manifest` at that file. `train()` rejects a train manifest that reuses those `utt_id` values and still refuses to launch a job.

Optional manifest keys that old rows may omit: `transcript_romanized`, `language`, `code_mix_segments`, `recording_channel`, `sample_rate`, `license`, `consent_status`, `lid_label`. Missing keys do not fail `read_manifest`. `lid_label` is required before any detection F1 is computed. The current eval file has none, so the detection gate is not evaluable.

License checklist before any download is used for training:

- Common Voice Telugu: CC0-1.0, small supplement only.
- Kathbath: confirm CC BY 4.0 packaging and separate source-text rights.
- IndicVoices: terms differ by archive. Do not assume one card covers every dump.
- Rasa: confirm CC BY 4.0 and record speaker consent before a voice is trained or cloned.
- FLEURS Telugu test clips: evaluation only.

MMS-TTS Telugu is CC-BY-NC-4.0. It is not a training source for a model this project would ship.
