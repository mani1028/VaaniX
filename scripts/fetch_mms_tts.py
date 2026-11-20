"""Download Meta MMS Telugu TTS once into models/telugu-tts. CC-BY-NC-4.0."""

from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer, VitsModel


def main() -> None:
    destination = Path("models/telugu-tts")
    destination.mkdir(parents=True, exist_ok=True)
    model_id = "facebook/mms-tts-tel"
    print(f"downloading {model_id} into {destination}")
    model = VitsModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    total = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    print(f"saved {total} bytes under {destination}")
    print("license: CC-BY-NC-4.0 — research and non-commercial use only until replaced")


if __name__ == "__main__":
    main()
