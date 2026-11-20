"""The six-voice pack already lives in lite/models/hima-tts-nano-te.

Run this only when that folder is missing.
"""

from __future__ import annotations

from pathlib import Path


DESTINATION = Path("lite/models/hima-tts-nano-te")
REPO_ID = "Prahaladha/hima-tts-nano-te-ONNX"


def main() -> None:
    marker = DESTINATION / "onnx" / "voices" / "step_int8.onnx"
    if marker.is_file():
        print("VOICE_PACK_PRESENT", DESTINATION)
        return
    from huggingface_hub import snapshot_download

    DESTINATION.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        REPO_ID,
        local_dir=str(DESTINATION),
        ignore_patterns=[
            "onnx/voices/step.onnx",
            "onnx/base/step.onnx",
            "onnx/voices/frame.onnx",
            "onnx/base/frame.onnx",
            "onnx/decoder.onnx",
        ],
    )
    print("VOICE_PACK_PRESENT", DESTINATION)


if __name__ == "__main__":
    main()
