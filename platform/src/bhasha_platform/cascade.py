"""Speech-to-speech cascade: ASR → translate → TTS."""

from __future__ import annotations

from pathlib import Path

from bhasha_platform.asr import transcribe
from bhasha_platform.translate import translate
from bhasha_platform.tts import synthesize


def speech_to_speech(
    audio: Path,
    output: Path,
    source_language: str = "te",
    target_language: str = "en",
) -> dict:
    asr = transcribe(audio, language=source_language)
    if source_language == target_language:
        mt = {"text": asr["text"], "backend": "identity", "source": source_language, "target": target_language}
    else:
        mt = translate(asr["text"], source=source_language, target=target_language)
    # Speak in the target language when TTS supports it; Telugu micro only speaks te.
    speak_text = mt["text"] if target_language == "te" else asr["text"]
    spoken_language = "te"
    tts = synthesize(speak_text, output)
    return {
        "transcript": asr,
        "translation": mt,
        "speech": tts,
        "note": (
            "Micro TTS currently speaks Telugu only. "
            "For target-language audio, configure IndicF5 with a consented reference."
            if target_language != "te"
            else "Telugu speech regenerated from transcript/translation path."
        ),
        "spoken_language": spoken_language,
    }
