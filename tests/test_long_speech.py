"""A long Telugu line must not hold a speech slot for every piece at once."""

from __future__ import annotations

import numpy as np

from telugu_speech.api.routes_openai import _SPEECH_GATE, _speech_pieces_guarded
from telugu_speech.api.schemas import OpenAISpeechRequest


def test_long_line_frees_the_slot_between_pieces(monkeypatch) -> None:
    active_while_speaking: list[int] = []

    class Backend:
        sample_rate_hz = 16000

        def iter_pcm(self, text: str):
            active_while_speaking.append(_SPEECH_GATE._active)
            yield np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(
        "telugu_speech.api.routes_openai._checked_tts_backend",
        lambda body: (Backend(), body.input, "telugu_srinivas"),
    )
    body = OpenAISpeechRequest(
        model="vaani-tts-hima",
        input="నమస్కారం " * 80,
        voice="srinivas",
        response_format="wav",
    )
    pieces = list(_speech_pieces_guarded(body))
    assert len(pieces) > 1
    assert active_while_speaking == [1] * len(pieces)
    assert _SPEECH_GATE._active == 0
