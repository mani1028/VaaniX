"""IndicTrans2 translation adapter. Lazy download; MIT open weights."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[3]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from telugu_speech.asr.infer import ModelUnavailable

# ISO-like codes used by this platform → IndicTrans2 language tags
INDIC_TAGS = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "or": "ory_Orya",
    "as": "asm_Beng",
    "ur": "urd_Arab",
}


def _pick_model(source: str, target: str) -> str:
    if source == "en" and target != "en":
        return "ai4bharat/indictrans2-en-indic-dist-200M"
    if source != "en" and target == "en":
        return "ai4bharat/indictrans2-indic-en-dist-200M"
    if source != "en" and target != "en":
        return "ai4bharat/indictrans2-indic-indic-dist-320M"
    raise ModelUnavailable("Unsupported translation direction")


@lru_cache(maxsize=2)
def _load_pipeline(model_id: str):
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except Exception as exc:
        raise ModelUnavailable("transformers is required for IndicTrans2") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        raise ModelUnavailable(
            f"Could not load {model_id}. Install sentencepiece and allow one HF download, "
            "or place the model under the Hugging Face cache."
        ) from exc
    return tokenizer, model


def translate(text: str, source: str = "te", target: str = "en") -> dict:
    if source not in INDIC_TAGS or target not in INDIC_TAGS:
        raise ModelUnavailable(f"Unsupported language pair {source}->{target}")
    model_id = _pick_model(source, target)
    tokenizer, model = _load_pipeline(model_id)
    src_tag = INDIC_TAGS[source]
    tgt_tag = INDIC_TAGS[target]
    # IndicTrans2 HF models expect language tags in the text for many checkpoints.
    tagged = f"{src_tag} {text}"
    inputs = tokenizer(tagged, return_tensors="pt", truncation=True, max_length=256)
    outputs = model.generate(**inputs, max_length=256, num_beams=5)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # Some builds prepend the target tag; strip if present.
    cleaned = decoded.replace(tgt_tag, "").strip()
    return {
        "text": cleaned or decoded,
        "source": source,
        "target": target,
        "backend": model_id,
        "license": "MIT",
    }
