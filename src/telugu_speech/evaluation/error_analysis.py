"""Error analysis over bench/results JSON — no new metrics invented.

Reads the JSON written by ``telugu_speech.bench`` (``bench/results/<model>/<date>.json``,
shape ``{"models": [...]}`` — see evaluation/benchmark.py) and reports
error *classes* from what that JSON actually contains:

- code-mix: the ``code_mix: yes`` slice and the ``latin_wer`` on it.
  The measured anchor is the Andhra set: English recall 0 of 53.
- numerals: ``numeral_wer`` on the overall slice.
- noisy / phone: the ``condition`` slices (clean vs noisy vs phone 8 kHz
  vs conversational) — a slice is only reported when it has clips.
- region: the ``region`` slices (Telangana / Coastal Andhra / Rayalaseema).
- script-collapse proxy: overall WER running far above AER on the same
  clips (Telugu WER runs ~3-4x AER because one wrong matra voids a whole
  word). This is a *proxy*, labelled as such — it is not a token-level
  diagnosis.
- high-WER utterances: clips with WER >= 0.5, counted, not re-scored.

What this file cannot do, honestly: bench JSON stores per-utterance
*rates* only, not reference/hypothesis text, so there is no token-level
"proper nouns" count here. Proper-noun errors are reported only when a
caller supplies reference/hypothesis pairs (see ``analyse_pairs``),
counted by the existing ``edit_counts`` — and even then a token is
called a proper noun only if it is capitalised Latin or matches a
supplied proper-noun list. No name list is invented here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ErrorClass:
    name: str
    clips: int
    rate: float | None
    evidence: str


@dataclass
class ErrorReport:
    model_id: str
    date: str | None
    clips: int
    wer: float | None
    cer: float | None
    aer: float | None
    classes: list[ErrorClass] = field(default_factory=list)
    high_wer_utterances: int = 0
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"Error analysis — {self.model_id} ({self.date or 'date unknown'}, n={self.clips})",
            f"Overall: WER {self.wer}, CER {self.cer}, AER {self.aer}",
            "",
        ]
        for item in self.classes:
            rate = "n/a" if item.rate is None else f"{item.rate:.4f}"
            lines.append(f"- {item.name}: {item.clips} clips, rate {rate} — {item.evidence}")
        lines.append(f"- high-WER utterances (WER >= 0.5): {self.high_wer_utterances}")
        for note in self.notes:
            lines.append(f"Note: {note}")
        return "\n".join(lines) + "\n"


def _slice(rows: list[dict[str, object]], kind: str, name: str) -> dict[str, object] | None:
    for row in rows:
        if row.get("kind") == kind and row.get("name") == name:
            return row
    return None


def _slices(rows: list[dict[str, object]], kind: str) -> list[dict[str, object]]:
    return [row for row in rows if row.get("kind") == kind]


def _rate(value: object) -> float | None:
    return value if isinstance(value, (int, float)) else None


def analyse_bench_json(path: Path) -> ErrorReport:
    """Analyse one bench JSON file (``{"models": [...]}`` or a single model dict)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models") if isinstance(payload, dict) else None
    if isinstance(models, list) and models:
        model = models[0]
    elif isinstance(payload, dict):
        model = payload
    else:
        raise ValueError(f"{path} is not a bench results JSON file")
    slices = model.get("slices") if isinstance(model.get("slices"), list) else []
    utterances = model.get("utterances") if isinstance(model.get("utterances"), list) else []
    report = ErrorReport(
        model_id=str(model.get("model_id", path.stem)),
        date=str(model.get("date")) if model.get("date") else None,
        clips=int(model.get("clips") or len(utterances)),
        wer=_rate(model.get("wer")),
        cer=_rate(model.get("cer")),
        aer=_rate(model.get("aer")),
    )
    overall = _slice(slices, "overall", "all")
    if overall is not None:
        numeral = _rate(overall.get("numeral_wer"))
        if numeral is not None:
            report.classes.append(
                ErrorClass("numerals", report.clips, numeral, "numeral_wer on the overall slice")
            )
        latin = _rate(overall.get("latin_wer"))
        if latin is not None:
            report.classes.append(
                ErrorClass(
                    "code-mix (Latin tokens)",
                    report.clips,
                    latin,
                    "latin_wer on the overall slice; measured anchor: English recall 0 of 53 on 16 Andhra clips",
                )
            )
    code_mix = _slice(slices, "code_mix", "yes")
    if code_mix is not None and int(code_mix.get("clips") or 0) > 0:
        report.classes.append(
            ErrorClass(
                "code-mix sentences",
                int(code_mix.get("clips") or 0),
                _rate(code_mix.get("wer")),
                "WER on the code_mix=yes slice (vs Telugu-only WER 0.4394 on the Andhra set)",
            )
        )
    for row in _slices(slices, "condition"):
        if row.get("name") in {"clean", "all"} or int(row.get("clips") or 0) == 0:
            continue
        report.classes.append(
            ErrorClass(
                f"condition: {row.get('name')}",
                int(row.get("clips") or 0),
                _rate(row.get("wer")),
                "WER on this condition slice",
            )
        )
    for row in _slices(slices, "region"):
        if int(row.get("clips") or 0) == 0:
            continue
        report.classes.append(
            ErrorClass(
                f"region: {row.get('name')}",
                int(row.get("clips") or 0),
                _rate(row.get("wer")),
                "WER on this region slice",
            )
        )
    if report.wer is not None and report.aer is not None and report.aer > 0:
        ratio = report.wer / report.aer
        report.classes.append(
            ErrorClass(
                "script-collapse proxy",
                report.clips,
                ratio,
                f"WER/AER ratio {ratio:.1f}x (Telugu WER normally runs ~3-4x AER; far above that suggests script collapse, not word errors). Rate here is the ratio, not an error rate.",
            )
        )
    report.high_wer_utterances = sum(1 for row in utterances if _rate(row.get("wer")) is not None and _rate(row.get("wer")) >= 0.5)  # type: ignore[operator]
    if not utterances:
        report.notes.append("No per-utterance rows in this JSON; high-WER count is 0 by absence, not by measurement.")
    report.notes.append(
        "Proper nouns are NOT counted from bench JSON (it stores rates, not text). "
        "Use analyse_pairs() with reference/hypothesis pairs and a supplied proper-noun list."
    )
    return report


def analyse_pairs(
    pairs: list[tuple[str, str]], proper_nouns: frozenset[str] | None = None
) -> dict[str, object]:
    """Token-level error classes for supplied (reference, hypothesis) pairs.

    Counts substitutions/deletions/insertions per class using the repo's
    own edit-distance (``asr_metrics.edit_counts``). A token counts as a
    proper noun only if it is in ``proper_nouns`` or is a capitalised
    Latin token. No name list is invented: with ``proper_nouns=None``
    only capitalised Latin tokens count, and Telugu proper nouns are
    reported as not counted.
    """
    from telugu_speech.evaluation.asr_metrics import edit_counts
    from telugu_speech.text.normalization import tokens

    classes: dict[str, list[int]] = {
        "numerals": [0, 0],
        "latin_code_mix": [0, 0],
        "proper_nouns": [0, 0],
        "other_telugu": [0, 0],
    }
    for reference, hypothesis in pairs:
        ref_tokens, hyp_tokens = tokens(reference), tokens(hypothesis)
        for token in ref_tokens:
            if any(char.isdigit() for char in token):
                classes["numerals"][1] += 1
            elif any("A" <= c <= "Z" or "a" <= c <= "z" for c in token):
                classes["latin_code_mix"][1] += 1
                if proper_nouns and token in proper_nouns:
                    classes["proper_nouns"][1] += 1
                elif token[:1].isupper():
                    classes["proper_nouns"][1] += 1
            else:
                classes["other_telugu"][1] += 1
        # Per-class error counts come from the class-filtered token lists.
        def is_proper_noun(token: str) -> bool:
            return bool(proper_nouns and token in proper_nouns) or token[:1].isupper()

        for name, pattern in (
            ("numerals", lambda t: any(c.isdigit() for c in t)),
            ("latin_code_mix", lambda t: any("A" <= c <= "Z" or "a" <= c <= "z" for c in t)),
            ("proper_nouns", is_proper_noun),
            (
                "other_telugu",
                lambda t: not any(c.isdigit() for c in t)
                and not any("A" <= c <= "Z" or "a" <= c <= "z" for c in t),
            ),
        ):
            ref = [t for t in ref_tokens if pattern(t)]
            hyp = [t for t in hyp_tokens if pattern(t)]
            counts = edit_counts(ref, hyp)
            classes[name][0] += counts.substitutions + counts.deletions + counts.insertions
    return {
        "pairs": len(pairs),
        "classes": {
            name: {"errors": errors, "reference_tokens": total}
            for name, (errors, total) in classes.items()
        },
        "note": "proper_nouns counts capitalised Latin tokens only unless a proper-noun list is supplied; Telugu proper nouns are not counted here.",
    }
