"""Detection Signal 2 — Stylometric heuristic analyzer (M4).

Pure standard library (no external dependencies). Independent of Signal 1.

    stylometric_analyze(text: str) -> {"ai_likeness_score": float in [0, 1]}

    0.0 = very human-like writing style
    1.0 = very AI-like writing style

The score is the mean of four heuristic components, each expressed as an
"AI-likeness" value in [0, 1]:
  1. Sentence-length variation  — humans vary sentence length; AI is uniform.
  2. Vocabulary diversity (TTR) — humans use more distinctive vocabulary.
  3. Repetition frequency       — AI text repeats words/structure more.
  4. Punctuation diversity      — humans use a wider variety of punctuation.
"""

from __future__ import annotations

import re
from collections import Counter

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"[.!?]+")
_PUNCT_RE = re.compile(r"[.,;:!?\-—()\"'/…]")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _pstdev(values: list[float]) -> float:
    mu = _mean(values)
    return (sum((v - mu) ** 2 for v in values) / len(values)) ** 0.5


def stylometric_analyze(text: str) -> dict:
    """Return a feature-based AI-likeness score in [0, 1] for `text`."""
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    words = _WORD_RE.findall(text.lower())
    # Too little signal to be meaningful -> neutral (favors uncertainty).
    if len(words) < 2:
        return {"ai_likeness_score": 0.5}

    # 1. Sentence-length variation. Low coefficient of variation -> AI-like.
    sentences = [s for s in _SENT_RE.split(text) if s.strip()]
    sent_lengths = [len(_WORD_RE.findall(s)) for s in sentences]
    sent_lengths = [n for n in sent_lengths if n > 0]
    if len(sent_lengths) >= 2 and _mean(sent_lengths) > 0:
        cv = _pstdev(sent_lengths) / _mean(sent_lengths)
        # cv ~0.6+ reads as fully human-like variety.
        ai_variation = 1.0 - _clamp01(cv / 0.6)
    else:
        ai_variation = 0.5  # single sentence: not enough to judge -> neutral

    # 2. Vocabulary diversity (type-token ratio). Low diversity -> AI-like.
    ttr = len(set(words)) / len(words)
    ai_vocab = 1.0 - ttr

    # 3. Repetition frequency: share of tokens that are repeats of a word seen
    #    earlier in the text. More repetition -> AI-like.
    counts = Counter(words)
    repeated_tokens = sum(c - 1 for c in counts.values() if c > 1)
    ai_repetition = _clamp01(repeated_tokens / len(words))

    # 4. Punctuation diversity. Fewer distinct marks -> AI-like.
    distinct_punct = len(set(_PUNCT_RE.findall(text)))
    # 5+ distinct punctuation marks reads as fully human-like variety.
    ai_punct = 1.0 - _clamp01(distinct_punct / 5.0)

    score = (ai_variation + ai_vocab + ai_repetition + ai_punct) / 4.0
    return {"ai_likeness_score": round(_clamp01(score), 4)}


if __name__ == "__main__":
    samples = [
        "ok",
        "had the best ramen tonight, no notes!! the broth?? unreal. went back for "
        "seconds — don't judge me.",
        "In conclusion, the proposed framework provides a comprehensive solution. "
        "The proposed framework is scalable. The proposed framework is robust. The "
        "framework delivers value.",
    ]
    for s in samples:
        print(repr(s[:45]), "->", stylometric_analyze(s))
