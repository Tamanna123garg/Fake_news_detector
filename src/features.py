"""Turn raw article text into the four stylometric features used in dataset.csv.

    word_count, sentiment_score, exclamation_count, all_caps_pct

A tiny rule-based sentiment lexicon is bundled so the app has NO external
dependency or download (works offline and on any free host).
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

RAW_FEATURES = ["word_count", "sentiment_score", "exclamation_count", "all_caps_pct"]
MODEL_FEATURES = RAW_FEATURES + ["abs_sentiment", "exclam_per_100w"]

FEATURE_LABELS = {
    "word_count": "Article length (words)",
    "sentiment_score": "Sentiment polarity",
    "exclamation_count": "Exclamation marks",
    "all_caps_pct": "ALL-CAPS words (share)",
}

# Range the model was trained on (matches dataset.csv / data_dictionary.csv)
TRAIN_RANGES = {
    "word_count": (200, 2000),
    "sentiment_score": (-1.0, 1.0),
    "exclamation_count": (0, 15),
    "all_caps_pct": (0.0, 0.30),
}

POSITIVE = set("""
good great excellent amazing wonderful positive success successful win wins won improve improved improvement
benefit benefits happy hope hopeful progress strong stronger best better support supported praise praised
agree agreement peace safe safety stable growth gain gains thrive celebrate celebrated approve approved
confident reliable fair honest trust trusted effective helpful helped brilliant miracle outstanding
secure victory proud love loved delighted encouraging promising recover recovered recovery boost boosted
achieve achieved achievement
""".split())

NEGATIVE = set("""
bad terrible horrible awful negative fail failed failure lose lost loss crisis disaster scandal outrage
outraged shocking shocked fear fears afraid danger dangerous threat threats corrupt corruption hoax lie lies
lied lying fraud fake fraudulent attack attacked attacks destroy destroyed destroying collapse collapsed
cover-up coverup betray betrayed betrayal evil criminal crime crimes cheat cheated cheating stolen steal
hate hated angry anger violent violence war panic chaos worst worse decline declined declining ruin ruined
disgrace disgusting conspiracy secret secretly exposed warning deadly death dead kill killed killing
toxic poison poisoned banned blame blamed
""".split())

NEGATORS = {"not", "no", "never", "n't", "without", "hardly", "neither", "nor", "cannot", "cant", "dont", "isnt", "wasnt", "didnt"}

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")
_ALPHA_RE = re.compile(r"[A-Za-z]")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def sentiment_score(text: str) -> float:
    """Lexicon polarity in [-1, 1] (negation aware, length-normalised)."""
    toks = [t.lower().strip("'’-") for t in _tokens(text)]
    if not toks:
        return 0.0
    total = 0.0
    for i, w in enumerate(toks):
        val = 1.0 if w in POSITIVE else -1.0 if w in NEGATIVE else 0.0
        if val:
            for back in (1, 2):  # negator directly before, or one word earlier (not a sentiment word)
                if i - back < 0:
                    break
                prev = toks[i - back]
                if prev in NEGATORS or prev.endswith("n't"):
                    val *= -0.75
                    break
                if prev in POSITIVE or prev in NEGATIVE:
                    break
        total += val
    density = total / len(toks)
    return float(math.tanh(12.0 * density))


def extract_features(text: str) -> dict:
    """Return the 4 raw dataset features for a piece of text."""
    toks = _tokens(text)
    n = len(toks)
    caps = sum(1 for t in toks if len(t) >= 2 and _ALPHA_RE.search(t) and t.upper() == t and not t.isdigit())
    return {
        "word_count": int(n),
        "sentiment_score": round(sentiment_score(text), 3),
        "exclamation_count": int(text.count("!")),
        "all_caps_pct": round(caps / n, 3) if n else 0.0,
    }


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Raw 4 columns -> 6 model columns. Module-level so joblib can pickle the pipeline."""
    out = pd.DataFrame(df, columns=RAW_FEATURES).astype(float).copy()
    out["abs_sentiment"] = out["sentiment_score"].abs()
    out["exclam_per_100w"] = out["exclamation_count"] / out["word_count"].clip(lower=1) * 100.0
    return out[MODEL_FEATURES].to_numpy()


def to_frame(features: dict) -> pd.DataFrame:
    return pd.DataFrame([[features[c] for c in RAW_FEATURES]], columns=RAW_FEATURES)


def out_of_range_warnings(features: dict) -> list[str]:
    warns = []
    for k, (lo, hi) in TRAIN_RANGES.items():
        v = features[k]
        if v < lo or v > hi:
            warns.append(
                f"{FEATURE_LABELS[k]} = {v} is outside the training range ({lo}-{hi}); the prediction is less reliable."
            )
    return warns


if __name__ == "__main__":  # tiny smoke test
    print(extract_features("BREAKING!!! The government is HIDING a shocking secret. Not good!"))
