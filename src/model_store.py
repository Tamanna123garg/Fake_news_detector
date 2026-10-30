"""Load the saved models; retrain automatically if missing or built with another scikit-learn version."""
from __future__ import annotations

import json

import joblib
import sklearn

from . import train, train_text


def _load(path, retrain):
    try:
        art = joblib.load(path)
        if art.get("sklearn_version") != sklearn.__version__:
            raise RuntimeError("scikit-learn version changed")
        return art
    except Exception:
        retrain(quiet=True)
        return joblib.load(path)


def load_style_model():
    return _load(train.MODEL_PATH, train.main)


def load_text_model():
    """Returns None if the LIAR files are not in data/liar (app still works without it)."""
    if not (train_text.LIAR_DIR / "train.tsv").exists():
        return None
    return _load(train_text.TEXT_MODEL_PATH, train_text.main)


def load_metrics() -> dict:
    out = {}
    for key, path in (("style", train.METRICS_PATH), ("text", train_text.TEXT_METRICS_PATH)):
        if path.exists():
            out[key] = json.loads(path.read_text())
    return out
