"""LIME-style local explanations (no external `lime` dependency needed).

Idea (Ribeiro et al., 2016): perturb the input, ask the black-box model for its
prediction on each perturbation, then fit a small weighted linear model around
the original instance. The linear model's coefficients are the explanation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .features import FEATURE_LABELS, RAW_FEATURES

POS = "FAKE"


def _fake_idx(model) -> int:
    return list(model.classes_).index(POS)


def explain_tabular(pipeline, features: dict, stats: dict, n: int = 1500, seed: int = 0) -> dict:
    """Why did the stylometric model say what it said? Contribution of each feature (in probability points)."""
    rng = np.random.default_rng(seed)
    mu = np.array([stats[c]["mean"] for c in RAW_FEATURES])
    sd = np.array([stats[c]["std"] for c in RAW_FEATURES])
    lo = np.array([stats[c]["min"] for c in RAW_FEATURES])
    hi = np.array([stats[c]["max"] for c in RAW_FEATURES])
    x = np.array([features[c] for c in RAW_FEATURES], dtype=float)
    x_c = np.clip(x, lo, hi)  # explain around the in-range point

    X = np.clip(x_c + rng.normal(0, 0.5, (n, len(x))) * sd, lo, hi)
    X[0] = x_c
    p = pipeline.predict_proba(pd.DataFrame(X, columns=RAW_FEATURES))[:, _fake_idx(pipeline)]
    S = (X - mu) / sd
    d2 = (((X - x_c) / sd) ** 2).sum(axis=1)
    w = np.exp(-d2 / (1.5 ** 2))
    sur = Ridge(alpha=1.0).fit(S, p, sample_weight=w)
    z = (x_c - mu) / sd
    contrib = sur.coef_ * z
    items = [{"feature": c, "label": FEATURE_LABELS[c], "value": features[c],
              "contribution": round(float(v), 4)} for c, v in zip(RAW_FEATURES, contrib)]
    items.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return {"baseline_p_fake": round(float(sur.intercept_), 4), "items": items,
            "fidelity_r2": round(float(sur.score(S, p, sample_weight=w)), 3)}


def explain_text(pipeline, text: str, n: int = 600, top_k: int = 8, seed: int = 0) -> dict:
    """Which words pushed the TF-IDF model toward FAKE (+) or REAL (-)?"""
    rng = np.random.default_rng(seed)
    words = text.split()
    vocab = list(dict.fromkeys(w.strip(".,;:!?\"'()[]").lower() for w in words))
    vocab = [w for w in vocab if w][:250]
    if len(vocab) < 2:
        return {"items": [], "fidelity_r2": None}
    masks = rng.integers(0, 2, (n, len(vocab)))
    masks[0] = 1
    keep = [set(np.array(vocab)[m.astype(bool)]) for m in masks]
    texts = [" ".join(w for w in words if w.strip(".,;:!?\"'()[]").lower() in k) for k in keep]
    p = pipeline.predict_proba(texts)[:, _fake_idx(pipeline)]
    removed = 1 - masks.mean(axis=1)
    w = np.exp(-(removed ** 2) / 0.25)
    sur = Ridge(alpha=1.0).fit(masks, p, sample_weight=w)
    order = np.argsort(-np.abs(sur.coef_))[:top_k]
    items = [{"word": vocab[i], "weight": round(float(sur.coef_[i]), 4)} for i in order if abs(sur.coef_[i]) > 1e-4]
    return {"items": items, "fidelity_r2": round(float(sur.score(masks, p, sample_weight=w)), 3)}
