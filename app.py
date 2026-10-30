"""Flask web app + JSON API for the Fake News Detector.

Run locally:   python app.py            -> http://127.0.0.1:5000
Production:    gunicorn app:app
"""
from __future__ import annotations

import os
import time

from flask import Flask, jsonify, render_template, request

from src import explain, features, model_store

MAX_CHARS = 20_000
UNCERTAIN_BAND = (0.40, 0.60)
MIN_WORDS_STYLE = 200  # stylometric model was trained on 200-2000 word articles
DISCLAIMER = ("Decision-support only. This tool scores writing style / wording patterns; it does not check facts "
              "and must not be used to flag political content without human review.")


def verdict(p_fake: float) -> str:
    lo, hi = UNCERTAIN_BAND
    if p_fake >= hi:
        return "LIKELY FAKE-STYLE"
    if p_fake <= lo:
        return "LIKELY REAL-STYLE"
    return "UNCERTAIN - NEEDS HUMAN REVIEW"


def create_app() -> Flask:
    app = Flask(__name__)
    style = model_store.load_style_model()
    text_art = model_store.load_text_model()
    metrics = model_store.load_metrics()

    def p_fake(model, X):
        return float(model.predict_proba(X)[0, list(model.classes_).index("FAKE")])

    @app.get("/")
    def index():
        return render_template("index.html", metrics=metrics, has_text_model=text_art is not None)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", style_model=True, text_model=text_art is not None)

    @app.get("/api/metrics")
    def get_metrics():
        return jsonify(metrics)

    @app.post("/api/predict")
    def predict():
        t0 = time.perf_counter()
        body = request.get_json(silent=True) or {}
        text = body.get("text")
        manual = body.get("features")
        result = {"disclaimer": DISCLAIMER, "warnings": []}
        probs = []

        if manual is not None:  # slider mode: 4 numbers
            try:
                feats = {k: float(manual[k]) for k in features.RAW_FEATURES}
            except (KeyError, TypeError, ValueError):
                return jsonify(error="features must contain numeric " + ", ".join(features.RAW_FEATURES)), 400
            feats["word_count"] = int(feats["word_count"]); feats["exclamation_count"] = int(feats["exclamation_count"])
        elif isinstance(text, str) and text.strip():
            if len(text) > MAX_CHARS:
                return jsonify(error=f"Text too long (max {MAX_CHARS} characters)."), 413
            feats = features.extract_features(text)
        else:
            return jsonify(error="Send JSON with a non-empty 'text' string (or a 'features' object)."), 400

        result["features"] = feats

        use_style = manual is not None or feats["word_count"] >= MIN_WORDS_STYLE
        if use_style:
            ps = p_fake(style["pipeline"], features.to_frame(feats))
            result["style_model"] = {
                "p_fake": round(ps, 4),
                "explanation": explain.explain_tabular(style["pipeline"], feats, style["stats"]),
            }
            result["warnings"] += features.out_of_range_warnings(feats)
            probs.append(ps)
        else:
            result["style_model"] = None
            result["warnings"].append(
                f"Only {feats['word_count']} words: the style model needs at least {MIN_WORDS_STYLE} words "
                "(it was trained on full articles), so it was skipped.")

        if text_art is not None and manual is None:
            pt = p_fake(text_art["pipeline"], [text.lower()])
            result["text_model"] = {"p_fake": round(pt, 4),
                                    "explanation": explain.explain_text(text_art["pipeline"], text)}
            probs.append(pt)
        else:
            result["text_model"] = None

        final = sum(probs) / len(probs)
        result["p_fake"] = round(final, 4)
        result["verdict"] = verdict(final)
        result["confidence"] = round(max(final, 1 - final), 4)
        result["models_used"] = [m for m, k in (("style", result["style_model"]), ("text", result["text_model"])) if k]
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return jsonify(result)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
