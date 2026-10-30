"""Train the TF-IDF text model on the LIAR benchmark (Wang, ACL 2017).

    python -m src.train_text

Binary mapping (same as notebooks/02_bert_ensemble_lime.ipynb):
    REAL = true, mostly-true, half-true
    FAKE = barely-true, false, pants-fire
Positive class for all metrics = FAKE.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
LIAR_DIR = ROOT / "data" / "liar"
TEXT_MODEL_PATH = ROOT / "models" / "text_model.joblib"
TEXT_METRICS_PATH = ROOT / "reports" / "text_metrics.json"
REAL_LABELS = {"true", "mostly-true", "half-true"}
POS = "FAKE"
SEED = 42


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(LIAR_DIR / f"{name}.tsv", sep="\t", header=None, quoting=3, dtype=str)
    df = df[[1, 2, 7]].rename(columns={1: "raw_label", 2: "statement", 7: "party"})
    df = df.dropna(subset=["raw_label", "statement"])
    df["clean"] = df["statement"].map(clean)
    df = df[df["clean"].str.len() > 0]
    df["label"] = df["raw_label"].map(lambda v: "REAL" if v in REAL_LABELS else "FAKE")
    return df.reset_index(drop=True)


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, stop_words="english", min_df=2)),
        ("clf", LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)),
    ])


def _metrics(y, pred, p_fake):
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision_fake": round(precision_score(y, pred, pos_label=POS), 4),
        "recall_fake": round(recall_score(y, pred, pos_label=POS), 4),
        "f1_fake": round(f1_score(y, pred, pos_label=POS), 4),
        "macro_f1": round(f1_score(y, pred, average="macro"), 4),
        "roc_auc": round(roc_auc_score((y == POS).astype(int), p_fake), 4),
    }


def main(quiet: bool = False) -> dict:
    log = (lambda *a: None) if quiet else print
    tr, va, te = (load_split(s) for s in ("train", "valid", "test"))
    pipe = make_pipeline().fit(tr["clean"], tr["label"])
    idx = list(pipe.classes_).index(POS)

    out = {"rows": {"train": len(tr), "valid": len(va), "test": len(te)}}
    for name, d in (("valid", va), ("test", te)):
        p = pipe.predict_proba(d["clean"])[:, idx]
        out[name] = _metrics(d["label"], pipe.predict(d["clean"]), p)
    cm = confusion_matrix(te["label"], pipe.predict(te["clean"]), labels=["REAL", "FAKE"])
    out["test"]["confusion_matrix"] = {"labels": ["REAL", "FAKE"], "matrix": cm.tolist()}
    out["majority_baseline_test_accuracy"] = round(float((te["label"] == "REAL").mean()), 4)

    # party fairness check (flag rate and FPR per party on test set)
    pred = pd.Series(pipe.predict(te["clean"]))
    rows = []
    for party in ["democrat", "republican", "none"]:
        m = (te["party"] == party).to_numpy()
        if m.sum() < 30:
            continue
        real = (te["label"][m] == "REAL").to_numpy()
        pr = pred[m].to_numpy()
        rows.append({"party": party, "n": int(m.sum()),
                     "flag_rate": round(float((pr == POS).mean()), 4),
                     "false_positive_rate": round(float(((pr == POS) & real).sum() / max(real.sum(), 1)), 4)})
    out["fairness_by_party"] = rows
    out["sklearn_version"] = sklearn.__version__

    # final model: fit on train + valid for deployment
    full = pd.concat([tr, va])
    final = make_pipeline().fit(full["clean"], full["label"])
    TEXT_MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": final, "sklearn_version": sklearn.__version__, "classes": list(final.classes_)}, TEXT_MODEL_PATH)
    TEXT_METRICS_PATH.parent.mkdir(exist_ok=True)
    TEXT_METRICS_PATH.write_text(json.dumps(out, indent=2))
    log(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
