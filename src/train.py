"""Train, evaluate and save the stylometric ensemble.

    python -m src.train

Outputs
-------
models/fake_news_model.joblib   pipeline + training statistics (used by the app)
reports/metrics.json            all metrics shown in the report / synopsis
reports/figures/*.png           confusion matrix, ROC curve, permutation importance, model comparison
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .features import FEATURE_LABELS, RAW_FEATURES, engineer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataset.csv"
MODEL_PATH = ROOT / "models" / "fake_news_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
FIG_DIR = ROOT / "reports" / "figures"
SEED = 42
POS = "FAKE"  # positive class


def load_data(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = RAW_FEATURES + ["label", "political_lean"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"dataset.csv is missing columns: {missing}")
    return df.dropna(subset=need).reset_index(drop=True)


def make_models() -> dict:
    eng = lambda: FunctionTransformer(engineer, validate=False)
    lr = make_pipeline(eng(), StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))
    rf = make_pipeline(eng(), RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1))
    gb = make_pipeline(eng(), GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=SEED))
    ens = Pipeline([
        ("engineer", eng()),
        ("vote", VotingClassifier(
            estimators=[
                ("lr", Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000, random_state=SEED))])),
                ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1)),
                ("gb", GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=SEED)),
            ],
            voting="soft")),
    ])
    return {"Logistic Regression": lr, "Random Forest": rf, "Gradient Boosting": gb, "Ensemble (soft vote)": ens}


def _scores(y_true, y_pred, proba_fake):
    yb = (y_true == POS).astype(int)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, pos_label=POS), 4),
        "recall": round(recall_score(y_true, y_pred, pos_label=POS), 4),
        "f1": round(f1_score(y_true, y_pred, pos_label=POS), 4),
        "roc_auc": round(roc_auc_score(yb, proba_fake), 4),
    }


def fairness_audit(y_true, y_pred, groups) -> list[dict]:
    rows = []
    for g in sorted(groups.unique()):
        m = (groups == g).to_numpy()
        yt, yp = y_true[m], y_pred[m]
        real, fake = (yt != POS), (yt == POS)
        rows.append({
            "political_lean": g, "n": int(m.sum()),
            "false_positive_rate": round(float(((yp == POS) & real).sum() / max(real.sum(), 1)), 4),
            "false_negative_rate": round(float(((yp != POS) & fake).sum() / max(fake.sum(), 1)), 4),
            "flag_rate": round(float((yp == POS).mean()), 4),
        })
    return rows


def _plots(models, best_name, X_te, y_te, proba, y_pred, perm):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # plots are optional
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # confusion matrix
    cm = confusion_matrix(y_te, y_pred, labels=["REAL", "FAKE"])
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["REAL", "FAKE"]); ax.set_yticks([0, 1], ["REAL", "FAKE"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix - {best_name}", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=160); plt.close(fig)
    # ROC
    fig, ax = plt.subplots(figsize=(4.8, 4))
    for name, m in models.items():
        fpr, tpr, _ = roc_curve((y_te == POS).astype(int), m.predict_proba(X_te)[:, list(m.classes_).index(POS)])
        ax.plot(fpr, tpr, label=name, lw=2 if name == best_name else 1)
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (hold-out test set)", fontsize=10); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG_DIR / "roc_curve.png", dpi=160); plt.close(fig)
    # permutation importance
    names = [FEATURE_LABELS[f] for f in RAW_FEATURES]
    order = np.argsort(perm.importances_mean)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.barh(np.array(names)[order], perm.importances_mean[order], xerr=perm.importances_std[order], color="#2f6fed")
    ax.set_xlabel("Drop in accuracy when feature is shuffled"); ax.set_title("Permutation feature importance", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG_DIR / "feature_importance.png", dpi=160); plt.close(fig)


def main(quiet: bool = False) -> dict:
    log = (lambda *a: None) if quiet else print
    df = load_data()
    X, y = df[RAW_FEATURES], df["label"]
    X_tr, X_te, y_tr, y_te, lean_tr, lean_te = train_test_split(
        X, y, df["political_lean"], test_size=0.2, stratify=y, random_state=SEED)
    models = make_models()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name, m in models.items():
        cvr = cross_validate(m, X_tr, y_tr, cv=cv, scoring={"acc": "accuracy", "f1": "f1_macro"})
        m.fit(X_tr, y_tr)
        fitted[name] = m
        pred = m.predict(X_te)
        proba = m.predict_proba(X_te)[:, list(m.classes_).index(POS)]
        results[name] = {**_scores(y_te, pred, proba),
                         "cv_accuracy_mean": round(float(cvr["test_acc"].mean()), 4),
                         "cv_accuracy_std": round(float(cvr["test_acc"].std()), 4)}
        log(f"{name:22s} acc={results[name]['accuracy']:.3f} f1={results[name]['f1']:.3f} auc={results[name]['roc_auc']:.3f}")

    best_name = "Ensemble (soft vote)"
    best = fitted[best_name]
    pred = best.predict(X_te)
    proba = best.predict_proba(X_te)[:, list(best.classes_).index(POS)]
    cm = confusion_matrix(y_te, pred, labels=["REAL", "FAKE"])

    # latency of a single prediction (what the user feels in the web app)
    one = X_te.iloc[[0]]
    best.predict_proba(one)
    t0 = time.perf_counter()
    for _ in range(50):
        best.predict_proba(one)
    latency_ms = (time.perf_counter() - t0) / 50 * 1000

    perm = permutation_importance(best, X_te, y_te, n_repeats=10, random_state=SEED, scoring="accuracy")
    _plots(fitted, best_name, X_te, y_te, proba, pred, perm)

    # sanity check on the ORIGINAL 500 rows only (no synthetic data)
    orig = df.iloc[:500]
    cv500 = cross_validate(make_models()[best_name], orig[RAW_FEATURES], orig["label"], cv=cv, scoring="accuracy")

    # refit the final model on ALL data for deployment
    final = make_models()[best_name].fit(X, y)
    stats = {c: {"mean": float(X[c].mean()), "std": float(X[c].std()), "min": float(X[c].min()), "max": float(X[c].max())}
             for c in RAW_FEATURES}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": final, "stats": stats, "sklearn_version": sklearn.__version__, "classes": list(final.classes_)}, MODEL_PATH)

    metrics = {
        "dataset": {"rows": int(len(df)), "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
                    "fake_share": round(float((y == POS).mean()), 4), "features": RAW_FEATURES},
        "models": results,
        "best_model": best_name,
        "test": {**results[best_name], "confusion_matrix": {"labels": ["REAL", "FAKE"], "matrix": cm.tolist()}},
        "latency_ms_per_prediction": round(latency_ms, 2),
        "original_500_rows_only_cv_accuracy": {"mean": round(float(cv500["test_score"].mean()), 4),
                                                "std": round(float(cv500["test_score"].std()), 4)},
        "permutation_importance": {FEATURE_LABELS[f]: round(float(v), 4) for f, v in zip(RAW_FEATURES, perm.importances_mean)},
        "fairness_by_political_lean": fairness_audit(y_te.reset_index(drop=True), pd.Series(pred), lean_te.reset_index(drop=True)),
        "sklearn_version": sklearn.__version__,
    }
    METRICS_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    log(f"\nBest: {best_name}  test acc={metrics['test']['accuracy']}  f1={metrics['test']['f1']}  latency={latency_ms:.1f} ms")
    log(f"Original-500-only CV accuracy: {metrics['original_500_rows_only_cv_accuracy']}")
    return metrics


if __name__ == "__main__":
    main()
