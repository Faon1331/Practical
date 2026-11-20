"""Обучение TF-IDF + логистической регрессии и подбор порогов по меткам.

Параметры векторизатора и классификатора - как в ноутбуке 2Gis_V2.ipynb.
Пороги подбираются на 10% train (валидация), test в подборе не участвует.

Запуск из папки backend:
    python scripts/train_tfidf.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from app import config
from app.preprocessing import preprocess_text

LABELS = config.LABEL_COLUMNS


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(config.DATASET_DIR / name, usecols=["text", *LABELS])
    df["clean"] = df["text"].map(preprocess_text)
    return df


def best_thresholds(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    grid = np.arange(0.05, 0.96, 0.01)
    result = []
    for i in range(y_true.shape[1]):
        scores = [f1_score(y_true[:, i], proba[:, i] >= t, zero_division=0) for t in grid]
        result.append(round(float(grid[int(np.argmax(scores))]), 2))
    return np.array(result)


def main():
    start = time.time()
    train = load("train.csv")
    test = load("test.csv")
    print(f"train: {len(train)}, test: {len(test)}, предобработка {time.time() - start:.0f} с")

    fit_df, val_df = train_test_split(train, test_size=0.1, random_state=42)

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                 sublinear_tf=True, min_df=3, max_df=0.95)
    x_fit = vectorizer.fit_transform(fit_df["clean"])
    x_val = vectorizer.transform(val_df["clean"])
    x_test = vectorizer.transform(test["clean"])
    y_fit = fit_df[LABELS].to_numpy()
    y_val = val_df[LABELS].to_numpy()
    y_test = test[LABELS].to_numpy()

    model = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"), n_jobs=-1
    )
    model.fit(x_fit, y_fit)
    print(f"обучение заняло {time.time() - start:.0f} с")

    thresholds = best_thresholds(y_val, model.predict_proba(x_val))

    proba = model.predict_proba(x_test)
    pred_05 = (proba >= 0.5).astype(int)
    pred_thr = (proba >= thresholds).astype(int)

    per_label = []
    for i, col in enumerate(LABELS):
        per_label.append({
            "label": config.display_name(col),
            "support": int(y_test[:, i].sum()),
            "threshold": float(thresholds[i]),
            "f1_05": round(f1_score(y_test[:, i], pred_05[:, i], zero_division=0), 4),
            "f1": round(f1_score(y_test[:, i], pred_thr[:, i], zero_division=0), 4),
            "pr_auc": round(average_precision_score(y_test[:, i], proba[:, i]), 4),
            "roc_auc": round(roc_auc_score(y_test[:, i], proba[:, i]), 4),
        })

    summary = {
        "f1_macro_05": round(f1_score(y_test, pred_05, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_test, pred_thr, average="macro", zero_division=0), 4),
        "f1_micro": round(f1_score(y_test, pred_thr, average="micro", zero_division=0), 4),
        "pr_auc_mean": round(float(np.mean([r["pr_auc"] for r in per_label])), 4),
        "roc_auc_mean": round(float(np.mean([r["roc_auc"] for r in per_label])), 4),
    }

    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "vectorizer": vectorizer,
        "model": model,
        "labels": LABELS,
        "thresholds": thresholds.tolist(),
        "sklearn_version": sklearn.__version__,
    }, config.TFIDF_PATH, compress=3)
    np.save(config.REPORTS_DIR / "test_proba_tfidf.npy", proba)
    with open(config.REPORTS_DIR / "metrics_tfidf.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_label": per_label}, f, ensure_ascii=False, indent=2)

    print(f"\n{'метка':<20}{'порог':>7}{'F1@0.5':>9}{'F1':>8}{'PR-AUC':>9}{'ROC-AUC':>9}")
    for r in per_label:
        print(f"{r['label']:<20}{r['threshold']:>7.2f}{r['f1_05']:>9.3f}{r['f1']:>8.3f}"
              f"{r['pr_auc']:>9.3f}{r['roc_auc']:>9.3f}")
    print()
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nмодель сохранена: {config.TFIDF_PATH}, всего {time.time() - start:.0f} с")


if __name__ == "__main__":
    main()
