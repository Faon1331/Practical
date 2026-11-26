"""Этап 3. Подбор порога для каждой метки.

Порог 0,5 не лучший: у редких меток вероятности почти всегда меньше.
Обучающую выборку делим ещё раз 80/20, модель учим на 80%, а на оставшихся
20% для каждой метки подбираем порог из 0,1 ... 0,9 с наибольшим F1.
Тестовая выборка в подборе не участвует. Пороги применяем к модели из
09_logreg.py (обучена на всей обучающей выборке).

Итоговая модель (векторизатор + классификатор + пороги) сохраняется
в models/final_model.joblib, её же использует веб-сервис.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier

from common import LABELS, MODELS, RESULTS, display, load_features, load_prediction, load_split, save_prediction

GRID = np.round(np.arange(0.1, 0.91, 0.1), 1)

train, _ = load_split()
y_train = train[LABELS].to_numpy()
f = load_features("tfidf_bigrams")

fit = train.sample(frac=0.8, random_state=42).index.to_numpy()
val = np.setdiff1d(np.arange(len(train)), fit)

model = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"), n_jobs=-1)
model.fit(f["X_train"][fit], y_train[fit])
proba_val = model.predict_proba(f["X_train"][val])

thresholds = []
for i in range(len(LABELS)):
    scores = [f1_score(y_train[val, i], proba_val[:, i] >= t, zero_division=0) for t in GRID]
    thresholds.append(float(GRID[int(np.argmax(scores))]))
thresholds = np.array(thresholds)

pd.DataFrame({"метка": [display(c) for c in LABELS], "порог": thresholds}) \
    .to_csv(RESULTS / "thresholds.csv", index=False)

base = load_prediction("logreg_bigrams_balanced")
save_prediction("final", base["proba"] >= thresholds, base["proba"])

joblib.dump({
    "vectorizer": f["vectorizer"],
    "model": joblib.load(MODELS / "logreg_bigrams_balanced.joblib"),
    "thresholds": thresholds,
    "labels": LABELS,
}, MODELS / "final_model.joblib", compress=3)

for col, t in zip(LABELS, thresholds):
    print(f"{display(col):<20} {t:.1f}")
print("сохранено: results/thresholds.csv, models/final_model.joblib")
