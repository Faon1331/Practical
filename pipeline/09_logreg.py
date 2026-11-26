"""Этап 3. Логистическая регрессия.

По одной модели "да/нет" на каждую из 19 меток (OneVsRest). Варианты:
- logreg_raw - TF-IDF по исходному тексту, порог 0,5 (базовое решение);
- logreg - TF-IDF после лемматизации;
- logreg_balanced - то же с весами классов (balanced);
- logreg_w2v_balanced - Word2Vec, с весами классов;
- logreg_bigrams_balanced - TF-IDF с биграммами, с весами классов
  (вероятности нужны для подбора порогов в 12_thresholds.py).
"""
import time

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from common import LABELS, MODELS, load_features, load_split, save_prediction

train, _ = load_split()
y_train = train[LABELS].to_numpy()

variants = [
    ("logreg_raw", "tfidf_raw", None),
    ("logreg", "tfidf_words", None),
    ("logreg_balanced", "tfidf_words", "balanced"),
    ("logreg_w2v_balanced", "word2vec", "balanced"),
    ("logreg_bigrams_balanced", "tfidf_bigrams", "balanced"),
]

for name, features, weights in variants:
    start = time.time()
    f = load_features(features)
    model = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight=weights), n_jobs=-1)
    model.fit(f["X_train"], y_train)
    proba = model.predict_proba(f["X_test"])
    save_prediction(name, proba >= 0.5, proba)
    if name == "logreg_bigrams_balanced":
        joblib.dump(model, MODELS / f"{name}.joblib")
    print(f"{name}: обучено за {time.time() - start:.0f} с")

print("сохранено: data/predictions/logreg*.joblib, models/logreg_bigrams_balanced.joblib")
