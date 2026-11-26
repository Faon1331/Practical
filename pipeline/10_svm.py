"""Этап 3. Линейный SVM (без весов классов и с весами balanced)."""
import time

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from common import LABELS, load_features, load_split, save_prediction

train, _ = load_split()
y_train = train[LABELS].to_numpy()
f = load_features("tfidf_words")

for name, weights in [("svm", None), ("svm_balanced", "balanced")]:
    start = time.time()
    model = OneVsRestClassifier(LinearSVC(class_weight=weights), n_jobs=-1)
    model.fit(f["X_train"], y_train)
    save_prediction(name, model.predict(f["X_test"]))
    print(f"{name}: обучено за {time.time() - start:.0f} с")

print("сохранено: data/predictions/svm.joblib, svm_balanced.joblib")
