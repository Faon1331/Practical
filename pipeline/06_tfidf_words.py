"""Этап 2. TF-IDF по словам.

Две матрицы: по исходному тексту (базовое решение, шаг 1) и по тексту
после лемматизации и удаления стоп-слов (все остальные модели).
"""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from common import FEATURES, load_split

train, test = load_split()

for name, column in [("tfidf_raw", "text"), ("tfidf_words", "text_clean")]:
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(train[column])
    x_test = vectorizer.transform(test[column])
    joblib.dump({"vectorizer": vectorizer, "X_train": x_train, "X_test": x_test},
                FEATURES / f"{name}.joblib")
    print(f"{name}: словарь {len(vectorizer.vocabulary_)} слов, матрица {x_train.shape}")

print("сохранено: data/features/tfidf_raw.joblib, tfidf_words.joblib")
