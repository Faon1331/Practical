"""Этап 2. TF-IDF по словам и биграммам (парам соседних слов).

Биграммы помогают увидеть сочетания вроде "не вкусно" или "детский комната"
целиком.
"""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from common import FEATURES, load_split

train, test = load_split()

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
x_train = vectorizer.fit_transform(train["text_clean"])
x_test = vectorizer.transform(test["text_clean"])
joblib.dump({"vectorizer": vectorizer, "X_train": x_train, "X_test": x_test},
            FEATURES / "tfidf_bigrams.joblib")

n_bigrams = sum(" " in term for term in vectorizer.vocabulary_)
print(f"признаков: {len(vectorizer.vocabulary_)}, из них биграмм: {n_bigrams}")
print("сохранено: data/features/tfidf_bigrams.joblib")
