"""Этап 2. Word2Vec.

Обучаем векторы слов на обучающей выборке, вектор отзыва - среднее векторов
его слов.
"""
import time

import joblib
import numpy as np
from gensim.models import Word2Vec

from common import FEATURES, load_split

SIZE = 100

start = time.time()
train, test = load_split()

sentences = [t.split() for t in train["text_clean"] if t]
w2v = Word2Vec(sentences=sentences, vector_size=SIZE, window=5, min_count=3,
               workers=4, epochs=10, seed=42)


def to_vector(text):
    vectors = [w2v.wv[w] for w in text.split() if w in w2v.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(SIZE)


x_train = np.vstack([to_vector(t) for t in train["text_clean"]])
x_test = np.vstack([to_vector(t) for t in test["text_clean"]])
joblib.dump({"X_train": x_train, "X_test": x_test}, FEATURES / "word2vec.joblib")

print(f"словарь Word2Vec: {len(w2v.wv)} слов, размер вектора {SIZE}")
print("похожие на 'уютный':", [w for w, _ in w2v.wv.most_similar("уютный", topn=5)])
print(f"готово за {time.time() - start:.0f} с, сохранено: data/features/word2vec.joblib")
