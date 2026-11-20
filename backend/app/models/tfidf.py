from pathlib import Path

import joblib
import numpy as np

from app.preprocessing import preprocess_text


class TfidfClassifier:
    """TF-IDF + логистическая регрессия (по модели на каждую метку).

    Файл модели создаёт scripts/train_tfidf.py.
    """

    name = "tfidf"

    def __init__(self, path: Path):
        bundle = joblib.load(path)
        self.vectorizer = bundle["vectorizer"]
        self.model = bundle["model"]
        self.labels = list(bundle["labels"])
        self.thresholds = np.asarray(bundle["thresholds"], dtype=float)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        clean = [preprocess_text(t) for t in texts]
        features = self.vectorizer.transform(clean)
        return self.model.predict_proba(features)
