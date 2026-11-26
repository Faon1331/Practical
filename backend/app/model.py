from pathlib import Path

import joblib
import numpy as np

from app.config import preprocess_text


class Classifier:
    """Итоговая модель: TF-IDF (слова и биграммы) + логистическая регрессия
    с весами классов и своим порогом для каждой метки.

    Файл создаёт pipeline/12_thresholds.py.
    """

    def __init__(self, path: Path):
        bundle = joblib.load(path)
        self.vectorizer = bundle["vectorizer"]
        self.model = bundle["model"]
        self.labels = list(bundle["labels"])
        self.thresholds = np.asarray(bundle["thresholds"], dtype=float)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        clean = [preprocess_text(t) for t in texts]
        return self.model.predict_proba(self.vectorizer.transform(clean))
