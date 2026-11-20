import json
from pathlib import Path

import numpy as np

from app.config import LABEL_COLUMNS


class BertClassifier:
    """Дообученный ruBERT из ноутбука 2Gis_V2.ipynb.

    Нужна папка, сохранённая через trainer.save_model() и
    tokenizer.save_pretrained(). Если рядом лежит thresholds.json
    (список из 19 чисел), используются эти пороги, иначе 0.5.
    torch и transformers ставятся отдельно, см. README.
    """

    name = "bert"

    def __init__(self, path: Path, max_length: int = 128, batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size
        self.labels = list(LABEL_COLUMNS)

        thresholds_file = Path(path) / "thresholds.json"
        if thresholds_file.exists():
            self.thresholds = np.asarray(json.loads(thresholds_file.read_text()), dtype=float)
        else:
            self.thresholds = np.full(len(self.labels), 0.5)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        result = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            enc = self.tokenizer(batch, max_length=self.max_length, padding=True,
                                 truncation=True, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                logits = self.model(**enc).logits
            result.append(self._torch.sigmoid(logits).cpu().numpy())
        return np.vstack(result)
