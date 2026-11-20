"""Строит профили заведений: прогоняет все отзывы через модель и считает,
сколько отзывов каждого заведения получили каждую метку.

Берём именно предсказания модели, а не исходную разметку: так сервис
работал бы с новыми отзывами, у которых разметки нет.

Запуск из папки backend (после train_tfidf.py):
    python scripts/build_venues.py
"""
import ast
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app import config
from app.models import load_classifier


def parse_caption(caption: str) -> tuple[str, str | None]:
    # caption в выгрузке выглядит так: "['Фиалка, кафе']"
    try:
        values = ast.literal_eval(caption)
        text = values[0] if values else ""
    except (ValueError, SyntaxError):
        text = caption or ""
    name, _, kind = text.rpartition(", ")
    if not name:
        return text, None
    return name, kind


def load_names() -> pd.DataFrame:
    with open(config.DATASET_DIR / "task_2_vibes.json", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for item in data:
        name, kind = parse_caption(item.get("caption", ""))
        rows.append({"branch_id": item["branch_id"], "name": name, "kind": kind,
                     "rating": item.get("rating")})
    return pd.DataFrame(rows).drop_duplicates("branch_id")


def main():
    start = time.time()
    reviews = pd.read_csv(config.DATASET_DIR / "reviews_clean.csv",
                          usecols=["branch_id", "text"], dtype={"branch_id": str})
    clf = load_classifier()
    print(f"отзывов: {len(reviews)}, модель: {clf.name}")

    pred = np.zeros((len(reviews), len(clf.labels)), dtype=np.int32)
    texts = reviews["text"].tolist()
    step = 5000
    for i in range(0, len(texts), step):
        proba = clf.predict_proba(texts[i:i + step])
        pred[i:i + step] = proba >= clf.thresholds
        print(f"  {min(i + step, len(texts))}/{len(texts)}")

    counts = pd.DataFrame(pred, columns=clf.labels)
    counts["branch_id"] = reviews["branch_id"].values
    venues = counts.groupby("branch_id").sum()
    venues.insert(0, "reviews", reviews.groupby("branch_id").size())

    venues = load_names().merge(venues, left_on="branch_id", right_index=True, how="inner")
    config.DATA_DIR.mkdir(exist_ok=True)
    venues.to_csv(config.VENUES_PATH, index=False)
    print(f"заведений: {len(venues)}, сохранено в {config.VENUES_PATH}, {time.time() - start:.0f} с")


if __name__ == "__main__":
    main()
