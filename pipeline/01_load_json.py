"""Этап 1. Загрузка JSON от 2ГИС и таблица меток.

В JSON отзывы сгруппированы по заведениям и по характеристикам. Переводим
в таблицу: один отзыв - одна строка, 19 колонок меток со значениями 0 и 1.
"""
import json

import pandas as pd

from common import DATA, DATASET, LABELS

with open(DATASET / "task_2_vibes.json", encoding="utf-8") as f:
    data = json.load(f)

reviews = {}
for item in data:
    for label, texts in item["nlp"]["aspect"].items():
        for text in texts:
            key = (item["branch_id"], text)
            if key not in reviews:
                reviews[key] = {"branch_id": item["branch_id"], "text": text,
                                "rating": item.get("rating")}
            reviews[key][label] = 1

df = pd.DataFrame(list(reviews.values()))
df = df.reindex(columns=["branch_id", "text", "rating", *LABELS])
df[LABELS] = df[LABELS].fillna(0).astype(int)
df.to_csv(DATA / "reviews.csv", index=False)

print(f"заведений в JSON: {len(data)}")
print(f"отзывов: {len(df)}, заведений с отзывами: {df['branch_id'].nunique()}")
print(f"меток: {len(LABELS)}")
print("сохранено: data/reviews.csv")
