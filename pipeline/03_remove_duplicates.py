"""Этап 1. Удаление повторов.

Одинаковые тексты встречаются у разных заведений (например, у сетевых кафе).
Оставляем первую копию. Заодно считаем, сколько копий размечены по-разному.
"""
import json

import pandas as pd

from common import DATA, LABELS, RESULTS

df = pd.read_csv(DATA / "reviews.csv")

dup = df[df.duplicated("text", keep=False)]
conflicts = int((dup.groupby("text")[LABELS].nunique() > 1).any(axis=1).sum())

clean = df.drop_duplicates("text", keep="first").reset_index(drop=True)
clean.to_csv(DATA / "reviews_clean.csv", index=False)

info = {
    "отзывов до очистки": len(df),
    "отзывов после очистки": len(clean),
    "исключено повторов": len(df) - len(clean),
    "повторяющихся текстов": int(dup["text"].nunique()),
    "из них с разной разметкой": conflicts,
}
with open(RESULTS / "duplicates.json", "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)

for k, v in info.items():
    print(f"{k}: {v}")
print("сохранено: data/reviews_clean.csv")
