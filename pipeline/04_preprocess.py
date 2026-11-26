"""Этап 1. Удаление стоп-слов и лемматизация (pymorphy3).

Правила - в common.preprocess_text. Результат пишется в колонку text_clean.
"""
import time

import pandas as pd

from common import DATA, preprocess_text

start = time.time()
df = pd.read_csv(DATA / "reviews_clean.csv")
df["text_clean"] = df["text"].map(preprocess_text)
df.to_csv(DATA / "reviews_lemm.csv", index=False)

example = df[df["text"].str.startswith("Всегда очень вкусный кофе и свежие десерты")]
if len(example):
    print("до:   ", example["text"].iloc[0])
    print("после:", example["text_clean"].iloc[0])
print(f"пустых после обработки: {(df['text_clean'] == '').sum()}")
print(f"обработано {len(df)} отзывов за {time.time() - start:.0f} с, сохранено: data/reviews_lemm.csv")
