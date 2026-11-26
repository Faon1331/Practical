"""Этап 1. Разбиение на обучающую и тестовую выборки 80/20."""
import pandas as pd

from common import DATA, LABELS

df = pd.read_csv(DATA / "reviews_lemm.csv")

train = df.sample(frac=0.8, random_state=42)
test = df.drop(train.index)

train.to_csv(DATA / "train.csv", index=False)
test.to_csv(DATA / "test.csv", index=False)

print(f"обучающая выборка: {len(train)}, тестовая: {len(test)}")
share = pd.DataFrame({"train, %": train[LABELS].mean() * 100, "test, %": test[LABELS].mean() * 100})
print(share.round(1).to_string())
print("сохранено: data/train.csv, data/test.csv")
