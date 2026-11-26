"""Этап 1. Статистика по данным и графики.

Сколько отзывов у каждой метки, насколько сильный дисбаланс классов,
какой длины отзывы.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from common import DATA, FIGURES, GROUPS, LABELS, RESULTS, display

df = pd.read_csv(DATA / "reviews.csv")

counts = df[LABELS].sum()
stats = pd.DataFrame({
    "метка": [display(c) for c in LABELS],
    "группа": [next(g for g, cols in GROUPS.items() if c in cols) for c in LABELS],
    "отзывов": counts.values,
    "доля, %": (counts.values / len(df) * 100).round(1),
}).sort_values("отзывов", ascending=False)
stats.to_csv(RESULTS / "label_stats.csv", index=False)

print(f"отзывов: {len(df)}, заведений: {df['branch_id'].nunique()}")
print(f"меток у отзыва в среднем: {df[LABELS].sum(axis=1).mean():.2f}")
print(stats.to_string(index=False))

# рисунок 1: количество отзывов по меткам, цвет - группа
colors = {"Цель визита": "tab:blue", "Обстановка": "tab:green",
          "Еда и напитки": "tab:orange", "Событие": "tab:red"}
rows = []
for group, cols in GROUPS.items():
    for col in counts[cols].sort_values(ascending=False).index:
        rows.append((display(col), counts[col], colors[group]))

fig, ax = plt.subplots(figsize=(10, 7))
y = np.arange(len(rows))[::-1]
ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows])
ax.set_yticks(y, [r[0] for r in rows])
ax.set_xlabel("Количество отзывов")
ax.set_title("Сколько отзывов у каждой метки (по группам)")
ax.legend(handles=[Patch(color=c, label=g) for g, c in colors.items()], loc="lower right")
fig.savefig(FIGURES / "labels_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

lengths = df["text"].str.len()
print(f"\nдлина отзыва: среднее {lengths.mean():.0f}, медиана {lengths.median():.0f} символов")
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(lengths.clip(upper=1000), bins=50, color="tab:blue")
ax.set_xlabel("Длина отзыва, символов (больше 1000 - в последнем столбце)")
ax.set_ylabel("Количество отзывов")
ax.set_title("Длина отзывов")
fig.savefig(FIGURES / "text_length.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("сохранено: results/label_stats.csv, results/figures/labels_distribution.png, text_length.png")
