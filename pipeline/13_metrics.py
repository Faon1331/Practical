"""Этап 4. Точность (P), полнота (R) и F1 по классам и macro, ROC-анализ.

Все метрики считаются на тестовой выборке. macro - среднее по 19 меткам.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from common import COMPARISON, FIGURES, LABELS, RESULTS, display, fmt, load_prediction, load_split

_, test = load_split()
y = test[LABELS].to_numpy()

names = ["logreg_raw", "logreg_bigrams_balanced"] + [name for name, _, _ in COMPARISON]
rows, f1_by_label = [], {}
for name in names:
    pred = load_prediction(name)["pred"]
    p = precision_score(y, pred, average=None, zero_division=0)
    r = recall_score(y, pred, average=None, zero_division=0)
    f1 = f1_score(y, pred, average=None, zero_division=0)
    f1_by_label[name] = f1
    rows.append({"модель": name, "P": p.mean(), "R": r.mean(), "macro-F1": f1.mean(),
                 "F1 min": f1.min(), "F1 max": f1.max()})

metrics = pd.DataFrame(rows).round(4)
metrics.to_csv(RESULTS / "metrics.csv", index=False)
print(metrics.to_string(index=False))

pd.DataFrame(f1_by_label, index=[display(c) for c in LABELS]).round(3) \
    .to_csv(RESULTS / "f1_by_label.csv")

# таблица "Улучшение базового решения"
steps = [
    (1, "TF-IDF + логистическая регрессия, порог 0,5", "logreg_raw"),
    (2, "Лемматизация и удаление стоп-слов", "logreg"),
    (3, "Балансировка классов (веса)", "logreg_balanced"),
    (4, "Биграммы + свой порог для каждой метки", "final"),
]
steps = pd.DataFrame([{"шаг": n, "что изменили": what,
                       "macro-F1": fmt(f1_by_label[name].mean())} for n, what, name in steps])
steps.to_csv(RESULTS / "steps.csv", index=False)
print("\n" + steps.to_string(index=False))

# рисунок: F1 по меткам на тестовой выборке
order = np.argsort(f1_by_label["final"])
series = [("logreg_balanced", "слова + балансировка"), ("logreg_bigrams_balanced", "+ биграммы"),
          ("final", "+ биграммы + свой порог")]
fig, ax = plt.subplots(figsize=(10, 8))
y_pos = np.arange(len(LABELS))
for i, (name, title) in enumerate(series):
    ax.barh(y_pos + (1 - i) * 0.27, f1_by_label[name][order], height=0.27, label=title)
ax.set_yticks(y_pos, [display(LABELS[i]) for i in order])
ax.set_xlabel("F1 на test")
ax.set_title("F1 по меткам: биграммы и подбор порога")
ax.legend(loc="lower right")
fig.savefig(FIGURES / "f1_by_label.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ROC-анализ сбалансированной модели (логистическая регрессия, TF-IDF по словам)
proba = load_prediction("logreg_balanced")["proba"]
auc = np.array([roc_auc_score(y[:, i], proba[:, i]) for i in range(len(LABELS))])
pd.DataFrame({"метка": [display(c) for c in LABELS], "AUC": auc.round(3)}) \
    .sort_values("AUC", ascending=False).to_csv(RESULTS / "roc_auc.csv", index=False)
print(f"\nROC AUC: среднее {auc.mean():.2f}, от {auc.min():.2f} до {auc.max():.2f}")

fig, ax = plt.subplots(figsize=(7, 7))
for col in ["Стильный интерьер", "Просто поесть", "Свидание", "Добавить мистики"]:
    i = LABELS.index(col)
    fpr, tpr, _ = roc_curve(y[:, i], proba[:, i])
    ax.plot(fpr, tpr, label=f"{display(col)} (AUC = {auc[i]:.2f})")
ax.plot([0, 1], [0, 1], "k--", label="случайное угадывание (AUC = 0.5)")
ax.set_xlabel("FPR (доля ложных срабатываний)")
ax.set_ylabel("TPR (доля найденных, recall)")
ax.set_title("ROC-кривые, сбалансированная модель, test")
ax.legend(loc="lower right")
fig.savefig(FIGURES / "roc.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("сохранено: results/metrics.csv, f1_by_label.csv, steps.csv, roc_auc.csv, "
      "figures/f1_by_label.png, figures/roc.png")
