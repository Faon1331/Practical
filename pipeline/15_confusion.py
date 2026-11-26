"""Этап 4. Матрицы ошибок итоговой модели по каждой метке."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from common import FIGURES, LABELS, RESULTS, display, load_prediction, load_split

_, test = load_split()
y = test[LABELS].to_numpy()
pred = load_prediction("final")["pred"]

rows, matrices = [], []
for i, col in enumerate(LABELS):
    tn, fp, fn, tp = confusion_matrix(y[:, i], pred[:, i], labels=[0, 1]).ravel()
    matrices.append(np.array([[tn, fp], [fn, tp]]))
    rows.append({"метка": display(col), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                 "пропущено, %": round(fn / (tp + fn) * 100, 1),
                 "лишних, %": round(fp / (tp + fp) * 100, 1) if tp + fp else 0.0})

table = pd.DataFrame(rows)
table.to_csv(RESULTS / "confusion.csv", index=False)
print(table.to_string(index=False))

fig, axes = plt.subplots(4, 5, figsize=(15, 12))
for ax, col, m in zip(axes.ravel(), LABELS, matrices):
    ax.imshow(m, cmap="Blues")
    for (r, c), v in np.ndenumerate(m):
        ax.text(c, r, str(v), ha="center", va="center",
                color="white" if v > m.max() / 2 else "black", fontsize=9)
    ax.set_title(display(col), fontsize=10)
    ax.set_xticks([0, 1], ["нет", "да"])
    ax.set_yticks([0, 1], ["нет", "да"])
for ax in axes.ravel()[len(LABELS):]:
    ax.axis("off")
fig.suptitle("Матрицы ошибок итоговой модели, test")
fig.supxlabel("Ответ модели")
fig.supylabel("Разметка")
fig.tight_layout()
fig.savefig(FIGURES / "confusion_matrices.png", dpi=120, bbox_inches="tight")
plt.close(fig)

print("сохранено: results/confusion.csv, figures/confusion_matrices.png")
