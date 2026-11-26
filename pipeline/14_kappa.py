"""Этап 4. Каппа Коэна: согласие ответов модели с разметкой сверх случайного.

kappa = 0 - как наугад, kappa = 1 - полное совпадение. Считается по каждой
метке, macro-kappa - среднее по 19 меткам. Здесь же собирается итоговая
таблица сравнения методов.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from common import COMPARISON, FIGURES, LABELS, RESULTS, display, fmt, load_prediction, load_split

_, test = load_split()
y = test[LABELS].to_numpy()


def kappa(true, pred):
    value = cohen_kappa_score(true, pred)
    return 0.0 if np.isnan(value) else value


names = ["logreg_raw"] + [name for name, _, _ in COMPARISON]
kappas = {name: np.array([kappa(y[:, i], load_prediction(name)["pred"][:, i])
                          for i in range(len(LABELS))]) for name in names}
pd.DataFrame(kappas, index=[display(c) for c in LABELS]).round(3) \
    .to_csv(RESULTS / "kappa_by_label.csv")

# таблица "Сравнение методов" (метрики точности берём из 13_metrics.py)
metrics = pd.read_csv(RESULTS / "metrics.csv").set_index("модель")
rows = []
for name, title, features in COMPARISON:
    m, k = metrics.loc[name], kappas[name]
    rows.append({
        "Модель": title, "Признаки": features, "P": fmt(m["P"]), "R": fmt(m["R"]),
        "macro-F1": fmt(m["macro-F1"]), "F1 по классам": f"{fmt(m['F1 min'])}-{fmt(m['F1 max'])}",
        "macro-K": fmt(k.mean()), "K по классам": f"{fmt(k.min())}-{fmt(k.max())}",
    })
comparison = pd.DataFrame(rows)
comparison.to_csv(RESULTS / "comparison.csv", index=False)
print(comparison.to_string(index=False))

# итоговая модель: F1 и каппа по меткам
f1 = pd.read_csv(RESULTS / "f1_by_label.csv", index_col=0)["final"]
final = pd.DataFrame({"F1": f1.values, "каппа": kappas["final"].round(3)},
                     index=[display(c) for c in LABELS]).sort_values("каппа", ascending=False)
final.to_csv(RESULTS / "final_by_label.csv")
print(f"\nитоговая модель: macro-F1 = {fmt(f1.mean())}, macro-K = {fmt(kappas['final'].mean())}")

fig, ax = plt.subplots(figsize=(10, 8))
y_pos = np.arange(len(final))[::-1]
ax.barh(y_pos + 0.2, final["F1"], height=0.4, label="F1")
ax.barh(y_pos - 0.2, final["каппа"], height=0.4, label="Каппа Коэна")
ax.set_yticks(y_pos, final.index)
ax.set_xlabel("Значение на test")
ax.set_title("F1 и каппа Коэна по меткам: итоговая модель")
ax.legend(loc="lower right")
fig.savefig(FIGURES / "kappa_by_label.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("сохранено: results/kappa_by_label.csv, comparison.csv, final_by_label.csv, "
      "figures/kappa_by_label.png")
