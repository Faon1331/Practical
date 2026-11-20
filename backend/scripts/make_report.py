"""Графики и цифры для отчёта и презентации.

Нужны результаты train_tfidf.py (reports/metrics_tfidf.json и
reports/test_proba_tfidf.npy). Цифры BERT и Word2Vec взяты из вывода
ноутбука 2Gis_V2.ipynb (ячейка "Сравнение всех моделей"), цифры шагов
1-3 - из экспериментов дата-инженера (черновик презентации).

Запуск из папки backend:
    python scripts/make_report.py
"""
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_curve

from app import config

LABELS = config.LABEL_COLUMNS
FIG_DIR = config.REPORTS_DIR / "figures"
plt.rcParams["font.family"] = "Arial"

# 2Gis_V2.ipynb, F1 по меткам на test (порядок как в LABELS)
F1_BERT = [0.6634, 0.3292, 0.0000, 0.4702, 0.7914, 0.4308, 0.3406, 0.5483, 0.6392, 0.5696,
           0.5299, 0.6460, 0.7073, 0.7193, 0.5419, 0.4660, 0.7953, 0.6903, 0.5551]


def names(cols):
    return [config.display_name(c) for c in cols]


def save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ", name)


def plot_distribution(reviews: pd.DataFrame):
    colors = {"Цель визита": "tab:blue", "Обстановка": "tab:green",
              "Еда и напитки": "tab:orange", "Событие": "tab:red"}
    rows = []
    for group, cols in config.LABEL_GROUPS.items():
        counts = reviews[cols].sum().sort_values(ascending=False)
        rows += [(config.display_name(c), int(n), group) for c, n in counts.items()]

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[colors[r[2]] for r in rows])
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Количество отзывов")
    handles = [Patch(color=color, label=group) for group, color in colors.items()]
    ax.legend(handles=handles, loc="lower right")
    save(fig, "labels_distribution.png")


def plot_steps(summary: dict):
    steps = [
        ("1. Исходный текст, без обработки", 0.36),
        ("2. + лемматизация", 0.39),
        ("3. + балансировка классов", 0.44),
        ("4. + биграммы", summary["f1_macro_05"]),
        ("5. + свой порог для каждой метки", summary["f1_macro"]),
        ("ruBERT (дообучение)", 0.549),
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(steps))[::-1]
    colors = ["tab:blue"] * 5 + ["tab:orange"]
    ax.barh(y, [s[1] for s in steps], color=colors)
    ax.set_yticks(y, [s[0] for s in steps])
    for yi, (_, v) in zip(y, steps):
        ax.text(v + 0.005, yi, f"{v:.3f}", va="center")
    ax.set_xlim(0, 0.65)
    ax.set_xlabel("macro-F1 на test")
    save(fig, "f1_steps.png")


def plot_roc(test: pd.DataFrame, proba: np.ndarray):
    chosen = ["Красивый вид", "Стильный интерьер", "Поработать", "Свидание",
              "Необычные вкусы", "Просто поесть"]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for col in chosen:
        i = LABELS.index(col)
        fpr, tpr, _ = roc_curve(test[col], proba[:, i])
        auc = np.trapezoid(tpr, fpr)
        ax.plot(fpr, tpr, label=f"{config.display_name(col)} (AUC = {auc:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("Доля ложноположительных (FPR)")
    ax.set_ylabel("Доля истинноположительных (TPR)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    save(fig, "roc_tfidf.png")


def plot_f1_compare(per_label: list[dict]):
    f1_tfidf = [r["f1"] for r in per_label]
    order = np.argsort(F1_BERT)
    fig, ax = plt.subplots(figsize=(9, 8))
    y = np.arange(len(LABELS))
    ax.barh(y + 0.2, np.array(f1_tfidf)[order], height=0.4, label="TF-IDF + лог. регрессия")
    ax.barh(y - 0.2, np.array(F1_BERT)[order], height=0.4, label="ruBERT")
    ax.set_yticks(y, np.array(names(LABELS))[order])
    ax.set_xlabel("F1 на test")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    save(fig, "f1_tfidf_vs_bert.png")


def label_agreement(reviews: pd.DataFrame) -> dict:
    """Согласованность разметки по одинаковым текстам у разных заведений.

    Один и тот же отзыв (например, у сетевых кафе) встречается в выгрузке
    несколько раз и размечен каждый раз отдельно. Сравниваем эти разметки
    попарно - это оценка того, насколько стабильно размечаются отзывы.
    """
    dup = reviews[reviews.duplicated("text", keep=False)]
    a, b = [], []
    for _, group in dup.groupby("text"):
        values = group[LABELS].to_numpy()
        for i, j in combinations(range(len(values)), 2):
            a.append(values[i])
            b.append(values[j])
    a, b = np.array(a), np.array(b)

    per_label = {}
    for k, col in enumerate(LABELS):
        if a[:, k].sum() + b[:, k].sum() >= 10:
            per_label[config.display_name(col)] = round(cohen_kappa_score(a[:, k], b[:, k]), 3)

    texts = dup.groupby("text")[LABELS].nunique()
    return {
        "texts": int(dup["text"].nunique()),
        "rows": int(len(dup)),
        "pairs": int(len(a)),
        "texts_same_labels": int((texts.max(axis=1) == 1).sum()),
        "kappa_pooled": round(cohen_kappa_score(a.ravel(), b.ravel()), 3),
        "kappa_mean_by_label": round(float(np.mean(list(per_label.values()))), 3),
        "kappa_by_label": per_label,
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((config.REPORTS_DIR / "metrics_tfidf.json").read_text(encoding="utf-8"))
    proba = np.load(config.REPORTS_DIR / "test_proba_tfidf.npy")
    test = pd.read_csv(config.DATASET_DIR / "test.csv", usecols=LABELS)
    reviews = pd.read_csv(config.DATASET_DIR / "reviews.csv", usecols=["text", *LABELS])

    print("графики:")
    plot_distribution(reviews)
    plot_steps(metrics["summary"])
    plot_roc(test, proba)
    plot_f1_compare(metrics["per_label"])

    agreement = label_agreement(reviews)
    with open(config.REPORTS_DIR / "agreement.json", "w", encoding="utf-8") as f:
        json.dump(agreement, f, ensure_ascii=False, indent=2)
    print("\nсогласованность разметки:")
    print(json.dumps(agreement, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
