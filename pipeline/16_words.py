"""Этап 4. На какие слова опирается модель.

1) Слова с наибольшим средним весом TF-IDF в отзывах с меткой (4 метки).
2) Облако слов для метки "Поработать": логистическая регрессия с весами
   классов на TF-IDF по словам, размер слова - его коэффициент в модели.
3) Таблица самых весомых признаков итоговой модели по всем меткам.
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud

from common import FIGURES, LABELS, MODELS, RESULTS, display, load_features, load_split

train, _ = load_split()
f = load_features("tfidf_words")
words = f["vectorizer"].get_feature_names_out()

# 1) средний TF-IDF слов в отзывах с меткой
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, col in zip(axes.ravel(), ["Просто поесть", "Свидание", "Поработать", "Покормить детей"]):
    mask = (train[col] == 1).to_numpy()
    mean = np.asarray(f["X_train"][mask].mean(axis=0)).ravel()
    top = np.argsort(mean)[::-1][:10]
    ax.barh(range(10)[::-1], mean[top])
    ax.set_yticks(range(10)[::-1], words[top])
    ax.set_title(f"{display(col)} ({mask.sum()} отзывов)")
    ax.set_xlabel("Средний TF-IDF")
fig.tight_layout()
fig.savefig(FIGURES / "top_words.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 2) облако слов модели для метки "Поработать"
col = "Поработать"
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(f["X_train"], train[col])
coef = model.coef_.ravel()
best = [j for j in np.argsort(coef)[::-1][:80] if coef[j] > 0]
cloud = WordCloud(width=1200, height=800, background_color="white", random_state=42,
                  font_path=font_manager.findfont("DejaVu Sans")) \
    .generate_from_frequencies({words[j]: float(coef[j]) for j in best})

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(cloud, interpolation="bilinear")
ax.set_title(f'Слова, на которые опирается модель: "{col}"')
ax.axis("off")
fig.savefig(FIGURES / "cloud_rabota.png", dpi=100, bbox_inches="tight")
plt.close(fig)
print(f"{col}: {', '.join(words[best[:20]])}")

# 3) самые весомые признаки итоговой модели (слова и биграммы)
final = joblib.load(MODELS / "final_model.joblib")
terms = final["vectorizer"].get_feature_names_out()
rows = []
for i, label in enumerate(LABELS):
    c = final["model"].estimators_[i].coef_.ravel()
    rows.append({"метка": display(label), "признаки": ", ".join(terms[np.argsort(c)[::-1][:15]])})
pd.DataFrame(rows).to_csv(RESULTS / "top_words.csv", index=False)

print("сохранено: results/top_words.csv, figures/top_words.png, figures/cloud_rabota.png")
