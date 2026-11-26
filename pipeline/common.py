"""Общие настройки и функции для скриптов 01-16."""
import re
from functools import lru_cache
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"          # исходный JSON от 2ГИС
DATA = ROOT / "data"                # промежуточные таблицы
FEATURES = DATA / "features"        # матрицы признаков
PREDICTIONS = DATA / "predictions"  # ответы моделей на тестовой выборке
MODELS = ROOT / "models"            # обученные модели
RESULTS = ROOT / "results"          # таблицы с метриками
FIGURES = RESULTS / "figures"       # графики

for folder in (DATA, FEATURES, PREDICTIONS, MODELS, RESULTS, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)

LABELS = [
    "Большой бар", "День рождения", "Добавить мистики", "Здоровое меню",
    "Красивый вид", "Кухни мира", "Необычные вкусы", "Поболтать",
    "Покормить детей", "Поработать", "Потусить", "Праздник",
    "Просто поесть", "Просторный зал", "Расслабиться", "Свидание",
    "Стильный интерьер", "Удобные сиденья", "Фоткаться",
]

GROUPS = {
    "Цель визита": ["Просто поесть", "Расслабиться", "Потусить", "Фоткаться",
                    "Покормить детей", "Поболтать", "Поработать"],
    "Обстановка": ["Удобные сиденья", "Стильный интерьер", "Красивый вид",
                   "Просторный зал", "Добавить мистики"],
    "Еда и напитки": ["Необычные вкусы", "Большой бар", "Здоровое меню", "Кухни мира"],
    "Событие": ["Праздник", "Свидание", "День рождения"],
}

# в таблицах и на графиках пишем полное название
DISPLAY = {"Фоткаться": "Фотографироваться"}


def display(label):
    return DISPLAY.get(label, label)


# модели, которые сравниваются на слайде "Сравнение методов"
COMPARISON = [
    ("naive_bayes", "Наивный Байес", "TF-IDF"),
    ("logreg_w2v_balanced", "Лог. регрессия, balanced", "Word2Vec"),
    ("logreg", "Лог. регрессия", "TF-IDF"),
    ("svm_balanced", "Линейный SVM, balanced", "TF-IDF"),
    ("logreg_balanced", "Лог. регрессия, balanced", "TF-IDF"),
    ("svm", "Линейный SVM", "TF-IDF"),
    ("final", "Лог. регрессия, balanced + пороги", "TF-IDF + биграммы"),
]


# ---------- предобработка текста ----------

# местоимения, предлоги, союзы, частицы, междометия
STOP_POS = {"NPRO", "PREP", "CONJ", "PRCL", "INTJ"}
# "не" и "нет" оставляем: "не уютно" и "уютно" - разные вещи
NEGATIONS = {"не", "нет"}
# в отзывах есть казахский текст, такие слова pymorphy3 портит, их не трогаем
KAZAKH = set("әғқңөұүһі")
WORD = re.compile(r"[^\W\d_]+")

_morph = None


def morph():
    global _morph
    if _morph is None:
        import pymorphy3
        _morph = pymorphy3.MorphAnalyzer()
    return _morph


@lru_cache(maxsize=None)
def lemma(word):
    if word in NEGATIONS:
        return word
    if KAZAKH & set(word):
        return word
    parsed = morph().parse(word)[0]
    if parsed.tag.POS in STOP_POS:
        return None
    return parsed.normal_form


def preprocess_text(text):
    """Нижний регистр, только слова из букв, без стоп-слов, в начальной форме."""
    if not isinstance(text, str):
        return ""
    lemmas = (lemma(w) for w in WORD.findall(text.lower()))
    return " ".join(x for x in lemmas if x and len(x) > 1)


# ---------- чтение и запись ----------

def load_split():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    for df in (train, test):
        df["text_clean"] = df["text_clean"].fillna("")
    return train, test


def load_features(name):
    return joblib.load(FEATURES / f"{name}.joblib")


def save_prediction(name, pred, proba=None):
    joblib.dump({"pred": np.asarray(pred, dtype=np.int8), "proba": proba},
                PREDICTIONS / f"{name}.joblib")


def load_prediction(name):
    return joblib.load(PREDICTIONS / f"{name}.joblib")


def fmt(x):
    """0.4371 -> '0,44' (как в презентации)."""
    if abs(x) < 0.005:
        x = 0.0
    return f"{x:.2f}".replace(".", ",")
