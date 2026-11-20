import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BASE_DIR.parent

DATASET_DIR = Path(os.getenv("DATASET_DIR", PROJECT_DIR / "dataset"))
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
STATIC_DIR = BASE_DIR / "app" / "static"

# какую модель поднимать: tfidf (по умолчанию) или bert
MODEL_TYPE = os.getenv("MODEL_TYPE", "tfidf")
TFIDF_PATH = Path(os.getenv("TFIDF_PATH", MODELS_DIR / "tfidf.joblib"))
BERT_PATH = Path(os.getenv("BERT_PATH", MODELS_DIR / "bert"))
VENUES_PATH = Path(os.getenv("VENUES_PATH", DATA_DIR / "venues.csv"))

# названия колонок в датасете 2ГИС, порядок как в train.csv
LABEL_COLUMNS = [
    "Большой бар", "День рождения", "Добавить мистики", "Здоровое меню",
    "Красивый вид", "Кухни мира", "Необычные вкусы", "Поболтать",
    "Покормить детей", "Поработать", "Потусить", "Праздник",
    "Просто поесть", "Просторный зал", "Расслабиться", "Свидание",
    "Стильный интерьер", "Удобные сиденья", "Фоткаться",
]

# наружу отдаём литературное название
DISPLAY_NAMES = {"Фоткаться": "Фотографироваться"}

LABEL_GROUPS = {
    "Цель визита": ["Просто поесть", "Расслабиться", "Потусить", "Фоткаться",
                    "Покормить детей", "Поболтать", "Поработать"],
    "Обстановка": ["Удобные сиденья", "Стильный интерьер", "Красивый вид",
                   "Просторный зал", "Добавить мистики"],
    "Еда и напитки": ["Необычные вкусы", "Большой бар", "Здоровое меню", "Кухни мира"],
    "Событие": ["Праздник", "Свидание", "День рождения"],
}

GROUP_BY_COLUMN = {col: group for group, cols in LABEL_GROUPS.items() for col in cols}


def display_name(column: str) -> str:
    return DISPLAY_NAMES.get(column, column)


COLUMN_BY_NAME = {display_name(c): c for c in LABEL_COLUMNS}
