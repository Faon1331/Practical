import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "app" / "static"

# итоговую модель сохраняет pipeline/12_thresholds.py
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_DIR / "models" / "final_model.joblib"))

# метки и предобработка текста берутся из pipeline/common.py,
# чтобы сервис обрабатывал текст ровно так же, как при обучении
sys.path.insert(0, str(PROJECT_DIR / "pipeline"))
from common import GROUPS, LABELS, display, preprocess_text  # noqa: E402

LABEL_COLUMNS = LABELS
GROUP_BY_COLUMN = {col: group for group, cols in GROUPS.items() for col in cols}
display_name = display
