# Backend: характеристики заведений по отзывам 2ГИС

Веб-сервис на FastAPI. Он определяет по тексту отзыва, какие из 19 характеристик
подходят заведению ("Свидание", "Поработать", "Красивый вид" и т.д.), и строит
профили заведений и подборки по этим характеристикам.

Модель по умолчанию - TF-IDF + логистическая регрессия с отдельным порогом
для каждой метки. Можно подключить дообученный ruBERT из ноутбука `2Gis_V2.ipynb`.

## Структура

```
backend/
  app/
    main.py           - API (маршруты)
    config.py         - пути, список меток и их группы
    preprocessing.py  - предобработка текста (как в ноутбуке)
    models/           - классификаторы: tfidf.py, bert.py
    venues.py         - профили заведений, подборки, поиск
    schemas.py        - форматы запросов и ответов
    static/           - демо-страница
  scripts/
    train_tfidf.py    - обучение модели и подбор порогов
    build_venues.py   - профили заведений по всем отзывам
    make_report.py    - графики и таблицы для презентации
  tests/              - тесты API
  models/, data/, reports/ - создаются скриптами
```

## Запуск

Нужен Python 3.11. Данные лежат в папке `../dataset`. Другой путь можно
задать переменной `DATASET_DIR`.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python scripts/train_tfidf.py      # ~1 мин, создаёт models/tfidf.joblib
python scripts/build_venues.py     # ~1 мин, создаёт data/venues.csv
uvicorn app.main:app --port 8000
```

Демо-страница: http://localhost:8000, описание API: http://localhost:8000/docs

Тесты: `pytest`

## API

| Метод | Адрес | Что делает |
|---|---|---|
| GET | `/api/health` | состояние сервиса и какая модель загружена |
| GET | `/api/labels` | список характеристик и групп |
| POST | `/api/predict` | `{"text": "..."}` -> характеристики и вероятности |
| POST | `/api/predict/batch` | `{"texts": [...]}`, до 100 отзывов за раз |
| GET | `/api/venues?label=Свидание&limit=10&min_reviews=10` | топ заведений по характеристике |
| GET | `/api/venues/search?q=Фиалка` | поиск заведения по названию |
| GET | `/api/venues/{branch_id}` | профиль заведения: доля отзывов с каждой характеристикой |

Пример ответа `/api/predict`:

```json
{
  "model": "tfidf",
  "labels": ["Свидание", "Расслабиться"],
  "scores": [
    {"label": "Свидание", "group": "Событие", "probability": 0.959, "threshold": 0.78, "predicted": true},
    ...
  ]
}
```

Профили заведений строятся по предсказаниям модели, а не по исходной разметке:
так сервис работал бы с новыми отзывами, которые никто не размечал. В подборках
не учитываются заведения, у которых меньше `min_reviews` отзывов, иначе наверх
попадают места с 1-2 отзывами.

## Подключение BERT

1. В Colab после обучения сохранить модель:
   ```python
   trainer.save_model('bert_final_model')
   tokenizer.save_pretrained('bert_final_model')
   ```
2. Положить папку в `backend/models/bert`.
3. Установить `pip install torch transformers`.
4. Запустить с переменной `MODEL_TYPE=bert`.

Если в папке модели лежит `thresholds.json` (19 чисел), используются эти пороги,
иначе 0.5.

## Качество модели TF-IDF (test, 25 057 отзывов)

| Метрика | Значение |
|---|---|
| F1-macro, порог 0.5 | 0.487 |
| F1-macro, свои пороги | 0.528 |
| F1-micro | 0.615 |
| PR-AUC (среднее) | 0.514 |
| ROC-AUC (среднее) | 0.902 |

Подробно по меткам: `reports/metrics_tfidf.json`.
