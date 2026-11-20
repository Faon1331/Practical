import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.models import load_classifier
from app.schemas import (BatchIn, BatchOut, LabelInfo, Prediction, PredictionOut,
                         ReviewIn, VenueOut, VenueShort)
from app.venues import VenueStore

log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = load_classifier()
    log.info("модель загружена: %s", app.state.classifier.name)

    app.state.venues = None
    if config.VENUES_PATH.exists():
        app.state.venues = VenueStore(config.VENUES_PATH)
        log.info("заведений загружено: %d", len(app.state.venues))
    else:
        log.warning("нет %s, запросы по заведениям будут недоступны", config.VENUES_PATH)
    yield


app = FastAPI(
    title="Характеристики заведений по отзывам 2ГИС",
    description="Многометочная классификация отзывов по 19 характеристикам "
                "и профили заведений на её основе.",
    version="1.0.0",
    lifespan=lifespan,
)


def to_predictions(classifier, proba: np.ndarray) -> list[Prediction]:
    result = []
    for row in proba:
        scores = []
        for col, p, thr in zip(classifier.labels, row, classifier.thresholds):
            scores.append({
                "label": config.display_name(col),
                "group": config.GROUP_BY_COLUMN[col],
                "probability": round(float(p), 4),
                "threshold": round(float(thr), 2),
                "predicted": bool(p >= thr),
            })
        scores.sort(key=lambda s: s["probability"], reverse=True)
        labels = [s["label"] for s in scores if s["predicted"]]
        result.append(Prediction(labels=labels, scores=scores))
    return result


def get_venues(request: Request) -> VenueStore:
    venues = request.app.state.venues
    if venues is None:
        raise HTTPException(503, "Профили заведений не построены: запустите scripts/build_venues.py")
    return venues


@app.get("/api/health")
def health(request: Request):
    return {
        "status": "ok",
        "model": request.app.state.classifier.name,
        "venues": len(request.app.state.venues) if request.app.state.venues else 0,
    }


@app.get("/api/labels", response_model=list[LabelInfo])
def labels():
    return [{"label": config.display_name(c), "group": config.GROUP_BY_COLUMN[c]}
            for c in config.LABEL_COLUMNS]


@app.post("/api/predict", response_model=PredictionOut)
def predict(review: ReviewIn, request: Request):
    clf = request.app.state.classifier
    pred = to_predictions(clf, clf.predict_proba([review.text]))[0]
    return PredictionOut(model=clf.name, **pred.model_dump())


@app.post("/api/predict/batch", response_model=BatchOut)
def predict_batch(batch: BatchIn, request: Request):
    clf = request.app.state.classifier
    return BatchOut(model=clf.name, items=to_predictions(clf, clf.predict_proba(batch.texts)))


@app.get("/api/venues", response_model=list[VenueShort])
def top_venues(
    request: Request,
    label: str = Query(..., description="Характеристика, например Свидание"),
    limit: int = Query(10, ge=1, le=100),
    min_reviews: int = Query(10, ge=1, description="Не учитывать заведения с малым числом отзывов"),
):
    column = config.COLUMN_BY_NAME.get(label)
    if column is None:
        raise HTTPException(404, f"Нет такой характеристики: {label}")
    return get_venues(request).top(column, limit=limit, min_reviews=min_reviews)


@app.get("/api/venues/search", response_model=list[VenueShort])
def search_venues(request: Request, q: str = Query(..., min_length=2),
                  limit: int = Query(10, ge=1, le=50)):
    return get_venues(request).search(q, limit=limit)


@app.get("/api/venues/{branch_id}", response_model=VenueOut)
def venue(branch_id: str, request: Request):
    item = get_venues(request).get(branch_id)
    if item is None:
        raise HTTPException(404, "Заведение не найдено")
    return item


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def index():
    return FileResponse(config.STATIC_DIR / "index.html")
