import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.model import Classifier
from app.schemas import BatchIn, BatchOut, LabelInfo, Prediction, ReviewIn

log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = Classifier(config.MODEL_PATH)
    log.info("модель загружена: %s", config.MODEL_PATH.name)
    yield


app = FastAPI(
    title="Характеристики заведений по отзывам 2ГИС",
    description="Многометочная классификация отзывов по 19 характеристикам: "
                "TF-IDF (слова и биграммы) + логистическая регрессия с порогами по меткам.",
    version="1.2.0",
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


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/labels", response_model=list[LabelInfo])
def labels():
    return [{"label": config.display_name(c), "group": config.GROUP_BY_COLUMN[c]}
            for c in config.LABEL_COLUMNS]


@app.post("/api/predict", response_model=Prediction)
def predict(review: ReviewIn, request: Request):
    clf = request.app.state.classifier
    return to_predictions(clf, clf.predict_proba([review.text]))[0]


@app.post("/api/predict/batch", response_model=BatchOut)
def predict_batch(batch: BatchIn, request: Request):
    clf = request.app.state.classifier
    return BatchOut(items=to_predictions(clf, clf.predict_proba(batch.texts)))


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def index():
    return FileResponse(config.STATIC_DIR / "index.html")
