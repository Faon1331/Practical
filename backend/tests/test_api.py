import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason="сначала запустите pipeline (скрипты 01-12), нужен models/final_model.joblib",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_labels(client):
    labels = client.get("/api/labels").json()
    assert len(labels) == 19
    names = [x["label"] for x in labels]
    assert "Фотографироваться" in names
    assert "Фоткаться" not in names


def test_predict_returns_all_labels(client):
    resp = client.post("/api/predict", json={"text": "Вкусно поели, приятный персонал"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["scores"]) == 19
    probs = [s["probability"] for s in data["scores"]]
    assert probs == sorted(probs, reverse=True)
    assert all(0 <= p <= 1 for p in probs)
    assert set(data["labels"]) == {s["label"] for s in data["scores"] if s["predicted"]}


@pytest.mark.parametrize("text, label", [
    ("Отличное место для семьи, есть детская комната и детское меню, ребёнок в восторге", "Покормить детей"),
    ("Тихо, есть розетки и wi-fi, удобно поработать с ноутбуком", "Поработать"),
    ("Панорамные окна, с террасы шикарный вид на реку и город", "Красивый вид"),
    ("Всегда очень вкусный кофе и свежие десерты) место тихое и спокойное, "
     "для посиделок вдвоём идеально)", "Свидание"),
])
def test_predict_obvious_cases(client, text, label):
    data = client.post("/api/predict", json={"text": text}).json()
    assert label in data["labels"]


def test_predict_rejects_empty_text(client):
    assert client.post("/api/predict", json={"text": "   "}).status_code == 422
    assert client.post("/api/predict", json={}).status_code == 422


def test_predict_batch(client):
    resp = client.post("/api/predict/batch", json={"texts": ["вкусный кофе", "большой бар, коктейли"]})
    assert resp.status_code == 200
    assert len(resp.json()["items"]) == 2


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Анализ отзыва" in resp.text
