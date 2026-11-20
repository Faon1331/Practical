import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app

pytestmark = pytest.mark.skipif(
    not config.TFIDF_PATH.exists(), reason="сначала обучите модель: scripts/train_tfidf.py"
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
    assert all(s["predicted"] == (s["probability"] >= s["threshold"]) for s in data["scores"])


@pytest.mark.parametrize("text, label", [
    ("Отличное место для семьи, есть детская комната и детское меню, ребёнок в восторге", "Покормить детей"),
    ("Тихо, есть розетки и wi-fi, удобно поработать с ноутбуком", "Поработать"),
    ("Панорамные окна, с террасы шикарный вид на реку и город", "Красивый вид"),
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


needs_venues = pytest.mark.skipif(not config.VENUES_PATH.exists(),
                                  reason="сначала постройте профили: scripts/build_venues.py")


@needs_venues
def test_top_venues(client):
    resp = client.get("/api/venues", params={"label": "Свидание", "limit": 5, "min_reviews": 10})
    assert resp.status_code == 200
    items = resp.json()
    assert 0 < len(items) <= 5
    shares = [v["share"] for v in items]
    assert shares == sorted(shares, reverse=True)
    assert all(v["reviews"] >= 10 for v in items)


@needs_venues
def test_top_venues_unknown_label(client):
    assert client.get("/api/venues", params={"label": "Нет такой"}).status_code == 404


@needs_venues
def test_search_and_profile(client):
    found = client.get("/api/venues/search", params={"q": "Фиалка"}).json()
    assert found
    venue = client.get(f"/api/venues/{found[0]['branch_id']}").json()
    assert venue["name"] == found[0]["name"]
    assert all(0 < x["share"] <= 1 for x in venue["labels"])


@needs_venues
def test_unknown_venue(client):
    assert client.get("/api/venues/nope").status_code == 404
