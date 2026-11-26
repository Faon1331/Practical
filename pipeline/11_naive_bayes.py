"""Этап 3. Наивный байесовский классификатор (MultinomialNB) на TF-IDF."""
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

from common import LABELS, load_features, load_split, save_prediction

train, _ = load_split()
y_train = train[LABELS].to_numpy()
f = load_features("tfidf_words")

model = OneVsRestClassifier(MultinomialNB(alpha=0.2))
model.fit(f["X_train"], y_train)
proba = model.predict_proba(f["X_test"])
save_prediction("naive_bayes", proba >= 0.5, proba)

print(f"меток поставлено на тесте: {int((proba >= 0.5).sum())}")
print("сохранено: data/predictions/naive_bayes.joblib")
