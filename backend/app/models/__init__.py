from app import config


def load_classifier(model_type: str = config.MODEL_TYPE):
    if model_type == "tfidf":
        from app.models.tfidf import TfidfClassifier
        return TfidfClassifier(config.TFIDF_PATH)
    if model_type == "bert":
        from app.models.bert import BertClassifier
        return BertClassifier(config.BERT_PATH)
    raise ValueError(f"Неизвестный тип модели: {model_type}")
