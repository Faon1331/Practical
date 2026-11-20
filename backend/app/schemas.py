from pydantic import BaseModel, Field, field_validator


class ReviewIn(BaseModel):
    text: str = Field(..., max_length=5000, examples=[
        "Уютное кафе, тихо, можно спокойно поработать с ноутбуком, кофе вкусный"
    ])

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("текст отзыва пустой")
        return value


class BatchIn(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class LabelScore(BaseModel):
    label: str
    group: str
    probability: float
    threshold: float
    predicted: bool


class Prediction(BaseModel):
    labels: list[str]
    scores: list[LabelScore]


class PredictionOut(Prediction):
    model: str


class BatchOut(BaseModel):
    model: str
    items: list[Prediction]


class LabelInfo(BaseModel):
    label: str
    group: str


class VenueShort(BaseModel):
    branch_id: str
    name: str
    kind: str | None = None
    rating: float | None = None
    reviews: int
    share: float | None = None


class VenueLabel(BaseModel):
    label: str
    group: str
    count: int
    share: float


class VenueOut(VenueShort):
    labels: list[VenueLabel]
