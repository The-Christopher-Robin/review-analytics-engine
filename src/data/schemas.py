from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class Review(BaseModel):
    review_id: str
    product_id: str
    rating: int = Field(ge=1, le=5)
    title: str
    text: str
    category: Optional[str] = None
    timestamp: Optional[datetime] = None
    verified_purchase: bool = False

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < 3:
            raise ValueError("Review text must be at least 3 characters")
        return stripped

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Title cannot be empty")
        return v.strip()

    @property
    def sentiment_label(self) -> str:
        labels = {1: "very_negative", 2: "negative", 3: "neutral", 4: "positive", 5: "very_positive"}
        return labels[self.rating]


class ReviewBatch(BaseModel):
    reviews: list[Review]
    source: str = "unknown"
    loaded_at: datetime = Field(default_factory=datetime.now)

    @property
    def size(self) -> int:
        return len(self.reviews)

    def rating_distribution(self) -> dict[int, int]:
        dist: dict[int, int] = {}
        for r in self.reviews:
            dist[r.rating] = dist.get(r.rating, 0) + 1
        return dict(sorted(dist.items()))


class PredictionResult(BaseModel):
    review_id: str
    true_label: int
    predicted_label: int
    confidence: float = Field(ge=0.0, le=1.0)
    model_name: str


class SummaryResult(BaseModel):
    review_id: str
    original_text: str
    summary: str
    num_sentences_selected: int
    compression_ratio: float
