import pytest
from src.data.schemas import Review, ReviewBatch, PredictionResult, SummaryResult


class TestReview:
    def test_valid_review(self):
        r = Review(
            review_id="rev_001",
            product_id="prod_001",
            rating=4,
            title="Good product",
            text="This is a solid product with good build quality.",
        )
        assert r.rating == 4
        assert r.sentiment_label == "positive"

    def test_rating_boundaries(self):
        for rating in [1, 2, 3, 4, 5]:
            r = Review(
                review_id="test",
                product_id="test",
                rating=rating,
                title="Test",
                text="Some review text here",
            )
            assert r.rating == rating

    def test_invalid_rating_too_high(self):
        with pytest.raises(Exception):
            Review(
                review_id="test",
                product_id="test",
                rating=6,
                title="Test",
                text="Some review text",
            )

    def test_invalid_rating_too_low(self):
        with pytest.raises(Exception):
            Review(
                review_id="test",
                product_id="test",
                rating=0,
                title="Test",
                text="Some review text",
            )

    def test_empty_text_rejected(self):
        with pytest.raises(Exception):
            Review(
                review_id="test",
                product_id="test",
                rating=3,
                title="Test",
                text="  ",
            )

    def test_empty_title_rejected(self):
        with pytest.raises(Exception):
            Review(
                review_id="test",
                product_id="test",
                rating=3,
                title="   ",
                text="Actual review content here",
            )

    def test_sentiment_labels(self):
        mapping = {1: "very_negative", 2: "negative", 3: "neutral", 4: "positive", 5: "very_positive"}
        for rating, expected in mapping.items():
            r = Review(
                review_id="test", product_id="test",
                rating=rating, title="Test", text="Review text"
            )
            assert r.sentiment_label == expected

    def test_whitespace_stripped(self):
        r = Review(
            review_id="test",
            product_id="test",
            rating=3,
            title="  Padded title  ",
            text="  Padded text content  ",
        )
        assert r.title == "Padded title"
        assert r.text == "Padded text content"


class TestReviewBatch:
    def test_batch_creation(self, sample_reviews_raw):
        reviews = [Review(**r) for r in sample_reviews_raw]
        batch = ReviewBatch(reviews=reviews, source="csv")
        assert batch.size == 5
        assert batch.source == "csv"

    def test_rating_distribution(self, sample_reviews_raw):
        reviews = [Review(**r) for r in sample_reviews_raw]
        batch = ReviewBatch(reviews=reviews)
        dist = batch.rating_distribution()
        assert dist[5] == 1
        assert dist[1] == 1
        assert sum(dist.values()) == 5


class TestPredictionResult:
    def test_valid_prediction(self):
        p = PredictionResult(
            review_id="rev_001",
            true_label=4,
            predicted_label=4,
            confidence=0.92,
            model_name="bert",
        )
        assert p.confidence == 0.92

    def test_confidence_out_of_range(self):
        with pytest.raises(Exception):
            PredictionResult(
                review_id="test",
                true_label=3,
                predicted_label=3,
                confidence=1.5,
                model_name="test",
            )
