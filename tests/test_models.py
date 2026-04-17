import pytest
import numpy as np

from src.models.classical import ClassicalModels
from src.models.summarizer import ExtractiveSummarizer


@pytest.fixture
def small_dataset():
    """Tiny synthetic dataset for fast unit tests."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 50)
    y = rng.randint(0, 5, size=100)
    return X, y


@pytest.fixture
def classical_models(sample_config):
    return ClassicalModels(sample_config)


class TestClassicalModels:
    def test_train_runs(self, classical_models, small_dataset):
        X, y = small_dataset
        results = classical_models.train(X, y)
        assert len(results) > 0
        for name, acc in results.items():
            assert 0.0 <= acc <= 1.0

    def test_evaluate_runs(self, classical_models, small_dataset):
        X, y = small_dataset
        classical_models.train(X, y)
        results = classical_models.evaluate(X, y)
        assert "logistic_regression" in results
        for name, res in results.items():
            assert "accuracy" in res
            assert "predictions" in res
            assert len(res["predictions"]) == len(y)

    def test_predict_before_train_raises(self, classical_models, small_dataset):
        X, _ = small_dataset
        with pytest.raises(RuntimeError):
            classical_models.predict("logistic_regression", X)

    def test_predict_unknown_model_raises(self, classical_models, small_dataset):
        X, _ = small_dataset
        with pytest.raises(ValueError):
            classical_models.predict("nonexistent_model", X)

    def test_predict_proba(self, classical_models, small_dataset):
        X, y = small_dataset
        classical_models.train(X, y)
        proba = classical_models.predict_proba("logistic_regression", X)
        assert proba is not None
        assert proba.shape[0] == len(X)


class TestExtractiveSummarizer:
    @pytest.fixture
    def summarizer(self, sample_config):
        return ExtractiveSummarizer(sample_config)

    def test_summarize_short_text(self, summarizer):
        text = "This product is great. I love it."
        result = summarizer.summarize(text, num_sentences=3)
        # text has <= 3 sentences so should return as-is
        assert len(result) > 0

    def test_summarize_long_text(self, summarizer):
        text = (
            "The sound quality on these headphones is truly exceptional. "
            "Bass response is deep and controlled without being overwhelming. "
            "Midrange clarity is impressive for a consumer product. "
            "High frequencies are crisp but never harsh or fatiguing. "
            "The active noise cancellation blocks out most ambient noise effectively. "
            "Battery life easily lasts twelve hours of continuous playback. "
            "The carrying case is compact and well designed. "
            "Bluetooth connectivity is stable with no dropouts during my testing. "
            "Overall this is a premium product that justifies the price tag."
        )
        result = summarizer.summarize(text, num_sentences=3)
        assert len(result) < len(text)

    def test_summarize_batch(self, summarizer):
        texts = [
            "Great product overall. Sound quality is amazing. Battery lasts forever. "
            "Very comfortable for long listening. Build quality feels premium.",
            "Terrible experience with this product. Broke after two days of normal use. "
            "Customer support was unhelpful. Would not buy again. Waste of money.",
        ]
        results = summarizer.summarize_batch(texts, num_sentences=2)
        assert len(results) == 2
        for r in results:
            assert "summary" in r
            assert "compression_ratio" in r
            assert 0 < r["compression_ratio"] <= 1.0

    def test_empty_text(self, summarizer):
        result = summarizer.summarize("")
        assert result == ""
