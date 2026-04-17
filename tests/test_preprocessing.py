import pytest
from src.data.schemas import Review
from src.data.preprocessing import TextPreprocessor


@pytest.fixture
def preprocessor(sample_config):
    return TextPreprocessor(sample_config)


@pytest.fixture
def reviews_from_raw(sample_reviews_raw):
    return [Review(**r) for r in sample_reviews_raw]


class TestTextCleaning:
    def test_html_removal(self, preprocessor):
        text = "This is <b>bold</b> and <a href='test'>linked</a> text"
        cleaned = preprocessor.clean_text(text)
        assert "<b>" not in cleaned
        assert "<a" not in cleaned
        assert "bold" in cleaned

    def test_url_removal(self, preprocessor):
        text = "Check https://example.com for more info about this product"
        cleaned = preprocessor.clean_text(text)
        assert "https" not in cleaned
        assert "example.com" not in cleaned

    def test_lowercasing(self, preprocessor):
        text = "GREAT Product With AMAZING Sound"
        cleaned = preprocessor.clean_text(text)
        assert cleaned == "great product with amazing sound"

    def test_whitespace_normalization(self, preprocessor):
        text = "too   many    spaces   here"
        cleaned = preprocessor.clean_text(text)
        assert "  " not in cleaned

    def test_empty_string(self, preprocessor):
        assert preprocessor.clean_text("") == ""


class TestDataframePreparation:
    def test_prepare_returns_expected_columns(self, preprocessor, reviews_from_raw):
        df = preprocessor.prepare_dataframe(reviews_from_raw)
        expected_cols = {"review_id", "text", "title", "rating", "label", "category"}
        assert expected_cols.issubset(set(df.columns))

    def test_labels_are_zero_indexed(self, preprocessor, reviews_from_raw):
        df = preprocessor.prepare_dataframe(reviews_from_raw)
        # rating 1 -> label 0, rating 5 -> label 4
        assert df["label"].min() >= 0
        assert df["label"].max() <= 4

    def test_short_reviews_filtered(self, sample_config):
        pp = TextPreprocessor(sample_config)
        reviews = [
            Review(review_id="r1", product_id="p1", rating=3, title="Short", text="bad"),
            Review(review_id="r2", product_id="p2", rating=4, title="Long", text="This is a much longer review with plenty of words"),
        ]
        df = pp.prepare_dataframe(reviews)
        # "bad" is only 3 chars after cleaning, below min_review_length
        assert len(df) == 1
        assert df.iloc[0]["review_id"] == "r2"


class TestSplitting:
    def test_split_proportions(self, preprocessor, reviews_from_raw):
        df = preprocessor.prepare_dataframe(reviews_from_raw)
        train_df, test_df = preprocessor.split_data(df, test_size=0.4, seed=42)
        assert len(train_df) + len(test_df) == len(df)
        assert len(test_df) >= 1


class TestTfidf:
    def test_fit_transform(self, preprocessor, reviews_from_raw):
        df = preprocessor.prepare_dataframe(reviews_from_raw)
        features = preprocessor.fit_tfidf(df["text"])
        assert features.shape[0] == len(df)
        assert features.shape[1] > 0

    def test_transform_after_fit(self, preprocessor, reviews_from_raw):
        df = preprocessor.prepare_dataframe(reviews_from_raw)
        preprocessor.fit_tfidf(df["text"])
        transformed = preprocessor.transform_tfidf(df["text"])
        assert transformed.shape[0] == len(df)

    def test_transform_before_fit_raises(self, preprocessor, reviews_from_raw):
        df = preprocessor.prepare_dataframe(reviews_from_raw)
        with pytest.raises(RuntimeError):
            preprocessor.transform_tfidf(df["text"])
