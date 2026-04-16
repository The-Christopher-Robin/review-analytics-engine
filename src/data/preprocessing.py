import re
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from .schemas import Review

logger = logging.getLogger(__name__)


class TextPreprocessor:
    def __init__(self, config: dict):
        pp_config = config.get("preprocessing", {})
        self.max_length = pp_config.get("max_length", 256)
        self.min_review_length = pp_config.get("min_review_length", 10)
        self.remove_html = pp_config.get("remove_html", True)
        self.lowercase = pp_config.get("lowercase", True)

        self._tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_df=0.95,
        )
        self._fitted = False

    def clean_text(self, text: str) -> str:
        if self.remove_html:
            text = re.sub(r"<[^>]+>", " ", text)

        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"[^\w\s.,!?'-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if self.lowercase:
            text = text.lower()

        return text

    def prepare_dataframe(self, reviews: list[Review]) -> pd.DataFrame:
        records = []
        for r in reviews:
            cleaned = self.clean_text(r.text)
            if len(cleaned) >= self.min_review_length:
                records.append({
                    "review_id": r.review_id,
                    "text": cleaned,
                    "title": r.title,
                    "rating": r.rating,
                    "label": r.rating - 1,  # 0-indexed for models
                    "category": r.category,
                })

        df = pd.DataFrame(records)
        logger.info(f"Prepared {len(df)} reviews after cleaning (dropped {len(reviews) - len(df)})")
        return df

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=seed, stratify=df["label"]
        )
        logger.info(f"Split: {len(train_df)} train, {len(test_df)} test")
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def fit_tfidf(self, texts: pd.Series) -> np.ndarray:
        features = self._tfidf.fit_transform(texts)
        self._fitted = True
        logger.info(f"TF-IDF fitted: {features.shape[1]} features from {features.shape[0]} docs")
        return features

    def transform_tfidf(self, texts: pd.Series) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TF-IDF vectorizer not fitted yet. Call fit_tfidf first.")
        return self._tfidf.transform(texts)

    @property
    def feature_names(self) -> list[str]:
        if not self._fitted:
            return []
        return list(self._tfidf.get_feature_names_out())
