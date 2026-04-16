import os
import logging
from pathlib import Path

import pandas as pd
import yaml

from .schemas import Review, ReviewBatch

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class DataLoader:
    """Loads review data from CSV (local dev) or Snowflake (production)."""

    def __init__(self, config: dict):
        self.config = config["data"]
        self.source = self.config["source"]
        self.sample_size = self.config.get("sample_size", 50000)

    def load(self) -> ReviewBatch:
        if self.source == "snowflake":
            df = self._load_from_snowflake()
        else:
            df = self._load_from_csv()

        if len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled {self.sample_size} reviews from {len(df)} total")

        reviews = self._validate_rows(df)
        batch = ReviewBatch(reviews=reviews, source=self.source)
        logger.info(f"Loaded {batch.size} validated reviews from {self.source}")
        return batch

    def _load_from_csv(self) -> pd.DataFrame:
        csv_path = self.config["csv_path"]
        if not Path(csv_path).exists():
            raise FileNotFoundError(
                f"Data file not found at {csv_path}. "
                "Run `python scripts/fetch_data.py` first."
            )
        df = pd.read_csv(csv_path)
        logger.info(f"Read {len(df)} rows from {csv_path}")
        return df

    def _load_from_snowflake(self) -> pd.DataFrame:
        sf_config = self.config["snowflake"]

        # resolve env vars for credentials
        account = os.environ.get("SNOWFLAKE_ACCOUNT", sf_config.get("account", ""))
        user = os.environ.get("SNOWFLAKE_USER", "")
        password = os.environ.get("SNOWFLAKE_PASSWORD", "")

        if not all([account, user, password]):
            raise EnvironmentError(
                "Snowflake credentials not set. Need SNOWFLAKE_ACCOUNT, "
                "SNOWFLAKE_USER, SNOWFLAKE_PASSWORD environment variables."
            )

        try:
            import snowflake.connector
        except ImportError:
            raise ImportError("snowflake-connector-python required for Snowflake source")

        conn = snowflake.connector.connect(
            account=account,
            user=user,
            password=password,
            warehouse=sf_config.get("warehouse", "COMPUTE_WH"),
            database=sf_config.get("database", "PRODUCT_ANALYTICS"),
            schema=sf_config.get("schema", "RAW"),
            role=sf_config.get("role"),
        )

        table = sf_config["table"]
        query = f"SELECT * FROM {table} LIMIT {self.sample_size}"
        logger.info(f"Querying Snowflake: {query}")

        try:
            df = pd.read_sql(query, conn)
        finally:
            conn.close()

        return df

    def _validate_rows(self, df: pd.DataFrame) -> list[Review]:
        required_cols = {"review_id", "product_id", "rating", "title", "text"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        reviews = []
        skipped = 0
        for _, row in df.iterrows():
            try:
                review = Review(
                    review_id=str(row["review_id"]),
                    product_id=str(row["product_id"]),
                    rating=int(row["rating"]),
                    title=str(row["title"]),
                    text=str(row["text"]),
                    category=str(row.get("category", "")) or None,
                    verified_purchase=bool(row.get("verified_purchase", False)),
                )
                reviews.append(review)
            except Exception:
                skipped += 1

        if skipped > 0:
            logger.warning(f"Skipped {skipped} invalid rows during validation")

        return reviews
