import pytest
import yaml


@pytest.fixture
def sample_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_reviews_raw():
    return [
        {
            "review_id": "rev_000001",
            "product_id": "prod_0001",
            "rating": 5,
            "title": "Great headphones",
            "text": "These headphones have excellent sound quality and noise cancellation. "
                    "Battery life easily lasts through a full day of work. Very comfortable "
                    "for long listening sessions and the build quality feels premium.",
            "category": "Electronics",
            "verified_purchase": True,
        },
        {
            "review_id": "rev_000002",
            "product_id": "prod_0002",
            "rating": 1,
            "title": "Broke after a week",
            "text": "The left ear stopped working after just seven days. Returned it immediately. "
                    "Would not recommend to anyone looking for reliability.",
            "category": "Electronics",
            "verified_purchase": True,
        },
        {
            "review_id": "rev_000003",
            "product_id": "prod_0003",
            "rating": 3,
            "title": "Decent but overpriced",
            "text": "Sound quality is okay for the price point but nothing special. The app "
                    "is clunky and takes too long to connect. Comfortable enough though.",
            "category": "Electronics",
            "verified_purchase": False,
        },
        {
            "review_id": "rev_000004",
            "product_id": "prod_0001",
            "rating": 4,
            "title": "Solid upgrade",
            "text": "Upgraded from the previous model and the noise cancellation is noticeably "
                    "better. Wish the case was smaller but overall a good purchase.",
            "category": "Electronics",
            "verified_purchase": True,
        },
        {
            "review_id": "rev_000005",
            "product_id": "prod_0004",
            "rating": 2,
            "title": "Disappointing bass",
            "text": "Expected better bass response at this price range. Mids are muddy and "
                    "the EQ app barely helps. Build quality is fine but sound is mediocre.",
            "category": "Electronics",
            "verified_purchase": True,
        },
    ]
