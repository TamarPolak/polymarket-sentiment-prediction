from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKET_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "market_features.csv"
SENTIMENT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "sentiment_features_by_hour.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "final" / "market_sentiment_dataset.csv"

SENTIMENT_NUMERIC_COLUMNS = [
    "avg_sentiment_score",
    "total_sentiment_score",
    "num_posts",
    "num_positive_posts",
    "num_negative_posts",
    "num_neutral_posts",
    "total_likes",
    "total_reposts",
    "total_comments",
]


def main() -> None:
    if not MARKET_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Market features not found: {MARKET_FEATURES_PATH}. Run python src/build_market_features.py first."
        )
    if not SENTIMENT_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Sentiment features not found: {SENTIMENT_FEATURES_PATH}. Run python src/build_combined_dataset.py first."
        )

    market = pd.read_csv(MARKET_FEATURES_PATH)
    sentiment = pd.read_csv(SENTIMENT_FEATURES_PATH)

    market["timestamp"] = pd.to_datetime(market["timestamp"], errors="coerce")
    sentiment["timestamp"] = pd.to_datetime(sentiment["timestamp"], errors="coerce")
    market = market.dropna(subset=["timestamp"]).sort_values("timestamp").copy()
    sentiment = sentiment.dropna(subset=["timestamp"]).sort_values("timestamp").copy()

    for column in SENTIMENT_NUMERIC_COLUMNS:
        if column not in sentiment.columns:
            sentiment[column] = 0
        sentiment[column] = pd.to_numeric(sentiment[column], errors="coerce").fillna(0)

    # Backward merge prevents leakage: each market row receives only sentiment features
    # from the same timestamp or the most recent earlier sentiment window.
    merged = pd.merge_asof(
        market,
        sentiment[["timestamp"] + SENTIMENT_NUMERIC_COLUMNS],
        on="timestamp",
        direction="backward",
    )

    merged[SENTIMENT_NUMERIC_COLUMNS] = merged[SENTIMENT_NUMERIC_COLUMNS].fillna(0)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Saved final market + sentiment dataset to {OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Rows created: {len(merged)}")
    print(merged.groupby("horizon")["target_multiclass"].value_counts().to_string())


if __name__ == "__main__":
    main()
