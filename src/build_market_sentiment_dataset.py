from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKET_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "market_features.csv"
DEFAULT_SENTIMENT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "sentiment_features_by_hour.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "final" / "market_sentiment_dataset.csv"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge market features with hourly sentiment features.")
    parser.add_argument("--market-input", default=str(DEFAULT_MARKET_FEATURES_PATH), help="Market features CSV path.")
    parser.add_argument("--sentiment-input", default=str(DEFAULT_SENTIMENT_FEATURES_PATH), help="Hourly sentiment features CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output final dataset CSV path.")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def display_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)


def main() -> None:
    args = parse_args()
    market_features_path = resolve_path(args.market_input)
    sentiment_features_path = resolve_path(args.sentiment_input)
    output_path = resolve_path(args.output)

    print(f"Market input path: {display_path(market_features_path)}")
    print(f"Sentiment input path: {display_path(sentiment_features_path)}")
    print(f"Output path: {display_path(output_path)}")

    if not market_features_path.exists():
        raise FileNotFoundError(
            f"Market features not found: {market_features_path}. Run python src/build_market_features.py first."
        )
    if not sentiment_features_path.exists():
        raise FileNotFoundError(
            f"Sentiment features not found: {sentiment_features_path}. Run python src/build_combined_dataset.py first."
        )

    market = pd.read_csv(market_features_path)
    sentiment = pd.read_csv(sentiment_features_path)

    market["timestamp"] = pd.to_datetime(market["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
    sentiment["timestamp"] = pd.to_datetime(sentiment["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved final market + sentiment dataset to {display_path(output_path)}")
    print(f"Rows created: {len(merged)}")
    print(merged.groupby("horizon")["target_multiclass"].value_counts().to_string())


if __name__ == "__main__":
    main()


