"""Merge all-candidates market features with candidate-hour sentiment features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKET_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "market_features_all_candidates.csv"
DEFAULT_SENTIMENT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "x_sentiment_features_by_hour_candidate_all.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "final" / "market_sentiment_dataset_all_candidates.csv"

SENTIMENT_COLUMNS = [
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
    parser = argparse.ArgumentParser(description="Build all-candidates market + sentiment dataset.")
    parser.add_argument("--market-input", default=str(DEFAULT_MARKET_INPUT_PATH), help="Input market features CSV path.")
    parser.add_argument("--sentiment-input", default=str(DEFAULT_SENTIMENT_INPUT_PATH), help="Input sentiment features CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output final dataset CSV path.")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def display_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)


def normalize_hour(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.floor("h")


def main() -> None:
    args = parse_args()
    market_input_path = resolve_path(args.market_input)
    sentiment_input_path = resolve_path(args.sentiment_input)
    output_path = resolve_path(args.output)

    print(f"Market input path: {display_path(market_input_path)}")
    print(f"Sentiment input path: {display_path(sentiment_input_path)}")
    print(f"Output path: {display_path(output_path)}")

    if not market_input_path.exists():
        raise FileNotFoundError(f"Market features file not found: {market_input_path}")

    market = pd.read_csv(market_input_path)
    required_market_columns = ["timestamp", "candidate", "horizon", "target_multiclass"]
    missing_market_columns = [column for column in required_market_columns if column not in market.columns]
    if missing_market_columns:
        raise ValueError(f"Missing market columns: {missing_market_columns}")

    market["timestamp"] = normalize_hour(market["timestamp"])
    market = market.dropna(subset=["timestamp", "candidate", "horizon", "target_multiclass"]).copy()

    if sentiment_input_path.exists():
        sentiment = pd.read_csv(sentiment_input_path)
        required_sentiment_columns = ["timestamp", "candidate", *SENTIMENT_COLUMNS]
        missing_sentiment_columns = [column for column in required_sentiment_columns if column not in sentiment.columns]
        if missing_sentiment_columns:
            raise ValueError(f"Missing sentiment columns: {missing_sentiment_columns}")
        sentiment["timestamp"] = normalize_hour(sentiment["timestamp"])
        sentiment = sentiment.dropna(subset=["timestamp", "candidate"]).copy()
    else:
        print("WARNING: Sentiment features file is missing. Filling all sentiment features with neutral/default values.")
        sentiment = pd.DataFrame(columns=["timestamp", "candidate", *SENTIMENT_COLUMNS])

    merged = market.merge(sentiment, on=["timestamp", "candidate"], how="left")
    for column in SENTIMENT_COLUMNS:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Rows created: {len(merged)}")
    print(f"Modeled candidates: {merged['candidate'].nunique()}")
    print("Target distribution by horizon:")
    print(merged.groupby(["horizon", "target_multiclass"]).size().to_string())
    print(f"Saved final all-candidates dataset to {display_path(output_path)}")


if __name__ == "__main__":
    main()
