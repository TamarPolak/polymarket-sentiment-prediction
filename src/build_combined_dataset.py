from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "text_posts_with_sentiment.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sentiment_features_by_hour.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate sentiment features by hour.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Input sentiment CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output hourly features CSV path.")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    print(f"Input path: {input_path.relative_to(PROJECT_ROOT).as_posix() if input_path.is_relative_to(PROJECT_ROOT) else input_path}")
    print(f"Output path: {output_path.relative_to(PROJECT_ROOT).as_posix() if output_path.is_relative_to(PROJECT_ROOT) else output_path}")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. Run python src/sentiment.py first."
        )

    df = pd.read_csv(input_path)
    required_columns = [
        "timestamp",
        "sentiment_score",
        "sentiment_label",
        "likes",
        "reposts",
        "comments",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df["hour"] = df["timestamp"].dt.floor("h")

    for column in ["sentiment_score", "likes", "reposts", "comments"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    hourly = (
        df.groupby("hour")
        .agg(
            avg_sentiment_score=("sentiment_score", "mean"),
            total_sentiment_score=("sentiment_score", "sum"),
            num_posts=("sentiment_score", "size"),
            num_positive_posts=("sentiment_label", lambda values: (values == "positive").sum()),
            num_negative_posts=("sentiment_label", lambda values: (values == "negative").sum()),
            num_neutral_posts=("sentiment_label", lambda values: (values == "neutral").sum()),
            total_likes=("likes", "sum"),
            total_reposts=("reposts", "sum"),
            total_comments=("comments", "sum"),
        )
        .reset_index()
        .rename(columns={"hour": "timestamp"})
    )

    hourly["avg_sentiment_score"] = hourly["avg_sentiment_score"].round(4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved hourly sentiment features to {output_path.relative_to(PROJECT_ROOT).as_posix() if output_path.is_relative_to(PROJECT_ROOT) else output_path}")
    print(f"Hourly rows created: {len(hourly)}")


if __name__ == "__main__":
    main()

