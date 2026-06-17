from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "text_posts_with_sentiment.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sentiment_features_by_hour.csv"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}. Run python src/sentiment.py first."
        )

    df = pd.read_csv(INPUT_PATH)
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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Saved hourly sentiment features to {OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Hourly rows created: {len(hourly)}")


if __name__ == "__main__":
    main()
