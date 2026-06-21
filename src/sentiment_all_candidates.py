"""Run lightweight sentiment scoring for all-candidates X posts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from sentiment import clean_text, detect_political_phrases, label_sentiment, score_sentiment


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "x_posts_all_candidates.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "x_posts_with_sentiment_all_candidates.csv"

OUTPUT_COLUMNS = [
    "timestamp", "text", "candidate", "query_group", "lang", "likes", "reposts", "comments", "post_id",
    "sentiment_score", "sentiment_label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all-candidates X sentiment processing.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Input X posts CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output sentiment CSV path.")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def display_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)


def score_text(text: str) -> tuple[int, str]:
    cleaned = clean_text(text)
    matched_phrases = detect_political_phrases(cleaned)
    score = score_sentiment(cleaned, matched_phrases)
    return score, label_sentiment(score)


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    print(f"Input path: {display_path(input_path)}")
    print(f"Output path: {display_path(output_path)}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    required_columns = ["timestamp", "text", "candidate", "query_group", "lang", "likes", "reposts", "comments", "post_id"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp", "text", "candidate"]).copy()

    for column in ["likes", "reposts", "comments"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

    scored = df["text"].apply(score_text)
    df["sentiment_score"] = [score for score, _ in scored]
    df["sentiment_label"] = [label for _, label in scored]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[OUTPUT_COLUMNS].to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Rows processed: {len(df)}")
    print("Sentiment label distribution:")
    print(df["sentiment_label"].value_counts().to_string())
    print(f"Saved sentiment output to {display_path(output_path)}")


if __name__ == "__main__":
    main()
