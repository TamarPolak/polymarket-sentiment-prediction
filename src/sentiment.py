from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "text_posts_sample.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "text_posts_with_sentiment.csv"


POSITIVE_WORDS = {
    "strong",
    "stronger",
    "confident",
    "clear",
    "effective",
    "support",
    "positive",
    "praise",
    "practical",
    "focused",
    "safe",
    "gaining",
    "momentum",
    "surprise",
    "stable",
    "dominates",
    "chance",
    "חזק",
    "חזקה",
    "מתחזק",
    "מוביל",
    "חיובי",
    "חיוביות",
    "תמיכה",
    "אמין",
    "יציב",
    "ברור",
    "ניסיון",
    "טובתו",
}

NEGATIVE_WORDS = {
    "weak",
    "criticism",
    "criticized",
    "bad",
    "tough",
    "difficult",
    "hard",
    "lacks",
    "unstable",
    "disappointing",
    "lost",
    "עייפות",
    "ביקורת",
    "קשה",
    "מתקשה",
    "איבד",
    "חוסר",
    "מאכזב",
}


def clean_text(text: str) -> str:
    """Normalize text while preserving English and Hebrew words."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[^0-9a-zA-Z\u0590-\u05FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def score_sentiment(clean_text_value: str) -> int:
    tokens = clean_text_value.split()
    score = 0

    for token in tokens:
        if token in POSITIVE_WORDS:
            score += 1
        if token in NEGATIVE_WORDS:
            score -= 1

    return score


def label_sentiment(score: int) -> str:
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    required_columns = ["timestamp", "text", "candidate", "likes", "reposts", "comments"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "text"]).copy()

    for column in ["likes", "reposts", "comments"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

    df["clean_text"] = df["text"].apply(clean_text)
    df["sentiment_score"] = df["clean_text"].apply(score_sentiment)
    df["sentiment_label"] = df["sentiment_score"].apply(label_sentiment)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[
        [
            "timestamp",
            "text",
            "candidate",
            "likes",
            "reposts",
            "comments",
            "clean_text",
            "sentiment_score",
            "sentiment_label",
        ]
    ].to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Saved sentiment output to {OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Rows processed: {len(df)}")
    print(df["sentiment_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
