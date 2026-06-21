"""Build richer all-candidates sentiment features by candidate and hour."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "x_posts_with_sentiment_topics_all_candidates.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "x_sentiment_features_by_hour_candidate_all.csv"
WINDOWS = ["1h", "2h", "6h", "12h", "24h", "48h"]
TOPICS = [
    "security_war",
    "hostages_gaza",
    "economy_cost_of_living",
    "religion_state",
    "internal_security_crime",
    "judicial_reform_governance",
    "elections_polls",
    "coalition_opposition_government",
    "foreign_relations",
    "corruption_public_trust",
]
TOPIC_COLUMNS = [f"topic_{topic}" for topic in TOPICS]

BASIC_FEATURES = [
    "avg_sentiment_score",
    "median_sentiment_score",
    "std_sentiment_score",
    "var_sentiment_score",
    "total_sentiment_score",
    "num_posts",
    "num_positive_posts",
    "num_negative_posts",
    "num_neutral_posts",
    "total_likes",
    "total_reposts",
    "total_comments",
    "engagement_weighted_sentiment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rich candidate-hour sentiment features.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Input topic-tagged sentiment CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output candidate-hour sentiment features CSV path.")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def display_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)


def engagement_weighted_sentiment(group: pd.DataFrame) -> float:
    weights = group["likes"] + group["reposts"] + group["comments"] + 1
    weight_sum = weights.sum()
    if weight_sum == 0:
        return 0.0
    return float((group["sentiment_score"] * weights).sum() / weight_sum)


def create_full_hour_grid(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly.empty:
        return hourly

    candidates = sorted(hourly["candidate"].dropna().unique())
    min_hour = hourly["timestamp"].min()
    max_hour = hourly["timestamp"].max()
    hours = pd.date_range(start=min_hour, end=max_hour, freq="h", tz="UTC")
    grid = pd.MultiIndex.from_product([candidates, hours], names=["candidate", "timestamp"]).to_frame(index=False)
    full = grid.merge(hourly, on=["candidate", "timestamp"], how="left")

    numeric_columns = [column for column in full.columns if column not in {"candidate", "timestamp"}]
    full[numeric_columns] = full[numeric_columns].fillna(0)
    return full.sort_values(["candidate", "timestamp"]).reset_index(drop=True)


def add_rolling_features(full: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, candidate_df in full.groupby("candidate", sort=False):
        candidate_df = candidate_df.sort_values("timestamp").set_index("timestamp")
        for window in WINDOWS:
            rolling = candidate_df.rolling(window=window, min_periods=1)
            prefix = f"rolling_{window}"
            candidate_df[f"{prefix}_sentiment_mean"] = rolling["avg_sentiment_score"].mean()
            candidate_df[f"{prefix}_sentiment_median"] = rolling["avg_sentiment_score"].median()
            candidate_df[f"{prefix}_sentiment_std"] = rolling["avg_sentiment_score"].std().fillna(0)
            candidate_df[f"{prefix}_post_count"] = rolling["num_posts"].sum()
            candidate_df[f"{prefix}_positive_count"] = rolling["num_positive_posts"].sum()
            candidate_df[f"{prefix}_negative_count"] = rolling["num_negative_posts"].sum()
        frames.append(candidate_df.reset_index())
    return pd.concat(frames, ignore_index=True)


def build_topic_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    topic_frames = []
    keys = ["timestamp", "candidate"]
    for topic in TOPICS:
        flag_column = f"topic_{topic}"
        topic_df = df[df[flag_column] == 1].copy()
        if topic_df.empty:
            continue
        topic_df["topic_positive"] = (topic_df["sentiment_label"] == "positive").astype(int)
        topic_df["topic_negative"] = (topic_df["sentiment_label"] == "negative").astype(int)
        aggregated = (
            topic_df.groupby(keys, as_index=False)
            .agg(
                **{
                    f"{topic}_post_count": ("sentiment_score", "size"),
                    f"{topic}_avg_sentiment": ("sentiment_score", "mean"),
                    f"{topic}_negative_count": ("topic_negative", "sum"),
                    f"{topic}_positive_count": ("topic_positive", "sum"),
                }
            )
        )
        topic_frames.append(aggregated)

    if not topic_frames:
        return pd.DataFrame(columns=keys)

    topic_features = topic_frames[0]
    for frame in topic_frames[1:]:
        topic_features = topic_features.merge(frame, on=keys, how="outer")
    return topic_features


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    print(f"Input path: {display_path(input_path)}")
    print(f"Output path: {display_path(output_path)}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    required_columns = [
        "timestamp", "candidate", "sentiment_score", "sentiment_label", "likes", "reposts", "comments", *TOPIC_COLUMNS,
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.floor("h")
    df = df.dropna(subset=["timestamp", "candidate"]).copy()

    for column in ["sentiment_score", "likes", "reposts", "comments", *TOPIC_COLUMNS]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    df["is_positive"] = (df["sentiment_label"] == "positive").astype(int)
    df["is_negative"] = (df["sentiment_label"] == "negative").astype(int)
    df["is_neutral"] = (df["sentiment_label"] == "neutral").astype(int)

    keys = ["timestamp", "candidate"]
    base = (
        df.groupby(keys, as_index=False)
        .agg(
            avg_sentiment_score=("sentiment_score", "mean"),
            median_sentiment_score=("sentiment_score", "median"),
            std_sentiment_score=("sentiment_score", "std"),
            var_sentiment_score=("sentiment_score", "var"),
            total_sentiment_score=("sentiment_score", "sum"),
            num_posts=("sentiment_score", "size"),
            num_positive_posts=("is_positive", "sum"),
            num_negative_posts=("is_negative", "sum"),
            num_neutral_posts=("is_neutral", "sum"),
            total_likes=("likes", "sum"),
            total_reposts=("reposts", "sum"),
            total_comments=("comments", "sum"),
        )
    )

    weighted = df.groupby(keys).apply(engagement_weighted_sentiment, include_groups=False).reset_index(name="engagement_weighted_sentiment")
    base = base.merge(weighted, on=keys, how="left")

    topic_features = build_topic_aggregations(df)
    hourly = base.merge(topic_features, on=keys, how="left")

    for topic in TOPICS:
        for suffix in ["post_count", "avg_sentiment", "negative_count", "positive_count"]:
            column = f"{topic}_{suffix}"
            if column not in hourly.columns:
                hourly[column] = 0

    full = create_full_hour_grid(hourly)
    full = add_rolling_features(full)

    numeric_columns = [column for column in full.columns if column not in {"timestamp", "candidate"}]
    full[numeric_columns] = full[numeric_columns].fillna(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Rows created: {len(full)}")
    print(f"Candidates with sentiment feature rows: {full['candidate'].nunique()}")
    print(f"Feature columns: {len(full.columns)}")
    print(f"Saved sentiment features to {display_path(output_path)}")


if __name__ == "__main__":
    main()
