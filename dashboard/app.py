from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRICE_HISTORY_PATH = PROJECT_ROOT / "data" / "raw" / "polymarket_price_history.csv"
SENTIMENT_POSTS_PATH = PROJECT_ROOT / "data" / "processed" / "text_posts_with_sentiment.csv"
SENTIMENT_HOURLY_PATH = PROJECT_ROOT / "data" / "processed" / "sentiment_features_by_hour.csv"

MODEL_RESULT_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "binary_model_results.csv",
    PROJECT_ROOT / "data" / "processed" / "model_results.csv",
    PROJECT_ROOT / "reports" / "binary_model_results.csv",
    PROJECT_ROOT / "reports" / "model_results.csv",
]


st.set_page_config(page_title="Polymarket Sentiment Dashboard", layout="wide")
st.title("Polymarket Sentiment Prediction Dashboard")


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not load {path.name}: {exc}")
        return None


def find_timestamp_column(df: pd.DataFrame) -> str | None:
    candidates = ["timestamp", "datetime", "date", "time", "created_at"]
    for column in candidates:
        if column in df.columns:
            return column
    return None


def find_price_column(df: pd.DataFrame, candidate_name: str) -> str | None:
    lower_candidate = candidate_name.lower()
    preferred = [
        f"{lower_candidate}_price",
        f"{lower_candidate}_probability",
        f"price_{lower_candidate}",
        f"probability_{lower_candidate}",
    ]
    lower_to_original = {column.lower(): column for column in df.columns}
    for column in preferred:
        if column in lower_to_original:
            return lower_to_original[column]

    for column in df.columns:
        lowered = column.lower()
        if lower_candidate in lowered and ("price" in lowered or "prob" in lowered):
            return column
    return None


def plot_price_history(df: pd.DataFrame) -> None:
    timestamp_column = find_timestamp_column(df)
    if timestamp_column is None:
        st.info("Price history file exists, but no timestamp/date column was found.")
        st.dataframe(df.head(20), use_container_width=True)
        return

    df = df.copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="coerce")
    df = df.dropna(subset=[timestamp_column]).sort_values(timestamp_column)

    netanyahu_column = find_price_column(df, "netanyahu")
    bennett_column = find_price_column(df, "bennett")

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Netanyahu Price")
        if netanyahu_column:
            st.line_chart(df.set_index(timestamp_column)[netanyahu_column])
        else:
            st.info("No Netanyahu price column was detected.")

    with cols[1]:
        st.subheader("Bennett Price")
        if bennett_column:
            st.line_chart(df.set_index(timestamp_column)[bennett_column])
        else:
            st.info("No Bennett price column was detected.")

    st.subheader("Price Gap")
    if netanyahu_column and bennett_column:
        gap_df = df[[timestamp_column, netanyahu_column, bennett_column]].copy()
        gap_df["price_gap"] = gap_df[netanyahu_column] - gap_df[bennett_column]
        st.line_chart(gap_df.set_index(timestamp_column)["price_gap"])
    else:
        st.info("Price gap requires both Netanyahu and Bennett price columns.")


def show_model_results() -> None:
    st.header("Model Results Summary")
    for path in MODEL_RESULT_CANDIDATES:
        df = read_csv_if_exists(path)
        if df is not None:
            st.caption(f"Loaded {path.relative_to(PROJECT_ROOT)}")
            st.dataframe(df, use_container_width=True)
            return
    st.info("No model results file found yet.")


def show_sentiment_summary() -> None:
    st.header("Sentiment Summary")

    posts_df = read_csv_if_exists(SENTIMENT_POSTS_PATH)
    hourly_df = read_csv_if_exists(SENTIMENT_HOURLY_PATH)

    if posts_df is None and hourly_df is None:
        st.info("No sentiment files found yet. Run python src/sentiment.py first.")
        return

    if posts_df is not None:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Posts", len(posts_df))
        if "sentiment_score" in posts_df.columns:
            metric_cols[1].metric("Avg Sentiment", round(posts_df["sentiment_score"].mean(), 3))
        if "sentiment_label" in posts_df.columns:
            counts = posts_df["sentiment_label"].value_counts()
            metric_cols[2].metric("Positive", int(counts.get("positive", 0)))
            metric_cols[3].metric("Negative", int(counts.get("negative", 0)))

        st.subheader("Sentiment Labels")
        if "sentiment_label" in posts_df.columns:
            st.bar_chart(posts_df["sentiment_label"].value_counts())

        st.subheader("Recent Sentiment Posts")
        st.dataframe(posts_df.head(25), use_container_width=True)

    if hourly_df is not None:
        st.subheader("Hourly Sentiment")
        timestamp_column = find_timestamp_column(hourly_df)
        if timestamp_column and "avg_sentiment_score" in hourly_df.columns:
            hourly_df = hourly_df.copy()
            hourly_df[timestamp_column] = pd.to_datetime(hourly_df[timestamp_column], errors="coerce")
            hourly_df = hourly_df.dropna(subset=[timestamp_column]).sort_values(timestamp_column)
            st.line_chart(hourly_df.set_index(timestamp_column)["avg_sentiment_score"])
        else:
            st.dataframe(hourly_df, use_container_width=True)


st.header("Market Prices")
price_df = read_csv_if_exists(PRICE_HISTORY_PATH)
if price_df is not None:
    plot_price_history(price_df)
else:
    st.info("No Polymarket price history file found at data/raw/polymarket_price_history.csv.")

show_model_results()
show_sentiment_summary()
