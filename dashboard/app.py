from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRICE_HISTORY_PATH = PROJECT_ROOT / "data" / "raw" / "polymarket_price_history.csv"
SENTIMENT_POSTS_PATH = PROJECT_ROOT / "data" / "processed" / "text_posts_with_sentiment.csv"
SENTIMENT_HOURLY_PATH = PROJECT_ROOT / "data" / "processed" / "sentiment_features_by_hour.csv"
MARKET_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "market_features.csv"
FINAL_DATASET_PATH = PROJECT_ROOT / "data" / "final" / "market_sentiment_dataset.csv"
FINAL_RESULTS_PATH = PROJECT_ROOT / "results" / "final_model_comparison.md"

st.set_page_config(page_title="Polymarket Sentiment Dashboard", layout="wide")
st.title("Polymarket Sentiment Prediction")
st.caption("Final target: Up / Down / Stable price movement for Benjamin Netanyahu and Naftali Bennett. Current horizons: 1h, 2h, 24h.")


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not load {path.relative_to(PROJECT_ROOT)}: {exc}")
        return None


def read_text_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        st.warning(f"Could not load {path.relative_to(PROJECT_ROOT)}: {exc}")
        return None


def prepare_price_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["timestamp", "candidate", "price"])
    df["timestamp"] = df["timestamp"].dt.floor("h")
    return (
        df.groupby(["timestamp", "candidate"], as_index=False)
        .agg(price=("price", "mean"))
        .sort_values("timestamp")
    )


def show_market_prices() -> None:
    st.header("Market Prices")
    df = read_csv_if_exists(PRICE_HISTORY_PATH)
    if df is None:
        st.info("Missing data/raw/polymarket_price_history.csv. Run the Polymarket collection step first.")
        return

    required = {"timestamp", "candidate", "price"}
    if not required.issubset(df.columns):
        st.warning("Price history exists, but it does not contain timestamp, candidate, and price columns.")
        st.dataframe(df.head(20), use_container_width=True)
        return

    prices = prepare_price_history(df)
    wide = prices.pivot_table(index="timestamp", columns="candidate", values="price", aggfunc="mean").sort_index()

    cols = st.columns(3)
    for index, candidate in enumerate(["Benjamin Netanyahu", "Naftali Bennett"]):
        with cols[index]:
            st.subheader(candidate)
            if candidate in wide.columns:
                st.line_chart(wide[candidate])
                st.metric("Latest price", f"{wide[candidate].dropna().iloc[-1]:.3f}")
            else:
                st.info(f"No price series found for {candidate}.")

    with cols[2]:
        st.subheader("Price Gap")
        if {"Benjamin Netanyahu", "Naftali Bennett"}.issubset(wide.columns):
            gap = wide["Benjamin Netanyahu"] - wide["Naftali Bennett"]
            st.line_chart(gap)
            st.metric("Latest gap", f"{gap.dropna().iloc[-1]:.3f}")
        else:
            st.info("Price gap requires both candidate series.")


def show_multiclass_dataset() -> None:
    st.header("Multiclass Dataset")
    df = read_csv_if_exists(FINAL_DATASET_PATH)
    market_df = read_csv_if_exists(MARKET_FEATURES_PATH)
    active_df = df if df is not None else market_df

    if active_df is None:
        st.info("No final dataset yet. Run build_market_features.py and build_market_sentiment_dataset.py.")
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", len(active_df))
    metric_cols[1].metric("Candidates", active_df["candidate"].nunique() if "candidate" in active_df else 0)
    metric_cols[2].metric("Horizons", active_df["horizon"].nunique() if "horizon" in active_df else 0)
    metric_cols[3].metric("Target", "Up / Down / Stable")

    if "horizon" in active_df.columns and "target_multiclass" in active_df.columns:
        st.subheader("Class Distribution by Horizon")
        distribution = (
            active_df.groupby(["horizon", "target_multiclass"])
            .size()
            .reset_index(name="count")
            .pivot(index="horizon", columns="target_multiclass", values="count")
            .fillna(0)
            .astype(int)
        )
        st.dataframe(distribution, use_container_width=True)
        st.bar_chart(distribution)


    st.subheader("Sample Rows")
    st.dataframe(active_df.head(30), use_container_width=True)


def parse_result_metrics(text: str) -> pd.DataFrame:
    rows = []
    current_horizon = None
    current_model = None
    for line in text.splitlines():
        horizon_match = re.match(r"## Horizon: (.+)", line)
        model_match = re.match(r"### (.+)", line)
        accuracy_match = re.match(r"- Accuracy: ([0-9.]+)", line)
        f1_match = re.match(r"- Macro-F1: ([0-9.]+)", line)
        if horizon_match:
            current_horizon = horizon_match.group(1)
        elif model_match:
            current_model = model_match.group(1)
            rows.append({"horizon": current_horizon, "model": current_model})
        elif accuracy_match and rows:
            rows[-1]["accuracy"] = float(accuracy_match.group(1))
        elif f1_match and rows:
            rows[-1]["macro_f1"] = float(f1_match.group(1))
    return pd.DataFrame(rows)


def show_model_results() -> None:
    st.header("Final Model Comparison")
    text = read_text_if_exists(FINAL_RESULTS_PATH)
    if text is None:
        st.info("No final model comparison yet. Run python src/train_combined_model.py.")
        return

    metrics = parse_result_metrics(text)
    if not metrics.empty:
        st.subheader("Metrics Summary")
        st.dataframe(metrics, use_container_width=True)
        chart_df = metrics.dropna(subset=["macro_f1"]).pivot(index="model", columns="horizon", values="macro_f1")
        if not chart_df.empty:
            st.bar_chart(chart_df)

    with st.expander("Full results report", expanded=False):
        st.markdown(text)


def show_sentiment_summary() -> None:
    st.header("Sentiment Features")
    posts_df = read_csv_if_exists(SENTIMENT_POSTS_PATH)
    hourly_df = read_csv_if_exists(SENTIMENT_HOURLY_PATH)

    if posts_df is None and hourly_df is None:
        st.info("No sentiment outputs yet. Run python src/sentiment.py and python src/build_combined_dataset.py.")
        return

    if posts_df is not None:
        cols = st.columns(5)
        cols[0].metric("Posts", len(posts_df))
        if "sentiment_score" in posts_df:
            cols[1].metric("Avg score", f"{posts_df['sentiment_score'].mean():.3f}")
        if "sentiment_label" in posts_df:
            counts = posts_df["sentiment_label"].value_counts()
            cols[2].metric("Positive", int(counts.get("positive", 0)))
            cols[3].metric("Negative", int(counts.get("negative", 0)))
            cols[4].metric("Neutral", int(counts.get("neutral", 0)))
            st.bar_chart(counts)

        available_cols = [
            "timestamp", "text", "main_entity", "mentioned_parties", "political_topics",
            "matched_political_phrases", "sentiment_label", "target_sentiment", "intensity_score",
        ]
        st.subheader("Sample Sentiment Rows")
        st.dataframe(posts_df[[c for c in available_cols if c in posts_df.columns]].head(25), use_container_width=True)

    if hourly_df is not None and "timestamp" in hourly_df.columns:
        st.subheader("Hourly Sentiment")
        hourly_df = hourly_df.copy()
        hourly_df["timestamp"] = pd.to_datetime(hourly_df["timestamp"], errors="coerce")
        hourly_df = hourly_df.dropna(subset=["timestamp"]).sort_values("timestamp")
        if "avg_sentiment_score" in hourly_df:
            st.line_chart(hourly_df.set_index("timestamp")["avg_sentiment_score"])


st.info("X/Twitter data is used as predictive features; labels are derived from future Polymarket price movements. Current sentiment data is sample data until real X data is collected.")
show_market_prices()
show_multiclass_dataset()
show_model_results()
show_sentiment_summary()

