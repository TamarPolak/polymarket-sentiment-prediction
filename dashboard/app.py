from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRICE_HISTORY_PATH = PROJECT_ROOT / "data" / "raw" / "polymarket_price_history.csv"

X_POSTS_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "x_posts_real_limited.csv"
SENTIMENT_POSTS_PATH = PROJECT_ROOT / "data" / "processed" / "x_posts_with_sentiment.csv"
SENTIMENT_HOURLY_PATH = PROJECT_ROOT / "data" / "processed" / "x_sentiment_features_by_hour.csv"

MARKET_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "market_features.csv"
FINAL_DATASET_PATH = PROJECT_ROOT / "data" / "final" / "market_sentiment_dataset_real_x.csv"
FINAL_RESULTS_PATH = PROJECT_ROOT / "results" / "final_model_comparison_real_x.md"


st.set_page_config(
    page_title="Polymarket Sentiment Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Polymarket Sentiment Prediction")
st.caption(
    "Final target: Up / Down / Stable price movement for Benjamin Netanyahu and Naftali Bennett. "
    "Current horizons: 1h, 2h, 24h."
)


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


def show_x_collection_summary() -> None:
    st.header("1. Real X/Twitter Data Collection")

    df = read_csv_if_exists(X_POSTS_RAW_PATH)
    if df is None:
        st.info("Real X dataset was not found locally. Expected: data/raw/x_posts_real_limited.csv")
        return

    cols = st.columns(4)

    cols[0].metric("Real X posts", f"{len(df):,}")

    if "post_id" in df.columns:
        cols[1].metric("Unique posts", f"{df['post_id'].nunique():,}")
    else:
        cols[1].metric("Unique posts", f"{len(df):,}")

    if "candidate" in df.columns:
        cols[2].metric("Candidates", df["candidate"].nunique())
    else:
        cols[2].metric("Candidates", "2")

    if "lang" in df.columns:
        cols[3].metric("Languages", df["lang"].nunique())
    else:
        cols[3].metric("Languages", "Hebrew + English")

    st.write(
        """
The project uses real public X/Twitter posts collected with the X Recent Search API.
The collection was performed under a limited prepaid budget and includes Hebrew and English posts.
"""
    )

    if "query_group" in df.columns:
        st.subheader("Query Group Distribution")

        query_counts = df["query_group"].value_counts().reset_index()
        query_counts.columns = ["query_group", "count"]

        st.dataframe(query_counts, use_container_width=True, hide_index=True)
        st.bar_chart(query_counts.set_index("query_group"))

    if "lang" in df.columns:
        st.subheader("Language Distribution")

        lang_counts = df["lang"].fillna("unknown").value_counts().reset_index()
        lang_counts.columns = ["language", "count"]

        st.dataframe(lang_counts, use_container_width=True, hide_index=True)
        st.bar_chart(lang_counts.set_index("language"))

    available_cols = [
        "timestamp",
        "candidate",
        "query_group",
        "lang",
        "likes",
        "reposts",
        "comments",
        "text",
    ]
    sample_cols = [c for c in available_cols if c in df.columns]

    st.subheader("Sample Real X Posts")
    st.dataframe(df[sample_cols].head(20), use_container_width=True)


def show_market_prices() -> None:
    st.header("2. Polymarket Market Prices")

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
    wide = prices.pivot_table(
        index="timestamp",
        columns="candidate",
        values="price",
        aggfunc="mean",
    ).sort_index()

    cols = st.columns(3)

    for index, candidate in enumerate(["Benjamin Netanyahu", "Naftali Bennett"]):
        with cols[index]:
            st.subheader(candidate)

            if candidate in wide.columns:
                st.line_chart(wide[candidate])
                latest_price = wide[candidate].dropna().iloc[-1]
                st.metric("Latest price", f"{latest_price:.3f}")
            else:
                st.info(f"No price series found for {candidate}.")

    with cols[2]:
        st.subheader("Price Gap")

        if {"Benjamin Netanyahu", "Naftali Bennett"}.issubset(wide.columns):
            gap = wide["Benjamin Netanyahu"] - wide["Naftali Bennett"]
            st.line_chart(gap)
            latest_gap = gap.dropna().iloc[-1]
            st.metric("Latest gap", f"{latest_gap:.3f}")
        else:
            st.info("Price gap requires both candidate series.")


def show_multiclass_dataset() -> None:
    st.header("3. Final Multiclass Dataset")

    df = read_csv_if_exists(FINAL_DATASET_PATH)
    market_df = read_csv_if_exists(MARKET_FEATURES_PATH)
    active_df = df if df is not None else market_df

    if active_df is None:
        st.info("No final dataset yet. Run build_market_features.py and build_market_sentiment_dataset.py.")
        return

    metric_cols = st.columns(4)

    metric_cols[0].metric("Rows", f"{len(active_df):,}")

    if "candidate" in active_df.columns:
        metric_cols[1].metric("Candidates", active_df["candidate"].nunique())
    else:
        metric_cols[1].metric("Candidates", 0)

    if "horizon" in active_df.columns:
        metric_cols[2].metric("Horizons", active_df["horizon"].nunique())
    else:
        metric_cols[2].metric("Horizons", 0)

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

    st.subheader("Sample Final Dataset Rows")
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
    st.header("4. Final Model Comparison")

    text = read_text_if_exists(FINAL_RESULTS_PATH)
    if text is None:
        st.info("No final model comparison yet. Run python src/train_combined_model.py.")
        return

    metrics = parse_result_metrics(text)

    if not metrics.empty:
        st.subheader("Metrics Summary")

        horizon_order = ["1h", "2h", "24h"]
        metrics["horizon"] = pd.Categorical(metrics["horizon"], categories=horizon_order, ordered=True)
        metrics = metrics.sort_values(["horizon", "macro_f1"], ascending=[True, False])

        st.dataframe(metrics, use_container_width=True, hide_index=True)

        st.subheader("Best Model by Horizon")

        best_by_horizon = (
            metrics.sort_values(["horizon", "macro_f1"], ascending=[True, False])
            .groupby("horizon", observed=False)
            .head(1)
            .reset_index(drop=True)
        )

        st.dataframe(best_by_horizon, use_container_width=True, hide_index=True)

        st.subheader("Macro-F1 by Model and Horizon")

        chart_df = metrics.dropna(subset=["macro_f1"]).pivot(
            index="model",
            columns="horizon",
            values="macro_f1",
        )

        if not chart_df.empty:
            st.bar_chart(chart_df)

    with st.expander("Full results report", expanded=False):
        st.markdown(text)


def show_sentiment_summary() -> None:
    st.header("5. Sentiment Features")

    posts_df = read_csv_if_exists(SENTIMENT_POSTS_PATH)
    hourly_df = read_csv_if_exists(SENTIMENT_HOURLY_PATH)

    if posts_df is None and hourly_df is None:
        st.info("No sentiment outputs yet. Run python src/sentiment.py and python src/build_combined_dataset.py.")
        return

    if posts_df is not None:
        cols = st.columns(5)

        cols[0].metric("Posts with sentiment", f"{len(posts_df):,}")

        if "sentiment_score" in posts_df.columns:
            avg_score = posts_df["sentiment_score"].mean()
            cols[1].metric("Avg score", f"{avg_score:.3f}")

        if "sentiment_label" in posts_df.columns:
            counts = posts_df["sentiment_label"].value_counts()

            cols[2].metric("Positive", int(counts.get("positive", 0)))
            cols[3].metric("Negative", int(counts.get("negative", 0)))
            cols[4].metric("Neutral", int(counts.get("neutral", 0)))

            st.subheader("Sentiment Label Distribution")
            st.bar_chart(counts)

        if "candidate" in posts_df.columns and "sentiment_label" in posts_df.columns:
            st.subheader("Sentiment by Candidate")

            sentiment_by_candidate = (
                posts_df.groupby(["candidate", "sentiment_label"])
                .size()
                .reset_index(name="count")
            )

            st.dataframe(sentiment_by_candidate, use_container_width=True, hide_index=True)

        available_cols = [
            "timestamp",
            "candidate",
            "query_group",
            "lang",
            "text",
            "sentiment_label",
            "sentiment_score",
            "target_sentiment",
            "intensity_score",
            "main_entity",
            "mentioned_parties",
            "political_topics",
            "matched_political_phrases",
        ]

        st.subheader("Sample Sentiment Rows")

        existing_cols = [c for c in available_cols if c in posts_df.columns]
        st.dataframe(posts_df[existing_cols].head(25), use_container_width=True)

    if hourly_df is not None and "timestamp" in hourly_df.columns:
        st.subheader("Hourly Sentiment Features")

        hourly_df = hourly_df.copy()
        hourly_df["timestamp"] = pd.to_datetime(hourly_df["timestamp"], errors="coerce")
        hourly_df = hourly_df.dropna(subset=["timestamp"]).sort_values("timestamp")

        st.dataframe(hourly_df.head(30), use_container_width=True)

        if "avg_sentiment_score" in hourly_df.columns:
            st.line_chart(hourly_df.set_index("timestamp")["avg_sentiment_score"])


def show_conclusions() -> None:
    st.header("6. Conclusions")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("1h Horizon")
        st.metric("Best Macro-F1", "0.4416")
        st.write(
            """
Best model: **Market-only Random Forest**.

For the 1h horizon, sentiment features did not improve the model.
"""
        )

    with c2:
        st.subheader("2h Horizon")
        st.metric("Best Macro-F1", "0.4968")
        st.write(
            """
Best model: **Market + Sentiment Random Forest**.

This was the clearest case where sentiment features improved prediction quality.
"""
        )

    with c3:
        st.subheader("24h Horizon")
        st.metric("Best Macro-F1", "0.3631")
        st.write(
            """
Best Macro-F1 model: **Market + Sentiment Logistic Regression**.

The result is mixed: sentiment improved Macro-F1, but Market-only Random Forest had higher Accuracy.
"""
        )

    st.success(
        """
Overall conclusion: X/Twitter sentiment showed a mixed but meaningful contribution.
The strongest improvement appeared in the 2h horizon, suggesting that sentiment may contain useful short-term predictive signal.
However, the effect was not consistent across all prediction windows.
"""
    )


def show_limitations() -> None:
    st.header("7. Limitations and Future Work")

    st.write(
        """
Main limitations:
- X/Twitter API access was limited by a small prepaid academic budget.
- The final collection includes 996 public posts, but a larger dataset could improve robustness.
- Hebrew sentiment analysis is challenging and may require a stronger Hebrew-specific sentiment model.
- Polymarket prices are available at hourly resolution, so the active horizons are 1h, 2h and 24h.

Future work:
- Collect more data over a longer time period.
- Improve Hebrew sentiment classification.
- Add more candidates and political events.
- Compare additional models and feature engineering methods.
"""
    )


st.info(
    "X/Twitter data is used as predictive features; labels are derived from future Polymarket price movements. "
    "This dashboard uses the real X dataset collected for the project: 996 unique public posts in Hebrew and English."
)

show_x_collection_summary()
show_market_prices()
show_multiclass_dataset()
show_model_results()
show_sentiment_summary()
show_conclusions()
show_limitations()