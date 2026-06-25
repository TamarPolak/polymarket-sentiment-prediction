from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PUBLIC_FINAL_DATASET_PATH = PROJECT_ROOT / "data" / "public" / "market_sentiment_dataset_all_candidates.csv"
LOCAL_FINAL_DATASET_PATH = PROJECT_ROOT / "data" / "final" / "market_sentiment_dataset_all_candidates.csv"
PUBLIC_PRICE_HISTORY_PATH = PROJECT_ROOT / "data" / "public" / "polymarket_price_history_all_candidates.csv"
LOCAL_PRICE_HISTORY_PATH = PROJECT_ROOT / "data" / "raw" / "polymarket_price_history_all_candidates.csv"
PUBLIC_TOPIC_POSTS_PATH = PROJECT_ROOT / "data" / "public" / "x_posts_with_sentiment_topics_all_candidates_public.csv"
LOCAL_TOPIC_POSTS_PATH = PROJECT_ROOT / "data" / "processed" / "x_posts_with_sentiment_topics_all_candidates.csv"
RESULTS_PATH = PROJECT_ROOT / "results" / "final_model_comparison_all_candidates.md"


def prefer_public_path(public_path: Path, local_path: Path) -> Path:
    return public_path if public_path.exists() else local_path


FINAL_DATASET_PATH = prefer_public_path(PUBLIC_FINAL_DATASET_PATH, LOCAL_FINAL_DATASET_PATH)
PRICE_HISTORY_PATH = prefer_public_path(PUBLIC_PRICE_HISTORY_PATH, LOCAL_PRICE_HISTORY_PATH)
TOPIC_POSTS_PATH = prefer_public_path(PUBLIC_TOPIC_POSTS_PATH, LOCAL_TOPIC_POSTS_PATH)

HORIZONS = ["1h", "2h", "24h"]
TARGET_CLASSES = ["Up", "Down", "Stable"]
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
TOPIC_LABELS = {
    "security_war": "Security/war",
    "hostages_gaza": "Hostages/Gaza",
    "economy_cost_of_living": "Economy",
    "religion_state": "Religion/state",
    "internal_security_crime": "Internal security",
    "judicial_reform_governance": "Judiciary",
    "elections_polls": "Elections/polls",
    "coalition_opposition_government": "Coalition/gov",
    "foreign_relations": "Foreign relations",
    "corruption_public_trust": "Public trust",
}
MARKET_FEATURES = [
    "candidate_price",
    "candidate_change_1h",
    "candidate_change_2h",
    "candidate_change_24h",
    "market_leader_price",
    "gap_to_market_leader",
    "candidate_rank",
    "price_share",
    "top_2_gap",
    "market_num_candidates",
]
SENTIMENT_FEATURES = [
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
    "rolling_24h_sentiment_mean",
    "rolling_24h_post_count",
    "rolling_48h_sentiment_mean",
    "rolling_48h_post_count",
]

st.set_page_config(page_title="All-Candidates Polymarket Sentiment", layout="wide")
st.title("All-Candidates Polymarket + X Sentiment Dashboard")
st.caption(
    "The model predicts candidate price movement on Polymarket, not the final election winner directly. "
    "X/Twitter data is used as explanatory and predictive features."
)


@st.cache_data(show_spinner=False)
def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not load {path.relative_to(PROJECT_ROOT)}: {exc}")
        return None


@st.cache_data(show_spinner=False)
def read_text_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        st.warning(f"Could not load {path.relative_to(PROJECT_ROOT)}: {exc}")
        return None


def prepare_timestamp(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors="coerce", utc=True)
    return df.dropna(subset=[column])


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    return df


def normalize_0_1(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    min_value = series.min()
    max_value = series.max()
    if max_value == min_value:
        return series * 0
    return (series - min_value) / (max_value - min_value)


def show_no_x_data_message() -> None:
    st.info("No X data available for this candidate. Neutral/default sentiment features were used after the merge.")


def candidate_has_x_posts(posts_df: pd.DataFrame | None, candidate: str | None) -> bool:
    if posts_df is None or not candidate or "candidate" not in posts_df.columns:
        return False
    return not posts_df[posts_df["candidate"] == candidate].empty


def candidate_has_nonzero_sentiment(data: pd.DataFrame, sentiment_columns: list[str]) -> bool:
    available = [column for column in sentiment_columns if column in data.columns]
    if not available or data.empty:
        return False
    values = data[available].apply(pd.to_numeric, errors="coerce").fillna(0)
    return bool((values.abs().sum(axis=1) > 0).any())


def clean_feature_display(row: pd.Series, columns: list[str]) -> pd.DataFrame:
    display = pd.DataFrame([row[columns]]).copy()
    numeric_columns = [column for column in display.columns if column != "timestamp"]
    for column in numeric_columns:
        display[column] = pd.to_numeric(display[column], errors="coerce").fillna(0)
    if "timestamp" in display.columns:
        display["timestamp"] = display["timestamp"].fillna("No X data")
    return display.fillna("No X data")


def parse_result_metrics(text: str | None) -> pd.DataFrame:
    if not text:
        return pd.DataFrame()

    rows = []
    current_horizon = None
    current_model = None
    for line in text.splitlines():
        horizon_match = re.match(r"## Horizon: (.+)", line)
        model_match = re.match(r"### (.+)", line)
        accuracy_match = re.match(r"- Accuracy: ([0-9.]+)", line)
        f1_match = re.match(r"- Macro-F1: ([0-9.]+)", line)

        if horizon_match:
            current_horizon = horizon_match.group(1).strip()
        elif model_match:
            current_model = model_match.group(1).strip()
            if current_model != "Horizon Summary":
                rows.append({"horizon": current_horizon, "model": current_model})
        elif accuracy_match and rows:
            rows[-1]["accuracy"] = float(accuracy_match.group(1))
        elif f1_match and rows:
            rows[-1]["macro_f1"] = float(f1_match.group(1))

    return pd.DataFrame(rows)


def latest_price_table(price_df: pd.DataFrame) -> pd.DataFrame:
    prices = prepare_timestamp(price_df)
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    prices = prices.dropna(subset=["candidate", "price"])
    latest_idx = prices.sort_values("timestamp").groupby("candidate")["timestamp"].idxmax()
    latest = prices.loc[latest_idx, ["timestamp", "candidate", "price"]].copy()
    latest = latest.sort_values("price", ascending=False).reset_index(drop=True)
    latest["rank"] = latest["price"].rank(ascending=False, method="min").astype(int)
    return latest[["rank", "candidate", "price", "timestamp"]]


def candidate_series(price_df: pd.DataFrame, candidate: str) -> pd.DataFrame:
    prices = prepare_timestamp(price_df)
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    prices = prices[prices["candidate"] == candidate].dropna(subset=["price"])
    return prices.groupby("timestamp", as_index=False).agg(price=("price", "mean")).sort_values("timestamp")


def compute_change_table(final_df: pd.DataFrame, window: str) -> pd.DataFrame:
    data = prepare_timestamp(final_df)
    data = data[data["horizon"] == "1h"].sort_values(["candidate", "timestamp"])
    keep_cols = ["timestamp", "candidate", "candidate_price", f"rolling_{window}_sentiment_mean"]
    keep_cols = [column for column in keep_cols if column in data.columns]
    data = data[keep_cols].drop_duplicates(["candidate", "timestamp"])
    rows = []
    delta = pd.Timedelta(hours=int(window.replace("h", "")))

    for candidate, group in data.groupby("candidate"):
        group = group.sort_values("timestamp")
        if group.empty:
            continue
        latest = group.iloc[-1]
        past_time = latest["timestamp"] - delta
        past_rows = group[group["timestamp"] <= past_time]
        if past_rows.empty:
            continue
        past = past_rows.iloc[-1]
        sentiment_col = f"rolling_{window}_sentiment_mean"
        rows.append(
            {
                "candidate": candidate,
                "latest_price": latest.get("candidate_price", 0),
                f"price_change_{window}": latest.get("candidate_price", 0) - past.get("candidate_price", 0),
                f"rolling_sentiment_change_{window}": latest.get(sentiment_col, 0) - past.get(sentiment_col, 0),
            }
        )
    return pd.DataFrame(rows).sort_values(f"price_change_{window}", key=lambda s: s.abs(), ascending=False)


def topic_summary(posts_df: pd.DataFrame, candidate: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    posts = posts_df[posts_df["candidate"] == candidate].copy()
    rows = []
    sentiment_rows = []
    for topic in TOPICS:
        column = f"topic_{topic}"
        if column not in posts.columns:
            continue
        topic_posts = posts[pd.to_numeric(posts[column], errors="coerce").fillna(0) == 1]
        rows.append({"topic": topic, "post_count": len(topic_posts)})
        if not topic_posts.empty:
            sentiment_rows.append(
                {
                    "topic": topic,
                    "avg_sentiment": pd.to_numeric(topic_posts["sentiment_score"], errors="coerce").fillna(0).mean(),
                    "positive": (topic_posts["sentiment_label"] == "positive").sum(),
                    "negative": (topic_posts["sentiment_label"] == "negative").sum(),
                    "neutral": (topic_posts["sentiment_label"] == "neutral").sum(),
                }
            )
        else:
            sentiment_rows.append({"topic": topic, "avg_sentiment": 0, "positive": 0, "negative": 0, "neutral": 0})
    counts = pd.DataFrame(rows)
    sentiment = pd.DataFrame(sentiment_rows)
    if not counts.empty:
        counts["topic_label"] = counts["topic"].map(TOPIC_LABELS).fillna(counts["topic"])
        counts = counts.sort_values("post_count", ascending=False)
    if not sentiment.empty:
        sentiment["topic_label"] = sentiment["topic"].map(TOPIC_LABELS).fillna(sentiment["topic"])
    return counts, sentiment


def attention_table(posts_df: pd.DataFrame, final_df: pd.DataFrame) -> pd.DataFrame:
    posts = numeric(posts_df.copy(), ["likes", "reposts", "comments"])
    post_count_source = "post_id" if "post_id" in posts.columns else "candidate"
    post_count_agg = "nunique" if post_count_source == "post_id" else "size"
    base = posts.groupby("candidate", as_index=False).agg(
        post_count=(post_count_source, post_count_agg),
        total_likes=("likes", "sum"),
        total_reposts=("reposts", "sum"),
        total_comments=("comments", "sum"),
    )
    base["total_engagement"] = base["total_likes"] + base["total_reposts"] + base["total_comments"]

    data = prepare_timestamp(final_df)
    data = data[data["horizon"] == "1h"].sort_values("timestamp")
    if "rolling_24h_post_count" in data.columns:
        latest = data.groupby("candidate").tail(1)[["candidate", "rolling_24h_post_count"]]
        base = base.merge(latest, on="candidate", how="left")
    return base.fillna(0).sort_values("post_count", ascending=False)


def prepare_model_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    available = [column for column in feature_columns if column in df.columns]
    X = df[available + ["candidate"]].copy()
    for column in available:
        X[column] = pd.to_numeric(X[column], errors="coerce").fillna(0)
    return pd.get_dummies(X, columns=["candidate"], drop_first=False).astype(float)


def example_prediction(final_df: pd.DataFrame, candidate: str, horizon: str) -> tuple[str | None, pd.Series | None, str]:
    data = prepare_timestamp(final_df)
    data = data[(data["candidate"] == candidate) & (data["horizon"] == horizon)].sort_values("timestamp").copy()
    if len(data) < 20:
        return None, None, "Not enough rows for an example prediction."

    latest_row = data.iloc[-1]
    train_df = data.iloc[:-1].copy()
    if train_df["target_multiclass"].nunique() < 2:
        return None, latest_row, "Not enough class variety before the latest row."

    forbidden_columns = {"target_multiclass", "future_price", "price_change"}
    features = [
        column for column in MARKET_FEATURES + SENTIMENT_FEATURES
        if column in data.columns and column not in forbidden_columns and "future" not in column.lower() and "target" not in column.lower()
    ]
    X_train = prepare_model_features(train_df, features)
    X_latest = prepare_model_features(pd.DataFrame([latest_row]), features)
    X_train, X_latest = X_train.align(X_latest, join="left", axis=1, fill_value=0)

    model = RandomForestClassifier(n_estimators=200, random_state=42, min_samples_leaf=2, class_weight="balanced")
    model.fit(X_train, train_df["target_multiclass"])
    prediction = model.predict(X_latest)[0]
    return str(prediction), latest_row, "Prediction generated by a dashboard-only Random Forest trained on earlier rows for the selected candidate/horizon."


price_df = read_csv_if_exists(PRICE_HISTORY_PATH)
final_df = read_csv_if_exists(FINAL_DATASET_PATH)
topic_posts_df = read_csv_if_exists(TOPIC_POSTS_PATH)
results_text = read_text_if_exists(RESULTS_PATH)

if price_df is None and final_df is None and topic_posts_df is None:
    st.warning("No all-candidates data files were found yet. Run the pipeline first.")

candidates = []
for df in [final_df, price_df, topic_posts_df]:
    if df is not None and "candidate" in df.columns:
        candidates.extend(df["candidate"].dropna().unique().tolist())
candidates = sorted(set(candidates))
selected_candidate = st.sidebar.selectbox("Candidate", candidates) if candidates else None
selected_horizon = st.sidebar.selectbox("Prediction horizon", HORIZONS)
selected_window = st.sidebar.selectbox("Change window", ["24h", "48h"])

st.sidebar.markdown("Run dashboard:")
st.sidebar.code("python -m streamlit run dashboard/app.py", language="powershell")

section_1, section_2, section_3, section_4, section_5, section_6, section_7 = st.tabs(
    [
        "1. Current Market",
        "2. 24h/48h Changes",
        "3. Topics",
        "4. Sentiment vs Price",
        "5. X Attention",
        "6. Model Performance",
        "7. Example Prediction",
    ]
)

with section_1:
    st.header("Current Polymarket state")
    if price_df is None:
        st.info("Missing data/raw/polymarket_price_history_all_candidates.csv")
    else:
        latest = latest_price_table(price_df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Candidates", latest["candidate"].nunique())
        c2.metric("Current leader", latest.iloc[0]["candidate"] if not latest.empty else "N/A")
        c3.metric("Leader price", f"{latest.iloc[0]['price']:.3f}" if not latest.empty else "N/A")
        st.subheader("Latest prices and ranks")
        st.dataframe(latest, use_container_width=True, hide_index=True)
        if selected_candidate:
            st.subheader(f"Price over time: {selected_candidate}")
            series = candidate_series(price_df, selected_candidate)
            if series.empty:
                st.info("No price series found for this candidate.")
            else:
                st.line_chart(series.set_index("timestamp")["price"])

with section_2:
    st.header(f"Who changed most in {selected_window}?")
    if final_df is None:
        st.info("Missing data/final/market_sentiment_dataset_all_candidates.csv")
    else:
        changes = compute_change_table(final_df, selected_window)
        if changes.empty:
            st.info("Not enough history to compute this window.")
        else:
            st.dataframe(changes, use_container_width=True, hide_index=True)
            st.subheader(f"Absolute price change by candidate ({selected_window})")
            chart = changes.set_index("candidate")[[f"price_change_{selected_window}"]]
            st.bar_chart(chart)
            sentiment_col = f"rolling_sentiment_change_{selected_window}"
            if sentiment_col in changes.columns:
                st.subheader(f"Rolling sentiment change by candidate ({selected_window})")
                st.bar_chart(changes.set_index("candidate")[[sentiment_col]])

with section_3:
    st.header("What topics are discussed around a selected candidate?")
    if topic_posts_df is None or not selected_candidate:
        st.info("Missing public/local topic-tagged X features or selected candidate.")
    else:
        topic_counts, sentiment_by_topic = topic_summary(topic_posts_df, selected_candidate)
        if not candidate_has_x_posts(topic_posts_df, selected_candidate) or topic_counts["post_count"].sum() == 0:
            show_no_x_data_message()
        st.subheader("Topic distribution")
        st.bar_chart(topic_counts.set_index("topic_label")[["post_count"]])
        st.subheader("Sentiment by topic")
        st.dataframe(sentiment_by_topic, use_container_width=True, hide_index=True)
        st.bar_chart(sentiment_by_topic.set_index("topic_label")[["positive", "negative", "neutral"]])

with section_4:
    st.header("Does sentiment move with price?")
    if final_df is None or not selected_candidate:
        st.info("Missing final dataset or selected candidate.")
    else:
        data = prepare_timestamp(final_df)
        data = data[(data["candidate"] == selected_candidate) & (data["horizon"] == "1h")].drop_duplicates(["timestamp", "candidate"])
        cols = ["timestamp", "candidate_price", "rolling_24h_sentiment_mean"]
        cols = [column for column in cols if column in data.columns]
        data = numeric(data[cols].copy(), ["candidate_price", "rolling_24h_sentiment_mean"]).sort_values("timestamp")
        if not candidate_has_x_posts(topic_posts_df, selected_candidate) or not candidate_has_nonzero_sentiment(data, ["rolling_24h_sentiment_mean"]):
            show_no_x_data_message()
        if data.empty or len(cols) < 3:
            st.info("Required price/sentiment columns are missing for this candidate.")
        else:
            chart_data = data.set_index("timestamp")[["candidate_price", "rolling_24h_sentiment_mean"]].copy()
            chart_data["price_normalized"] = normalize_0_1(chart_data["candidate_price"])
            chart_data["rolling_24h_sentiment_normalized"] = normalize_0_1(chart_data["rolling_24h_sentiment_mean"])
            st.line_chart(chart_data[["price_normalized", "rolling_24h_sentiment_normalized"]])
            st.caption("Both series are normalized to 0-1 so their timing and direction are comparable.")

with section_5:
    st.header("Which candidates get most X attention?")
    if topic_posts_df is None or final_df is None:
        st.info("Missing public/local topic-tagged X features or final dataset.")
    else:
        attention = attention_table(topic_posts_df, final_df)
        if selected_candidate and not candidate_has_x_posts(topic_posts_df, selected_candidate):
            show_no_x_data_message()
        attention_top = attention.head(10)
        st.dataframe(attention, use_container_width=True, hide_index=True)
        st.subheader("Top 10 by post count")
        st.bar_chart(attention_top.sort_values("post_count", ascending=False).set_index("candidate")[["post_count"]])
        st.subheader("Top 10 by engagement")
        engagement_top = attention.sort_values("total_engagement", ascending=False).head(10)
        st.bar_chart(engagement_top.set_index("candidate")[["total_engagement"]])
        if "rolling_24h_post_count" in attention.columns:
            st.subheader("Top 10 by rolling 24h post count")
            rolling_top = attention.sort_values("rolling_24h_post_count", ascending=False).head(10)
            st.bar_chart(rolling_top.set_index("candidate")[["rolling_24h_post_count"]])

with section_6:
    st.header("Model performance")
    metrics = parse_result_metrics(results_text)
    if metrics.empty:
        st.info("Missing or empty results/final_model_comparison_all_candidates.md")
    else:
        metrics = metrics.dropna(subset=["macro_f1"]).sort_values(["horizon", "macro_f1"], ascending=[True, False])
        st.subheader("Model comparison by horizon")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        st.subheader("Best model by Macro-F1")
        best = metrics.groupby("horizon", as_index=False).head(1)
        st.dataframe(best, use_container_width=True, hide_index=True)
        chart_horizon = st.selectbox("Horizon for Macro-F1 chart", sorted(metrics["horizon"].dropna().unique()), key="model_chart_horizon")
        chart_metrics = metrics[metrics["horizon"] == chart_horizon].sort_values("macro_f1", ascending=False)
        st.subheader(f"Macro-F1 by model ({chart_horizon})")
        st.bar_chart(chart_metrics.set_index("model")[["macro_f1"]])
        with st.expander("Full model report", expanded=False):
            st.markdown(results_text or "")

with section_7:
    st.header("Example prediction")
    st.write("This prediction is about the selected candidate's Polymarket price movement: Up, Down, or Stable.")
    st.warning("This is a model prediction for Polymarket price movement, not the actual election winner.")
    if final_df is None or not selected_candidate:
        st.info("Missing final dataset or selected candidate.")
    else:
        prediction, latest_row, note = example_prediction(final_df, selected_candidate, selected_horizon)
        st.caption(note)
        if prediction is None:
            st.info("Could not create an example prediction for this selection.")
        else:
            st.metric("Latest predicted movement", prediction)
        if latest_row is not None:
            key_features = [
                "timestamp",
                "candidate_price",
                "candidate_rank",
                "gap_to_market_leader",
                "price_share",
                "rolling_24h_sentiment_mean",
                "rolling_24h_post_count",
                "num_posts",
                "avg_sentiment_score",
            ]
            available = [column for column in key_features if column in latest_row.index]
            if not candidate_has_x_posts(topic_posts_df, selected_candidate) or not candidate_has_nonzero_sentiment(pd.DataFrame([latest_row]), ["rolling_24h_sentiment_mean", "rolling_24h_post_count", "num_posts", "avg_sentiment_score"]):
                show_no_x_data_message()
            st.subheader("Key latest features")
            st.dataframe(clean_feature_display(latest_row, available), use_container_width=True, hide_index=True)



