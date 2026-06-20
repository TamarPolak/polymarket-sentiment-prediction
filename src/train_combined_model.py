from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "final" / "market_sentiment_dataset.csv"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "results" / "final_model_comparison.md"
TARGET_COLUMN = "target_multiclass"
EXPECTED_CLASSES = ["Up", "Down", "Stable"]
EXPECTED_HORIZONS = ["1h", "2h", "24h"]

MARKET_FEATURES = [
    "candidate_price",
    "opponent_price",
    "price_gap",
    "candidate_change_1h",
    "opponent_change_1h",
    "gap_change_1h",
    "candidate_change_2h",
    "opponent_change_2h",
    "price_gap_abs",
]

SENTIMENT_FEATURES = [
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


def prepare_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    X = df[feature_columns + ["candidate"]].copy()
    for column in feature_columns:
        X[column] = pd.to_numeric(X[column], errors="coerce").fillna(0)
    return pd.get_dummies(X, columns=["candidate"], drop_first=False)


def markdown_series(series: pd.Series, value_name: str = "count") -> str:
    lines = [f"| class | {value_name} |", "|---|---:|"]
    for index, value in series.items():
        lines.append(f"| {index} | {value} |")
    return "\n".join(lines)


def markdown_matrix(matrix_df: pd.DataFrame) -> str:
    header = "| true\\predicted | " + " | ".join(str(column) for column in matrix_df.columns) + " |"
    separator = "|---" + "|---:" * len(matrix_df.columns) + "|"
    rows = [header, separator]
    for index, row in matrix_df.iterrows():
        rows.append("| " + str(index) + " | " + " | ".join(str(int(value)) for value in row.values) + " |")
    return "\n".join(rows)


def build_models() -> dict[str, object]:
    return {
        "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
    }


def align_train_test_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
    return X_train, X_test


def add_model_results(lines: list[str], model_name: str, model, X_train, y_train, X_test, y_test) -> None:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    report = classification_report(y_test, predictions, labels=EXPECTED_CLASSES, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=EXPECTED_CLASSES)
    matrix_df = pd.DataFrame(matrix, index=EXPECTED_CLASSES, columns=EXPECTED_CLASSES)

    lines.append(f"### {model_name}")
    lines.append("")
    lines.append(f"- Accuracy: {accuracy:.4f}")
    lines.append(f"- Macro-F1: {macro_f1:.4f}")
    lines.append("")
    lines.append("Classification report:")
    lines.append("")
    lines.append("```text")
    lines.append(report)
    lines.append("```")
    lines.append("")
    lines.append("Confusion matrix rows=true, columns=predicted:")
    lines.append("")
    lines.append(markdown_matrix(matrix_df))
    lines.append("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final multiclass market and market+sentiment models.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Final dataset CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_RESULTS_PATH), help="Output markdown results path.")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def display_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    print(f"Input path: {display_path(input_path)}")
    print(f"Output path: {display_path(output_path)}")
    print(f"Target used: {TARGET_COLUMN} = Up / Down / Stable")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Final dataset not found: {input_path}. Run python src/build_market_sentiment_dataset.py first."
        )

    df = pd.read_csv(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", TARGET_COLUMN, "horizon"]).sort_values("timestamp").copy()

    lines = [
        "# Final Model Comparison",
        "",
        "Target: `target_multiclass = Up / Down / Stable`.",
        "",
        "Time-based train/test split is used for every horizon. The first 75% of rows by timestamp are used for training and the final 25% are used for testing.",
        "",
        "X/Twitter sentiment features are predictive features. Labels are derived from future Polymarket price movements.",
        "",
    ]

    available_horizons = set(df["horizon"].dropna().unique())
    for missing_horizon in [h for h in EXPECTED_HORIZONS if h not in available_horizons]:
        lines.append(f"## Horizon: {missing_horizon}")
        lines.append("")
        lines.append(
            f"NOTE: Horizon `{missing_horizon}` is not available in the final dataset. "
            "For 10m, this usually means the current Polymarket price resolution is not frequent enough, so it was skipped gracefully during market feature creation."
        )
        lines.append("")

    horizons_evaluated = sorted(df["horizon"].dropna().unique())
    print(f"Horizons evaluated: {', '.join(horizons_evaluated)}")

    for horizon in horizons_evaluated:
        horizon_df = df[df["horizon"] == horizon].sort_values("timestamp").copy()
        class_counts = horizon_df[TARGET_COLUMN].value_counts().reindex(EXPECTED_CLASSES, fill_value=0)

        lines.append(f"## Horizon: {horizon}")
        lines.append("")
        lines.append("Class distribution:")
        lines.append("")
        lines.append(markdown_series(class_counts))
        lines.append("")

        missing_classes = [label for label, count in class_counts.items() if count == 0]
        if missing_classes:
            lines.append(
                f"NOTE: Skipping model training for horizon `{horizon}` because these classes are missing: {', '.join(missing_classes)}."
            )
            lines.append("")
            continue

        split_index = int(len(horizon_df) * 0.75)
        if split_index <= 0 or split_index >= len(horizon_df):
            lines.append(f"NOTE: Skipping horizon `{horizon}` because there are not enough rows for time split.")
            lines.append("")
            continue

        train_df = horizon_df.iloc[:split_index].copy()
        test_df = horizon_df.iloc[split_index:].copy()

        train_classes = set(train_df[TARGET_COLUMN])
        test_classes = set(test_df[TARGET_COLUMN])
        if len(train_classes) < 2 or len(test_classes) < 2:
            lines.append(
                f"NOTE: Skipping horizon `{horizon}` because train/test split has insufficient class variety. "
                f"Train classes: {sorted(train_classes)}. Test classes: {sorted(test_classes)}."
            )
            lines.append("")
            continue

        y_train = train_df[TARGET_COLUMN]
        y_test = test_df[TARGET_COLUMN]

        X_market_train = prepare_features(train_df, MARKET_FEATURES)
        X_market_test = prepare_features(test_df, MARKET_FEATURES)
        X_market_train, X_market_test = align_train_test_columns(X_market_train, X_market_test)

        X_combined_train = prepare_features(train_df, MARKET_FEATURES + SENTIMENT_FEATURES)
        X_combined_test = prepare_features(test_df, MARKET_FEATURES + SENTIMENT_FEATURES)
        X_combined_train, X_combined_test = align_train_test_columns(X_combined_train, X_combined_test)

        lines.append(f"Train rows: {len(train_df)}")
        lines.append(f"Test rows: {len(test_df)}")
        lines.append("")

        dummy = DummyClassifier(strategy="most_frequent")
        add_model_results(lines, "Dummy Baseline", dummy, X_market_train, y_train, X_market_test, y_test)

        market_models = build_models()
        for model_name in ["Logistic Regression", "Random Forest"]:
            add_model_results(
                lines,
                f"Market-only {model_name}",
                market_models[model_name],
                X_market_train,
                y_train,
                X_market_test,
                y_test,
            )

        combined_models = build_models()
        for model_name in ["Logistic Regression", "Random Forest"]:
            add_model_results(
                lines,
                f"Market + Sentiment {model_name}",
                combined_models[model_name],
                X_combined_train,
                y_train,
                X_combined_test,
                y_test,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved final model comparison to {display_path(output_path)}")


if __name__ == "__main__":
    main()



