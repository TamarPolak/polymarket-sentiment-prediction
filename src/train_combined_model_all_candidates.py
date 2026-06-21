"""Train all-candidates multiclass market-only and market+sentiment models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "final" / "market_sentiment_dataset_all_candidates.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "final_model_comparison_all_candidates.md"
TARGET_COLUMN = "target_multiclass"
EXPECTED_CLASSES = ["Up", "Down", "Stable"]
EXPECTED_HORIZONS = ["1h", "2h", "24h"]

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
    "total_sentiment_score",
    "num_posts",
    "num_positive_posts",
    "num_negative_posts",
    "num_neutral_posts",
    "total_likes",
    "total_reposts",
    "total_comments",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all-candidates market and market+sentiment models.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Input final dataset CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output markdown report path.")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def display_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)


def markdown_series(series: pd.Series, value_name: str = "count") -> str:
    lines = [f"| class | {value_name} |", "|---|---:|"]
    for index, value in series.items():
        lines.append(f"| {index} | {int(value)} |")
    return "\n".join(lines)


def markdown_matrix(matrix_df: pd.DataFrame) -> str:
    header = "| true\\predicted | " + " | ".join(str(column) for column in matrix_df.columns) + " |"
    separator = "|---" + "|---:" * len(matrix_df.columns) + "|"
    rows = [header, separator]
    for index, row in matrix_df.iterrows():
        rows.append("| " + str(index) + " | " + " | ".join(str(int(value)) for value in row.values) + " |")
    return "\n".join(rows)


def prepare_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    X = df[feature_columns + ["candidate"]].copy()
    for column in feature_columns:
        X[column] = pd.to_numeric(X[column], errors="coerce").fillna(0)
    X = pd.get_dummies(X, columns=["candidate"], drop_first=False)
    return X.astype(float)


def align_train_test_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return X_train.align(X_test, join="left", axis=1, fill_value=0)


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


def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    report = classification_report(y_test, predictions, labels=EXPECTED_CLASSES, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=EXPECTED_CLASSES)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "report": report,
        "matrix": pd.DataFrame(matrix, index=EXPECTED_CLASSES, columns=EXPECTED_CLASSES),
    }


def add_model_section(lines: list[str], model_name: str, result: dict) -> None:
    lines.append(f"### {model_name}")
    lines.append("")
    lines.append(f"- Accuracy: {result['accuracy']:.4f}")
    lines.append(f"- Macro-F1: {result['macro_f1']:.4f}")
    lines.append("")
    lines.append("Classification report:")
    lines.append("")
    lines.append("```text")
    lines.append(result["report"])
    lines.append("```")
    lines.append("")
    lines.append("Confusion matrix rows=true, columns=predicted:")
    lines.append("")
    lines.append(markdown_matrix(result["matrix"]))
    lines.append("")


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = ["timestamp", "candidate", "horizon", TARGET_COLUMN, *MARKET_FEATURES, *SENTIMENT_FEATURES]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    print(f"Input path: {display_path(input_path)}")
    print(f"Output path: {display_path(output_path)}")
    print(f"Target used: {TARGET_COLUMN} = Up / Down / Stable")

    if not input_path.exists():
        raise FileNotFoundError(f"Final dataset not found: {input_path}")

    df = pd.read_csv(input_path)
    validate_columns(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp", "candidate", "horizon", TARGET_COLUMN]).sort_values("timestamp").copy()

    lines = [
        "# Final Model Comparison - All Candidates",
        "",
        "Target: `target_multiclass = Up / Down / Stable`.",
        "",
        "Time split: first 75% of rows by timestamp for training, last 25% for testing.",
        "",
        f"Number of modeled candidates: {df['candidate'].nunique()}",
        "",
        "X/Twitter sentiment features are predictive features. Labels are derived from future Polymarket price movements.",
        "",
    ]

    available_horizons = set(df["horizon"].dropna().unique())
    for missing_horizon in [horizon for horizon in EXPECTED_HORIZONS if horizon not in available_horizons]:
        lines.append(f"## Horizon: {missing_horizon}")
        lines.append("")
        lines.append(f"NOTE: Horizon `{missing_horizon}` is not available in the final dataset.")
        lines.append("")

    horizons_evaluated = [horizon for horizon in EXPECTED_HORIZONS if horizon in available_horizons]
    print(f"Horizons evaluated: {', '.join(horizons_evaluated)}")

    for horizon in horizons_evaluated:
        horizon_df = df[df["horizon"] == horizon].sort_values("timestamp").copy()
        class_counts = horizon_df[TARGET_COLUMN].value_counts().reindex(EXPECTED_CLASSES, fill_value=0)

        lines.append(f"## Horizon: {horizon}")
        lines.append("")
        lines.append(f"Rows: {len(horizon_df)}")
        lines.append(f"Modeled candidates: {horizon_df['candidate'].nunique()}")
        lines.append("")
        lines.append("Class distribution:")
        lines.append("")
        lines.append(markdown_series(class_counts))
        lines.append("")

        split_index = int(len(horizon_df) * 0.75)
        if split_index <= 0 or split_index >= len(horizon_df):
            lines.append(f"NOTE: Skipping horizon `{horizon}` because there are not enough rows for a 75/25 time split.")
            lines.append("")
            continue

        train_df = horizon_df.iloc[:split_index].copy()
        test_df = horizon_df.iloc[split_index:].copy()
        y_train = train_df[TARGET_COLUMN]
        y_test = test_df[TARGET_COLUMN]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            lines.append(
                f"NOTE: Skipping horizon `{horizon}` because train/test split has insufficient class variety. "
                f"Train classes: {sorted(y_train.unique())}. Test classes: {sorted(y_test.unique())}."
            )
            lines.append("")
            continue

        X_market_train = prepare_features(train_df, MARKET_FEATURES)
        X_market_test = prepare_features(test_df, MARKET_FEATURES)
        X_market_train, X_market_test = align_train_test_columns(X_market_train, X_market_test)

        X_combined_train = prepare_features(train_df, MARKET_FEATURES + SENTIMENT_FEATURES)
        X_combined_test = prepare_features(test_df, MARKET_FEATURES + SENTIMENT_FEATURES)
        X_combined_train, X_combined_test = align_train_test_columns(X_combined_train, X_combined_test)

        lines.append(f"Train rows: {len(train_df)}")
        lines.append(f"Test rows: {len(test_df)}")
        lines.append("")

        results: dict[str, dict] = {}
        dummy_result = evaluate_model(build_models()["Dummy Baseline"], X_market_train, y_train, X_market_test, y_test)
        results["Dummy Baseline"] = dummy_result
        add_model_section(lines, "Dummy Baseline", dummy_result)

        for base_name, X_train, X_test in [
            ("Market-only", X_market_train, X_market_test),
            ("Market + Sentiment", X_combined_train, X_combined_test),
        ]:
            models = build_models()
            for model_name in ["Logistic Regression", "Random Forest"]:
                full_name = f"{base_name} {model_name}"
                result = evaluate_model(models[model_name], X_train, y_train, X_test, y_test)
                results[full_name] = result
                add_model_section(lines, full_name, result)

        best_model_name, best_result = max(results.items(), key=lambda item: item[1]["macro_f1"])
        best_market_only = max(
            (result for name, result in results.items() if name.startswith("Market-only")),
            key=lambda result: result["macro_f1"],
        )
        best_combined = max(
            (result for name, result in results.items() if name.startswith("Market + Sentiment")),
            key=lambda result: result["macro_f1"],
        )
        sentiment_delta = best_combined["macro_f1"] - best_market_only["macro_f1"]

        lines.append("### Horizon Summary")
        lines.append("")
        lines.append(f"- Best model by Macro-F1: {best_model_name} ({best_result['macro_f1']:.4f})")
        lines.append(f"- Best market-only Macro-F1: {best_market_only['macro_f1']:.4f}")
        lines.append(f"- Best market + sentiment Macro-F1: {best_combined['macro_f1']:.4f}")
        lines.append(f"- Sentiment improved over market-only: {'yes' if sentiment_delta > 0 else 'no'} ({sentiment_delta:+.4f})")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved all-candidates model comparison to {display_path(output_path)}")


if __name__ == "__main__":
    main()
