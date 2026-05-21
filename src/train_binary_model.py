import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_PATH = "data/raw/polymarket_price_history.csv"


def create_binary_target(current_price, future_price, threshold=0.001):
    """
    Create a binary classification label:
    Move / Stable

    If the absolute price change is larger than the threshold -> Move
    Otherwise -> Stable
    """
    change = future_price - current_price

    if abs(change) > threshold:
        return "Move"
    else:
        return "Stable"


def build_ml_dataset():
    """
    Build a supervised learning dataset from Polymarket price history.
    Each row represents one hourly time window.
    The model predicts whether Netanyahu's price will move in the next hour.
    """
    df = pd.read_csv(INPUT_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["price"] = df["price"].astype(float)

    # Align timestamps to hourly windows
    df["timestamp"] = df["timestamp"].dt.floor("h")

    # Average multiple points in the same hour
    df = (
        df.groupby(["timestamp", "candidate"], as_index=False)
        .agg(price=("price", "mean"))
    )

    # Convert to wide format
    wide_df = df.pivot_table(
        index="timestamp",
        columns="candidate",
        values="price",
        aggfunc="mean"
    ).reset_index()

    wide_df = wide_df.sort_values("timestamp")

    wide_df = wide_df.rename(columns={
        "Benjamin Netanyahu": "netanyahu_price",
        "Naftali Bennett": "bennett_price"
    })

    wide_df = wide_df.dropna(subset=["netanyahu_price", "bennett_price"])

    # Features
    wide_df["price_gap"] = wide_df["netanyahu_price"] - wide_df["bennett_price"]
    wide_df["netanyahu_change_1h"] = wide_df["netanyahu_price"].diff()
    wide_df["bennett_change_1h"] = wide_df["bennett_price"].diff()
    wide_df["gap_change_1h"] = wide_df["price_gap"].diff()

    # More useful features for movement prediction
    wide_df["netanyahu_change_2h"] = wide_df["netanyahu_price"].diff(2)
    wide_df["bennett_change_2h"] = wide_df["bennett_price"].diff(2)
    wide_df["price_gap_abs"] = wide_df["price_gap"].abs()

    # Future price one hour ahead
    wide_df["netanyahu_future_price_1h"] = wide_df["netanyahu_price"].shift(-1)

    # Binary target
    wide_df["target"] = wide_df.apply(
        lambda row: create_binary_target(
            row["netanyahu_price"],
            row["netanyahu_future_price_1h"]
        )
        if pd.notnull(row["netanyahu_future_price_1h"])
        else None,
        axis=1
    )

    wide_df = wide_df.dropna()

    return wide_df


def print_model_results(model_name, y_test, predictions):
    print(f"\n========== {model_name} RESULTS ==========")

    print(f"\n{model_name} Accuracy:", accuracy_score(y_test, predictions))
    print(f"{model_name} F1 macro:", f1_score(y_test, predictions, average="macro"))

    print(f"\n{model_name} classification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print(f"\n{model_name} confusion matrix:")
    print(confusion_matrix(y_test, predictions))


def train_models(dataset):
    features = [
        "netanyahu_price",
        "bennett_price",
        "price_gap",
        "netanyahu_change_1h",
        "bennett_change_1h",
        "gap_change_1h",
        "netanyahu_change_2h",
        "bennett_change_2h",
        "price_gap_abs"
    ]

    X = dataset[features]
    y = dataset["target"]

    print("========== DATA CHECK ==========")
    print("\nDataset size:", len(dataset))

    print("\nTarget distribution:")
    print(y.value_counts())

    print("\nTarget distribution percentage:")
    print(y.value_counts(normalize=True))

    if y.nunique() < 2:
        print("\nERROR: The dataset contains only one class.")
        return None

    split_index = int(len(dataset) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("\nTrain size:", len(X_train))
    print("Test size:", len(X_test))

    if y_train.nunique() < 2:
        print("\nERROR: The training set contains only one class.")
        return None

    # Dummy baseline
    dummy_model = DummyClassifier(strategy="most_frequent")
    dummy_model.fit(X_train, y_train)
    dummy_predictions = dummy_model.predict(X_test)
    print_model_results("DUMMY BASELINE", y_test, dummy_predictions)

    # Logistic Regression
    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    logistic_model.fit(X_train, y_train)
    logistic_predictions = logistic_model.predict(X_test)
    print_model_results("LOGISTIC REGRESSION", y_test, logistic_predictions)

    # Random Forest
    random_forest_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )

    random_forest_model.fit(X_train, y_train)
    random_forest_predictions = random_forest_model.predict(X_test)
    print_model_results("RANDOM FOREST", y_test, random_forest_predictions)

    return {
        "dummy": dummy_model,
        "logistic_regression": logistic_model,
        "random_forest": random_forest_model
    }


def main():
    dataset = build_ml_dataset()

    print("First rows of binary ML dataset:")
    print(dataset.head())

    train_models(dataset)


if __name__ == "__main__":
    main()