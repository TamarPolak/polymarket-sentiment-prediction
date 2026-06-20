from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "polymarket_price_history.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "market_features.csv"

MOVEMENT_THRESHOLD = 0.001
CANDIDATES = ["Benjamin Netanyahu", "Naftali Bennett"]
HORIZONS = {
    "10m": pd.Timedelta(minutes=10),
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "24h": pd.Timedelta(hours=24),
}


def classify_price_change(price_change: float) -> str:
    if price_change > MOVEMENT_THRESHOLD:
        return "Up"
    if price_change < -MOVEMENT_THRESHOLD:
        return "Down"
    return "Stable"


def infer_median_resolution(df: pd.DataFrame) -> pd.Timedelta | None:
    diffs = []
    for _, group in df.sort_values("timestamp").groupby("candidate"):
        candidate_diffs = group["timestamp"].diff().dropna()
        if not candidate_diffs.empty:
            diffs.extend(candidate_diffs.tolist())
    if not diffs:
        return None
    return pd.Series(diffs).median()


def nearest_price_lookup(wide: pd.DataFrame, target_times: pd.Series, column: str, tolerance: pd.Timedelta) -> pd.Series:
    lookup = wide[["timestamp", column]].dropna().sort_values("timestamp")
    requests = pd.DataFrame({"target_time": target_times})
    merged = pd.merge_asof(
        requests.sort_values("target_time"),
        lookup,
        left_on="target_time",
        right_on="timestamp",
        direction="nearest",
        tolerance=tolerance,
    )
    merged = merged.sort_index()
    return merged[column]


def build_features_for_horizon(wide: pd.DataFrame, horizon_name: str, horizon_delta: pd.Timedelta) -> pd.DataFrame:
    rows = []
    tolerance = pd.Timedelta(minutes=35) if horizon_delta < pd.Timedelta(hours=2) else pd.Timedelta(minutes=75)

    for candidate in CANDIDATES:
        opponent = next(name for name in CANDIDATES if name != candidate)
        current = wide[["timestamp", candidate, opponent]].copy()
        current = current.rename(columns={candidate: "candidate_price", opponent: "opponent_price"})
        current["candidate"] = candidate
        current["horizon"] = horizon_name
        current["price_gap"] = current["candidate_price"] - current["opponent_price"]
        current["price_gap_abs"] = current["price_gap"].abs()

        current["candidate_change_1h"] = current["candidate_price"] - nearest_price_lookup(
            wide, current["timestamp"] - pd.Timedelta(hours=1), candidate, pd.Timedelta(minutes=35)
        )
        current["opponent_change_1h"] = current["opponent_price"] - nearest_price_lookup(
            wide, current["timestamp"] - pd.Timedelta(hours=1), opponent, pd.Timedelta(minutes=35)
        )
        previous_gap_1h = nearest_price_lookup(
            wide.assign(price_gap=wide[candidate] - wide[opponent]),
            current["timestamp"] - pd.Timedelta(hours=1),
            "price_gap",
            pd.Timedelta(minutes=35),
        )
        current["gap_change_1h"] = current["price_gap"] - previous_gap_1h

        current["candidate_change_2h"] = current["candidate_price"] - nearest_price_lookup(
            wide, current["timestamp"] - pd.Timedelta(hours=2), candidate, pd.Timedelta(minutes=75)
        )
        current["opponent_change_2h"] = current["opponent_price"] - nearest_price_lookup(
            wide, current["timestamp"] - pd.Timedelta(hours=2), opponent, pd.Timedelta(minutes=75)
        )

        current["future_price"] = nearest_price_lookup(
            wide, current["timestamp"] + horizon_delta, candidate, tolerance
        )
        current["price_change"] = current["future_price"] - current["candidate_price"]
        current["target_multiclass"] = current["price_change"].apply(
            lambda value: classify_price_change(value) if pd.notna(value) else pd.NA
        )
        rows.append(current)

    result = pd.concat(rows, ignore_index=True)
    required = [
        "candidate_price",
        "opponent_price",
        "candidate_change_1h",
        "opponent_change_1h",
        "gap_change_1h",
        "candidate_change_2h",
        "opponent_change_2h",
        "future_price",
        "target_multiclass",
    ]
    return result.dropna(subset=required).copy()


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    required_columns = ["timestamp", "candidate", "price"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["candidate"].isin(CANDIDATES)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["timestamp", "candidate", "price"])

    median_resolution = infer_median_resolution(df)
    print(f"Median Polymarket resolution: {median_resolution}")

    # Align near-identical candidate timestamps to the available data resolution.
    # Current Polymarket data is usually hourly, while future 10m data can use 10-minute buckets.
    if median_resolution is not None and median_resolution <= pd.Timedelta(minutes=10):
        df["timestamp"] = df["timestamp"].dt.floor("10min")
    else:
        df["timestamp"] = df["timestamp"].dt.floor("h")

    df = df.groupby(["timestamp", "candidate"], as_index=False).agg(price=("price", "mean"))
    wide = df.pivot_table(index="timestamp", columns="candidate", values="price", aggfunc="mean").reset_index()
    wide = wide.sort_values("timestamp").dropna(subset=CANDIDATES).copy()

    all_features = []
    for horizon_name, horizon_delta in HORIZONS.items():
        if horizon_name == "10m" and median_resolution is not None and median_resolution > pd.Timedelta(minutes=10):
            print(
                "WARNING: Skipping 10m horizon because current Polymarket price resolution "
                f"is {median_resolution}, which is not frequent enough for 10-minute labels."
            )
            continue
        horizon_features = build_features_for_horizon(wide, horizon_name, horizon_delta)
        if horizon_features.empty:
            print(f"WARNING: No valid rows created for horizon {horizon_name}.")
            continue
        all_features.append(horizon_features)

    if not all_features:
        raise ValueError("No valid market feature rows were created for any horizon.")

    output = pd.concat(all_features, ignore_index=True)
    output = output[
        [
            "timestamp",
            "candidate",
            "horizon",
            "candidate_price",
            "opponent_price",
            "price_gap",
            "candidate_change_1h",
            "opponent_change_1h",
            "gap_change_1h",
            "candidate_change_2h",
            "opponent_change_2h",
            "price_gap_abs",
            "future_price",
            "price_change",
            "target_multiclass",
        ]
    ].sort_values(["horizon", "timestamp", "candidate"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Saved market features to {OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Rows created: {len(output)}")
    print(output.groupby("horizon")["target_multiclass"].value_counts().to_string())


if __name__ == "__main__":
    main()

