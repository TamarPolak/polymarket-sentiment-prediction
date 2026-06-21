"""Build generalized all-candidate Polymarket market features.

This all-candidates version avoids the old 2-candidate opponent_price/price_gap
features. It instead creates rank, leader-gap, price-share, and market-level
features that work for any number of active Polymarket outcomes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/raw/polymarket_price_history_all_candidates.csv")
OUTPUT_PATH = Path("data/processed/market_features_all_candidates.csv")
HORIZONS = {
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "24h": pd.Timedelta(hours=24),
}
MOVEMENT_THRESHOLD = 0.001


def label_price_change(price_change: float) -> str:
    if price_change > MOVEMENT_THRESHOLD:
        return "Up"
    if price_change < -MOVEMENT_THRESHOLD:
        return "Down"
    return "Stable"


def get_price_at(series: pd.Series, target_timestamp: pd.Timestamp) -> float:
    if target_timestamp in series.index:
        return series.loc[target_timestamp]
    return np.nan


def build_hourly_price_matrix(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path)
    required_columns = {"timestamp", "candidate", "price"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.floor("h")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["timestamp", "candidate", "price"])

    if df.empty:
        raise ValueError(f"No valid price rows found in {input_path}")

    hourly = (
        df.groupby(["timestamp", "candidate"], as_index=False)["price"]
        .mean()
        .sort_values(["timestamp", "candidate"])
    )
    return hourly.pivot(index="timestamp", columns="candidate", values="price").sort_index()


def build_features(price_matrix: pd.DataFrame) -> pd.DataFrame:
    market_total_price = price_matrix.sum(axis=1, min_count=1)
    market_leader_price = price_matrix.max(axis=1)
    market_num_candidates = price_matrix.notna().sum(axis=1)
    candidate_ranks = price_matrix.rank(axis=1, ascending=False, method="min")

    sorted_prices = np.sort(price_matrix.fillna(-np.inf).to_numpy(), axis=1)[:, ::-1]
    top_1 = sorted_prices[:, 0]
    top_2 = sorted_prices[:, 1] if price_matrix.shape[1] > 1 else np.full(len(price_matrix), np.nan)
    top_2_gap = pd.Series(
        np.where(np.isfinite(top_2), top_1 - top_2, np.nan),
        index=price_matrix.index,
    )

    rows = []
    for candidate in price_matrix.columns:
        series = price_matrix[candidate]
        for timestamp, candidate_price in series.dropna().items():
            previous_1h = get_price_at(series, timestamp - pd.Timedelta(hours=1))
            previous_2h = get_price_at(series, timestamp - pd.Timedelta(hours=2))
            previous_24h = get_price_at(series, timestamp - pd.Timedelta(hours=24))

            for horizon_name, horizon_delta in HORIZONS.items():
                future_price = get_price_at(series, timestamp + horizon_delta)
                if pd.isna(future_price):
                    continue

                price_change = future_price - candidate_price
                total_price = market_total_price.loc[timestamp]
                leader_price = market_leader_price.loc[timestamp]

                rows.append(
                    {
                        "timestamp": timestamp,
                        "candidate": candidate,
                        "horizon": horizon_name,
                        "candidate_price": candidate_price,
                        "candidate_change_1h": candidate_price - previous_1h if pd.notna(previous_1h) else np.nan,
                        "candidate_change_2h": candidate_price - previous_2h if pd.notna(previous_2h) else np.nan,
                        "candidate_change_24h": candidate_price - previous_24h if pd.notna(previous_24h) else np.nan,
                        "market_leader_price": leader_price,
                        "gap_to_market_leader": leader_price - candidate_price,
                        "candidate_rank": candidate_ranks.loc[timestamp, candidate],
                        "price_share": candidate_price / total_price if total_price and pd.notna(total_price) else np.nan,
                        "top_2_gap": top_2_gap.loc[timestamp],
                        "market_num_candidates": market_num_candidates.loc[timestamp],
                        "future_price": future_price,
                        "price_change": price_change,
                        "target_multiclass": label_price_change(price_change),
                    }
                )

    features = pd.DataFrame(rows)
    if features.empty:
        return features

    return features.sort_values(["timestamp", "candidate", "horizon"]).reset_index(drop=True)


def main() -> None:
    print(f"Input path: {INPUT_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Horizons: {', '.join(HORIZONS.keys())}")

    price_matrix = build_hourly_price_matrix(INPUT_PATH)
    print(f"Hourly timestamps: {len(price_matrix)}")
    print(f"Candidates in price file: {len(price_matrix.columns)}")

    features = build_features(price_matrix)
    if features.empty:
        raise RuntimeError("No feature rows were created. Check price history coverage and horizons.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved all-candidate market features to {OUTPUT_PATH}")
    print(f"Rows created: {len(features)}")
    print(features.groupby(["horizon", "target_multiclass"]).size())


if __name__ == "__main__":
    main()
