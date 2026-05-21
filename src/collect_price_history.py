import requests
import pandas as pd
from pathlib import Path


PRICE_HISTORY_URL = "https://clob.polymarket.com/prices-history"

TOKENS = {
    "Benjamin Netanyahu": "20006732765674855733524007935991362439352594042415161039767275461950712817548",
    "Naftali Bennett": "58945936829392654397399777214757608864525173988558750256164021442231118224154",
}


def fetch_price_history(candidate_name: str, token_id: str, interval: str = "1w", fidelity: int = 60):
    """
    Fetch price history for a Polymarket outcome token.

    interval options: max, 1w, 1d, 6h, 1h
    fidelity: resolution in minutes
    """
    params = {
        "market": token_id,
        "interval": interval,
        "fidelity": fidelity,
    }

    response = requests.get(PRICE_HISTORY_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    history = data.get("history", [])

    rows = []

    for point in history:
        rows.append({
            "candidate": candidate_name,
            "token_id": token_id,
            "timestamp": pd.to_datetime(point.get("t"), unit="s"),
            "price": point.get("p"),
        })

    return rows


def main():
    all_rows = []

    for candidate_name, token_id in TOKENS.items():
        print(f"Fetching price history for {candidate_name}...")
        rows = fetch_price_history(candidate_name, token_id)
        print(f"Found {len(rows)} price points for {candidate_name}")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    output_path = Path("data/raw/polymarket_price_history.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved price history to {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()