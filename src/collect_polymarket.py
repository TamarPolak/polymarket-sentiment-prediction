import requests
import pandas as pd
from datetime import datetime
from pathlib import Path


GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"


def fetch_markets(search_query="Israel Prime Minister"):
    """
    Fetch markets from Polymarket Gamma API by search query.
    """
    params = {
        "search": search_query,
        "limit": 20
    }

    response = requests.get(GAMMA_API_URL, params=params, timeout=20)
    response.raise_for_status()

    return response.json()


def save_markets_to_csv(markets, output_path):
    """
    Save selected market data into a CSV file.
    """
    rows = []

    for market in markets:
        rows.append({
            "id": market.get("id"),
            "question": market.get("question"),
            "slug": market.get("slug"),
            "active": market.get("active"),
            "closed": market.get("closed"),
            "volume": market.get("volume"),
            "liquidity": market.get("liquidity"),
            "end_date": market.get("endDate"),
            "created_at": market.get("createdAt"),
            "updated_at": market.get("updatedAt"),
            "collected_at": datetime.now()
        })

    df = pd.DataFrame(rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved {len(df)} markets to {output_path}")


def main():
    markets = fetch_markets("Who will be the next Prime Minister of Israel")
    save_markets_to_csv(markets, "data/raw/polymarket_markets.csv")


if __name__ == "__main__":
    main()