import json
import requests


EVENT_SLUG = "who-will-be-the-next-prime-minister-of-israel-after-the-next-election"
GAMMA_EVENT_BY_SLUG_URL = f"https://gamma-api.polymarket.com/events/slug/{EVENT_SLUG}"


def main():
    response = requests.get(GAMMA_EVENT_BY_SLUG_URL, timeout=30)
    response.raise_for_status()

    event = response.json()

    print("=" * 80)
    print("Event title:", event.get("title"))
    print("Event slug:", event.get("slug"))
    print("Active:", event.get("active"))
    print("Closed:", event.get("closed"))
    print("Volume:", event.get("volume"))
    print("Liquidity:", event.get("liquidity"))

    markets = event.get("markets", [])
    print("\nNumber of inner markets:", len(markets))

    for index, market in enumerate(markets, start=1):
        print("\n" + "-" * 80)
        print(f"Inner market #{index}")
        print("Question:", market.get("question"))
        print("ID:", market.get("id"))
        print("Slug:", market.get("slug"))
        print("Active:", market.get("active"))
        print("Closed:", market.get("closed"))
        print("Outcomes:", market.get("outcomes"))
        print("Outcome Prices:", market.get("outcomePrices"))
        print("CLOB Token IDs:", market.get("clobTokenIds"))

    with open("data/raw/israel_pm_event.json", "w", encoding="utf-8") as file:
        json.dump(event, file, ensure_ascii=False, indent=2)

    print("\nSaved full event data to data/raw/israel_pm_event.json")


if __name__ == "__main__":
    main()