"""Collect Polymarket price history for all active Israeli PM candidates.

This all-candidates collector uses the exact Polymarket event slug instead of a
broad text search. The event contains multiple binary candidate markets; for each
candidate market, this script collects the candidate's Yes price history.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from candidates import get_active_candidates


GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
PRICE_HISTORY_URL = "https://clob.polymarket.com/prices-history"
EVENT_SLUG = "who-will-be-the-next-prime-minister-of-israel-after-the-next-election"
REQUIRED_TITLE_TEXT = "Prime Minister of Israel"
REQUIRED_SLUG_TEXT = "next-prime-minister-of-israel"
OUTPUT_PATH = Path("data/raw/polymarket_price_history_all_candidates.csv")
PRICE_INTERVAL = "1w"
PRICE_FIDELITY_MINUTES = 60
SOURCE = "polymarket_clob_prices_history"


def parse_maybe_json(value: Any) -> list[Any]:
    """Gamma fields may arrive as lists or JSON-encoded strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def normalize(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("'", "")
        .replace("’", "")
        .replace("`", "")
        .replace('"', "")
        .replace("-", " ")
        .replace("?", "")
    )


def candidate_aliases(candidate: dict) -> set[str]:
    aliases = [
        candidate["canonical_name"],
        *candidate.get("hebrew_aliases", []),
        *candidate.get("english_aliases", []),
    ]

    if candidate["canonical_name"] == "Gadi Eisenkot":
        aliases.extend(["Gadi Eizenkot", "Gadi Eisenkot", "גדי איזנקוט", "גדי אייזנקוט"])

    return {normalize(alias) for alias in aliases}


def fetch_event_by_exact_slug() -> dict:
    response = requests.get(GAMMA_EVENTS_URL, params={"slug": EVENT_SLUG}, timeout=30)
    response.raise_for_status()
    events = response.json()

    if isinstance(events, dict):
        event = events
    elif isinstance(events, list):
        matching_events = [event for event in events if event.get("slug") == EVENT_SLUG]
        if not matching_events:
            available_slugs = [event.get("slug") for event in events[:5]]
            raise RuntimeError(
                "Exact Polymarket event slug was not returned. "
                f"Expected: {EVENT_SLUG}. Got slugs: {available_slugs}"
            )
        event = matching_events[0]
    else:
        raise RuntimeError(f"Unexpected Gamma events response type: {type(events).__name__}")

    validate_event(event)
    return event


def validate_event(event: dict) -> None:
    title = event.get("title") or event.get("question") or event.get("name") or ""
    slug = event.get("slug") or ""

    if REQUIRED_TITLE_TEXT not in title:
        raise RuntimeError(
            "Refusing to collect prices from unrelated Polymarket event. "
            f"Event title must contain '{REQUIRED_TITLE_TEXT}'. Got: {title}"
        )

    if REQUIRED_SLUG_TEXT not in slug:
        raise RuntimeError(
            "Refusing to collect prices from unrelated Polymarket event. "
            f"Event slug must contain '{REQUIRED_SLUG_TEXT}'. Got: {slug}"
        )


def get_event_markets(event: dict) -> list[dict]:
    markets = event.get("markets") or event.get("childMarkets") or []
    if isinstance(markets, str):
        markets = parse_maybe_json(markets)
    if not isinstance(markets, list):
        markets = []
    return [market for market in markets if isinstance(market, dict)]


def get_market_title(market: dict) -> str:
    return str(market.get("question") or market.get("title") or market.get("name") or "")


def match_candidate_to_market(candidate: dict, market: dict) -> bool:
    market_text = normalize(" ".join([get_market_title(market), str(market.get("slug", ""))]))
    aliases = candidate_aliases(candidate)
    return any(alias and alias in market_text for alias in aliases)


def extract_yes_token_id(market: dict) -> str | None:
    token_objects = parse_maybe_json(market.get("tokens"))
    for token in token_objects:
        if not isinstance(token, dict):
            continue
        outcome = normalize(token.get("outcome") or token.get("name") or "")
        token_id = token.get("token_id") or token.get("tokenId") or token.get("id")
        if outcome == "yes" and token_id:
            return str(token_id)

    outcomes = parse_maybe_json(market.get("outcomes"))
    token_ids = (
        parse_maybe_json(market.get("clobTokenIds"))
        or parse_maybe_json(market.get("clobTokenIDs"))
        or parse_maybe_json(market.get("clob_token_ids"))
    )
    if outcomes and token_ids and len(outcomes) == len(token_ids):
        for outcome, token_id in zip(outcomes, token_ids):
            if normalize(outcome) == "yes":
                return str(token_id)

    return None


def build_candidate_market_map(event_markets: list[dict], candidates: list[dict]) -> dict[str, dict]:
    candidate_market_map: dict[str, dict] = {}

    for candidate in candidates:
        candidate_name = candidate["canonical_name"]
        matched_markets = [market for market in event_markets if match_candidate_to_market(candidate, market)]

        if not matched_markets:
            print(f"WARNING: Candidate market not found in Polymarket event: {candidate_name}")
            continue

        if len(matched_markets) > 1:
            titles = [get_market_title(market) for market in matched_markets]
            print(f"WARNING: Multiple markets matched {candidate_name}; using first match: {titles}")

        selected_market = matched_markets[0]
        yes_token_id = extract_yes_token_id(selected_market)
        if not yes_token_id:
            print(f"WARNING: Yes token not found for candidate market: {candidate_name}")
            continue

        candidate_market_map[candidate_name] = {
            "market": selected_market,
            "yes_token_id": yes_token_id,
        }

    return candidate_market_map


def fetch_price_history(candidate_name: str, token_id: str, event: dict, market: dict) -> list[dict]:
    params = {
        "market": token_id,
        "interval": PRICE_INTERVAL,
        "fidelity": PRICE_FIDELITY_MINUTES,
    }
    response = requests.get(PRICE_HISTORY_URL, params=params, timeout=30)
    response.raise_for_status()
    history = response.json().get("history", [])

    rows = []
    for point in history:
        rows.append(
            {
                "timestamp": pd.to_datetime(point.get("t"), unit="s", utc=True),
                "candidate": candidate_name,
                "price": point.get("p"),
                "volume": point.get("v") or point.get("volume") or market.get("volume"),
                "market_slug": market.get("slug") or event.get("slug"),
                "market_id": market.get("id"),
                "event_slug": event.get("slug"),
                "event_id": event.get("id"),
                "token_id": token_id,
                "source": SOURCE,
            }
        )
    return rows


def main() -> None:
    candidates = get_active_candidates()
    print(f"Active candidates configured: {len(candidates)}")
    print(f"Fetching exact Polymarket event slug: {EVENT_SLUG}")

    event = fetch_event_by_exact_slug()
    event_title = event.get("title") or event.get("question") or event.get("name")
    event_slug = event.get("slug")
    print(f"Selected event title: {event_title}")
    print(f"Selected event slug: {event_slug}")

    event_markets = get_event_markets(event)
    if not event_markets:
        raise RuntimeError("The selected Polymarket event does not contain candidate markets.")

    candidate_market_map = build_candidate_market_map(event_markets, candidates)
    found_candidate_names = sorted(candidate_market_map.keys())
    print(f"Candidate markets/outcomes found: {len(candidate_market_map)}")
    print(f"Candidate names found: {found_candidate_names}")

    if not candidate_market_map:
        raise RuntimeError("No active configured candidates matched the selected Polymarket event markets.")

    all_rows = []
    for candidate_name, match in candidate_market_map.items():
        market = match["market"]
        token_id = match["yes_token_id"]
        print(f"Fetching Yes price history for {candidate_name}: {get_market_title(market)}")
        try:
            rows = fetch_price_history(candidate_name, token_id, event, market)
        except requests.HTTPError as exc:
            print(f"WARNING: Failed to fetch {candidate_name}: HTTP {exc.response.status_code}")
            continue
        print(f"Found {len(rows)} price points for {candidate_name}")
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No all-candidate price history rows were collected.")

    output = pd.DataFrame(all_rows)
    output = output.sort_values(["timestamp", "candidate"]).drop_duplicates(
        subset=["timestamp", "candidate", "token_id"], keep="last"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved all-candidate price history to {OUTPUT_PATH}")
    print(f"Rows saved: {len(output)}")
    print(output.groupby("candidate").size().sort_values(ascending=False))


if __name__ == "__main__":
    main()
