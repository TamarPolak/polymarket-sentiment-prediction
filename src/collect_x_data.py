from __future__ import annotations

import csv
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "x_posts_real_limited.csv"

# Safety defaults. Change DRY_RUN to False only after reviewing the printed plan.
DRY_RUN = True
MAX_POSTS_TOTAL = 500
MAX_POSTS_PER_QUERY = 100
REQUEST_SLEEP_SECONDS = 1.0
RECENT_SEARCH_URL = "https://api.x.com/2/tweets/search/recent"

TWEET_FIELDS = ["created_at", "text", "lang", "public_metrics", "author_id"]

QUERY_GROUPS = [
    {
        "name": "netanyahu_hebrew",
        "candidate": "Benjamin Netanyahu",
        "query": '(נתניהו OR ביבי) (בחירות OR "ראש הממשלה" OR ממשלה OR ליכוד OR "רק ביבי" OR "רק לא ביבי") lang:iw -is:retweet',
    },
    {
        "name": "netanyahu_english",
        "candidate": "Benjamin Netanyahu",
        "query": '(Netanyahu OR Bibi) ("prime minister" OR election OR Israel OR Likud) lang:en -is:retweet',
    },
    {
        "name": "bennett_hebrew",
        "candidate": "Naftali Bennett",
        "query": '(בנט OR "נפתלי בנט") (בחירות OR "ראש הממשלה" OR ממשלה OR אלטרנטיבה) lang:iw -is:retweet',
    },
    {
        "name": "bennett_english",
        "candidate": "Naftali Bennett",
        "query": '("Naftali Bennett" OR Bennett) ("prime minister" OR election OR Israel) lang:en -is:retweet',
    },
    {
        "name": "general_israeli_election_hebrew",
        "candidate": "General",
        "query": '("בחירות בישראל" OR בחירות OR "ראש הממשלה") (ימין OR שמאל OR מרכז OR קואליציה OR אופוזיציה) lang:iw -is:retweet',
    },
    {
        "name": "general_israeli_election_english",
        "candidate": "General",
        "query": '("Israeli election" OR "Israel election" OR "next prime minister") (Netanyahu OR Bennett OR Likud) lang:en -is:retweet',
    },
]

def safe_print(value: object = "") -> None:
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="backslashreplace").decode(encoding))


OUTPUT_COLUMNS = [
    "timestamp",
    "text",
    "candidate",
    "likes",
    "reposts",
    "comments",
    "post_id",
    "query_group",
    "lang",
    "replies",
    "quotes",
    "source",
]


def print_collection_plan() -> None:
    safe_print("X Recent Search collection plan")
    safe_print(f"DRY_RUN: {DRY_RUN}")
    safe_print(f"MAX_POSTS_TOTAL: {MAX_POSTS_TOTAL}")
    safe_print(f"MAX_POSTS_PER_QUERY: {MAX_POSTS_PER_QUERY}")
    safe_print(f"Endpoint: {RECENT_SEARCH_URL}")
    safe_print("Queries:")
    for index, group in enumerate(QUERY_GROUPS, start=1):
        safe_print(f"{index}. {group['name']} [{group['candidate']}]")
        safe_print(f"   {group['query']}")
    safe_print(f"Output path: {OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")


def get_bearer_token() -> str:
    token = os.getenv("X_BEARER_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "Missing X_BEARER_TOKEN environment variable. "
            "Set it in PowerShell with: $env:X_BEARER_TOKEN='YOUR_TOKEN'"
        )
    return token


def build_url(query: str, max_results: int, next_token: str | None = None) -> str:
    params = {
        "query": query,
        "max_results": str(max(10, min(max_results, 100))),
        "tweet.fields": ",".join(TWEET_FIELDS),
    }
    if next_token:
        params["next_token"] = next_token
    return f"{RECENT_SEARCH_URL}?{urllib.parse.urlencode(params)}"


def request_recent_search(token: str, query: str, max_results: int, next_token: str | None = None) -> dict[str, Any]:
    url = build_url(query, max_results, next_token)
    request = urllib.request.Request(url)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("User-Agent", "polymarket-sentiment-prediction/1.0")

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"X API request failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"X API request failed: {exc}") from exc


def normalize_post(post: dict[str, Any], query_group: dict[str, str]) -> dict[str, Any]:
    metrics = post.get("public_metrics") or {}
    reply_count = int(metrics.get("reply_count") or 0)
    return {
        "timestamp": post.get("created_at", ""),
        "text": post.get("text", ""),
        "candidate": query_group["candidate"],
        "likes": int(metrics.get("like_count") or 0),
        "reposts": int(metrics.get("retweet_count") or 0),
        "comments": reply_count,
        "post_id": post.get("id", ""),
        "query_group": query_group["name"],
        "lang": post.get("lang", ""),
        "replies": reply_count,
        "quotes": int(metrics.get("quote_count") or 0),
        "source": "x_recent_search",
    }


def save_rows(rows: list[dict[str, Any]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def collect_posts() -> list[dict[str, Any]]:
    token = get_bearer_token()
    rows: list[dict[str, Any]] = []
    seen_post_ids: set[str] = set()

    for query_group in QUERY_GROUPS:
        if len(rows) >= MAX_POSTS_TOTAL:
            break

        query_count = 0
        next_token = None
        safe_print(f"Collecting query group: {query_group['name']}")

        while len(rows) < MAX_POSTS_TOTAL and query_count < MAX_POSTS_PER_QUERY:
            remaining_total = MAX_POSTS_TOTAL - len(rows)
            remaining_query = MAX_POSTS_PER_QUERY - query_count
            page_size = min(100, remaining_total, remaining_query)
            if page_size < 10:
                break

            payload = request_recent_search(token, query_group["query"], page_size, next_token)
            posts = payload.get("data") or []

            for post in posts:
                post_id = str(post.get("id", ""))
                if not post_id or post_id in seen_post_ids:
                    continue
                seen_post_ids.add(post_id)
                rows.append(normalize_post(post, query_group))
                query_count += 1
                if len(rows) >= MAX_POSTS_TOTAL or query_count >= MAX_POSTS_PER_QUERY:
                    break

            meta = payload.get("meta") or {}
            next_token = meta.get("next_token")
            safe_print(f"  collected so far: total={len(rows)}, query={query_count}")

            if not next_token or not posts:
                break
            time.sleep(REQUEST_SLEEP_SECONDS)

    return rows


def main() -> None:
    print_collection_plan()

    if DRY_RUN:
        safe_print("DRY_RUN=True, stopping before any API calls.")
        safe_print("Set DRY_RUN=False in src/collect_x_data.py only after reviewing the plan and X spending limits.")
        return

    rows = collect_posts()
    save_rows(rows)
    safe_print(f"Saved {len(rows)} unique posts to {OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()

