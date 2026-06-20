from __future__ import annotations

import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "x_posts_real_limited.csv"

# Safety defaults. Keep DRY_RUN=True until the collection plan is reviewed.
DRY_RUN = True
MAX_POSTS_TOTAL = 1000
MAX_POSTS_PER_QUERY = 200
APPEND_EXISTING = True
TARGET_TOTAL_UNIQUE_POSTS = 1000
STOP_ON_FIRST_SUCCESSFUL_QUERY = False
TEST_ONE_QUERY_ONLY = False
TEST_QUERY_GROUP_NAME = None
TEST_MAX_POSTS_TOTAL = 30
TEST_MAX_POSTS_PER_QUERY = 30
ESTIMATED_COST_PER_POST_USD = 0.005
REQUEST_SLEEP_SECONDS = 1.0
RECENT_SEARCH_URL = "https://api.x.com/2/tweets/search/recent"

# Hebrew keywords already indicate Hebrew content. Leave this False unless language filtering is needed.
USE_HEBREW_LANGUAGE_FILTER = False

TWEET_FIELDS = ["created_at", "text", "lang", "public_metrics", "author_id"]

QUERY_GROUPS = [
    {
        "name": "netanyahu_hebrew_broad",
        "candidate": "Benjamin Netanyahu",
        "is_hebrew": True,
        "query": '(נתניהו OR ביבי OR "רק ביבי" OR "רק לא ביבי" OR "ביבי הביתה" OR "משפט נתניהו") -is:retweet',
    },
    {
        "name": "bennett_hebrew_broad",
        "candidate": "Naftali Bennett",
        "is_hebrew": True,
        "query": '(בנט OR "נפתלי בנט" OR "ימינה" OR "ממשלת בנט") -is:retweet',
    },
    {
        "name": "israeli_election_hebrew_broad",
        "candidate": "General",
        "is_hebrew": True,
        "query": '(בחירות OR "בחירות בישראל" OR "ראש הממשלה" OR ממשלה OR קואליציה OR אופוזיציה OR ימין OR שמאל OR ליכוד) -is:retweet',
    },
    {
        "name": "israeli_politics_hebrew_broad",
        "candidate": "General",
        "is_hebrew": True,
        "query": '("אתה הראש אתה אשם" OR "בחירות עכשיו" OR "מדינת הלכה" OR "יהודה ושומרון" OR איו"ש OR מתנחלים OR מתיישבים OR ליברמן OR לפיד OR גנץ OR "בן גביר" OR סמוטריץ) -is:retweet',
    },
    {
        "name": "netanyahu_hebrew_focused",
        "candidate": "Benjamin Netanyahu",
        "is_hebrew": True,
        "query": '(נתניהו OR ביבי) (בחירות OR "ראש הממשלה" OR ממשלה OR ליכוד OR "רק ביבי" OR "רק לא ביבי") lang:he -is:retweet',
    },
    {
        "name": "bennett_hebrew_focused",
        "candidate": "Naftali Bennett",
        "is_hebrew": True,
        "query": '(בנט OR "נפתלי בנט") (בחירות OR "ראש הממשלה" OR ממשלה OR אלטרנטיבה) lang:he -is:retweet',
    },
    {
        "name": "general_israeli_election_hebrew_focused",
        "candidate": "General",
        "is_hebrew": True,
        "query": '("בחירות בישראל" OR בחירות OR "ראש הממשלה") (ימין OR שמאל OR מרכז OR קואליציה OR אופוזיציה) lang:he -is:retweet',
    },
    {
        "name": "netanyahu_english",
        "candidate": "Benjamin Netanyahu",
        "is_hebrew": False,
        "query": '(Netanyahu OR Bibi) ("prime minister" OR election OR Israel OR Likud) lang:en -is:retweet',
    },
    {
        "name": "bennett_english",
        "candidate": "Naftali Bennett",
        "is_hebrew": False,
        "query": '("Naftali Bennett" OR Bennett) ("prime minister" OR election OR Israel) lang:en -is:retweet',
    },
    {
        "name": "general_israeli_election_english",
        "candidate": "General",
        "is_hebrew": False,
        "query": '("Israeli election" OR "Israel election" OR "next prime minister") (Netanyahu OR Bennett OR Likud) lang:en -is:retweet',
    },
]

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


class QueryRequestError(Exception):
    def __init__(self, status_code: int | None, body: str, original: Exception | None = None) -> None:
        self.status_code = status_code
        self.body = body
        self.original = original
        super().__init__(body)


def safe_print(value: object = "") -> None:
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="backslashreplace").decode(encoding))


def active_limits(existing_unique_count: int = 0) -> tuple[int, int]:
    if TEST_ONE_QUERY_ONLY:
        return TEST_MAX_POSTS_TOTAL, TEST_MAX_POSTS_PER_QUERY
    remaining_to_target = max(TARGET_TOTAL_UNIQUE_POSTS - existing_unique_count, 0)
    return min(MAX_POSTS_TOTAL, remaining_to_target), MAX_POSTS_PER_QUERY


def active_queries() -> list[dict[str, Any]]:
    groups = []
    for group in QUERY_GROUPS:
        if TEST_ONE_QUERY_ONLY and TEST_QUERY_GROUP_NAME and group["name"] != TEST_QUERY_GROUP_NAME:
            continue
        query = group["query"]
        if group.get("is_hebrew") and not USE_HEBREW_LANGUAGE_FILTER:
            query = query.replace(" lang:he", "")
        next_group = dict(group)
        next_group["query"] = query
        groups.append(next_group)
    if TEST_ONE_QUERY_ONLY and TEST_QUERY_GROUP_NAME and not groups:
        valid_names = ", ".join(group["name"] for group in QUERY_GROUPS)
        raise ValueError(f"Unknown TEST_QUERY_GROUP_NAME={TEST_QUERY_GROUP_NAME!r}. Valid names: {valid_names}")
    return groups


def load_existing_rows() -> list[dict[str, Any]]:
    if not APPEND_EXISTING or not OUTPUT_PATH.exists():
        return []
    with OUTPUT_PATH.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = []
        seen = set()
        for row in reader:
            post_id = str(row.get("post_id", "")).strip()
            if not post_id or post_id in seen:
                continue
            seen.add(post_id)
            rows.append({column: row.get(column, "") for column in OUTPUT_COLUMNS})
    return rows


def print_collection_plan(existing_unique_count: int) -> None:
    max_total, max_per_query = active_limits(existing_unique_count)
    queries = active_queries()
    remaining_to_target = max(TARGET_TOTAL_UNIQUE_POSTS - existing_unique_count, 0)
    planned_new_posts = min(max_total, remaining_to_target)
    estimated_additional_cost = planned_new_posts * ESTIMATED_COST_PER_POST_USD

    safe_print("X Recent Search collection plan")
    safe_print(f"DRY_RUN: {DRY_RUN}")
    safe_print(f"APPEND_EXISTING: {APPEND_EXISTING}")
    safe_print(f"Existing unique posts: {existing_unique_count}")
    safe_print(f"Target total unique posts: {TARGET_TOTAL_UNIQUE_POSTS}")
    safe_print(f"Remaining posts to collect: {remaining_to_target}")
    safe_print(f"TEST_ONE_QUERY_ONLY: {TEST_ONE_QUERY_ONLY}")
    safe_print(f"TEST_QUERY_GROUP_NAME: {TEST_QUERY_GROUP_NAME}")
    safe_print(f"STOP_ON_FIRST_SUCCESSFUL_QUERY: {STOP_ON_FIRST_SUCCESSFUL_QUERY}")
    safe_print(f"USE_HEBREW_LANGUAGE_FILTER: {USE_HEBREW_LANGUAGE_FILTER}")
    safe_print(f"Planned queries: {len(queries)}")
    safe_print(f"Max new posts this run: {max_total}")
    safe_print(f"Max posts per query: {max_per_query}")
    safe_print(f"Estimated additional cost: ${estimated_additional_cost:.2f} at ${ESTIMATED_COST_PER_POST_USD:.3f}/post")
    safe_print(f"Endpoint: {RECENT_SEARCH_URL}")
    safe_print(f"Output path: {OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    safe_print("Queries:")
    for index, group in enumerate(queries, start=1):
        safe_print(f"{index}. {group['name']} [{group['candidate']}]")
        safe_print(f"   {group['query']}")


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
        raise QueryRequestError(exc.code, body, exc) from exc
    except urllib.error.URLError as exc:
        raise QueryRequestError(None, str(exc), exc) from exc


def normalize_post(post: dict[str, Any], query_group: dict[str, Any]) -> dict[str, Any]:
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


def print_query_failure(query_group: dict[str, Any], error: QueryRequestError) -> None:
    safe_print("QUERY FAILED - skipping this query group")
    safe_print(f"Query group: {query_group['name']}")
    safe_print(f"Query text: {query_group['query']}")
    safe_print(f"HTTP status code: {error.status_code if error.status_code is not None else 'N/A'}")
    safe_print(f"Error body: {error.body}")


def collect_posts(existing_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, list[str], float]:
    token = get_bearer_token()
    existing_seen_ids = {str(row.get("post_id", "")).strip() for row in existing_rows if row.get("post_id")}
    max_total_new, max_per_query = active_limits(len(existing_seen_ids))
    rows = list(existing_rows)
    failed_query_groups: list[str] = []
    seen_post_ids: set[str] = set(existing_seen_ids)
    new_posts_added = 0
    api_posts_received = 0

    if max_total_new <= 0:
        safe_print("Target total unique posts already reached. No API calls needed.")
        return rows, new_posts_added, failed_query_groups, 0.0

    for query_group in active_queries():
        if new_posts_added >= max_total_new or len(seen_post_ids) >= TARGET_TOTAL_UNIQUE_POSTS:
            break

        query_count = 0
        next_token = None
        query_had_success = False
        query_returned_posts = False
        safe_print(f"Collecting query group: {query_group['name']}")

        while new_posts_added < max_total_new and query_count < max_per_query and len(seen_post_ids) < TARGET_TOTAL_UNIQUE_POSTS:
            remaining_run = max_total_new - new_posts_added
            remaining_query = max_per_query - query_count
            remaining_target = TARGET_TOTAL_UNIQUE_POSTS - len(seen_post_ids)
            page_size = min(100, remaining_run, remaining_query, remaining_target)
            if page_size < 10:
                break

            try:
                payload = request_recent_search(token, query_group["query"], page_size, next_token)
            except QueryRequestError as error:
                failed_query_groups.append(query_group["name"])
                print_query_failure(query_group, error)
                break

            query_had_success = True
            posts = payload.get("data") or []
            api_posts_received += len(posts)
            if not posts:
                safe_print("No posts returned for this query, trying next query.")

            for post in posts:
                post_id = str(post.get("id", ""))
                if not post_id or post_id in seen_post_ids:
                    continue
                seen_post_ids.add(post_id)
                rows.append(normalize_post(post, query_group))
                query_count += 1
                new_posts_added += 1
                query_returned_posts = True
                if new_posts_added >= max_total_new or query_count >= max_per_query or len(seen_post_ids) >= TARGET_TOTAL_UNIQUE_POSTS:
                    break

            meta = payload.get("meta") or {}
            next_token = meta.get("next_token")
            safe_print(
                f"  collected so far: total_unique={len(seen_post_ids)}, "
                f"new_this_run={new_posts_added}, query_new={query_count}"
            )

            if not next_token or not posts:
                break
            time.sleep(REQUEST_SLEEP_SECONDS)

        if TEST_ONE_QUERY_ONLY and query_returned_posts:
            safe_print("TEST_ONE_QUERY_ONLY=True, stopping after the first query group that returned posts.")
            break

        if STOP_ON_FIRST_SUCCESSFUL_QUERY and query_had_success and query_count > 0:
            safe_print("STOP_ON_FIRST_SUCCESSFUL_QUERY=True, stopping after first successful query with posts.")
            break

    estimated_run_cost = api_posts_received * ESTIMATED_COST_PER_POST_USD
    return rows, new_posts_added, failed_query_groups, estimated_run_cost


def print_final_summary(rows: list[dict[str, Any]], new_posts_added: int, failed_query_groups: list[str], estimated_run_cost: float) -> None:
    safe_print("Final collection summary")
    safe_print(f"Total unique posts saved: {len(rows)}")
    safe_print(f"New posts added in this run: {new_posts_added}")
    safe_print(f"Estimated cost for this run: ${estimated_run_cost:.2f}")
    safe_print("Query group distribution:")
    distribution = Counter(row.get("query_group", "unknown") for row in rows)
    for query_group, count in sorted(distribution.items()):
        safe_print(f"- {query_group}: {count}")

    if failed_query_groups:
        safe_print("Failed query groups:")
        for name in failed_query_groups:
            safe_print(f"- {name}")
    else:
        safe_print("Failed query groups: none")

    safe_print("Reminder: set DRY_RUN=True after collection to avoid accidental extra X API usage.")


def main() -> None:
    existing_rows = load_existing_rows()
    existing_unique_count = len({row.get("post_id") for row in existing_rows if row.get("post_id")})
    print_collection_plan(existing_unique_count)

    if DRY_RUN:
        safe_print("DRY_RUN=True, stopping before any API calls.")
        safe_print("Set DRY_RUN=False only after reviewing the plan, X spending limits, and auto-recharge settings.")
        return

    safe_print("REAL API COLLECTION MODE — this may use X credits.")
    rows, new_posts_added, failed_query_groups, estimated_run_cost = collect_posts(existing_rows)
    save_rows(rows)
    safe_print(f"Saved {len(rows)} unique posts to {OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print_final_summary(rows, new_posts_added, failed_query_groups, estimated_run_cost)


if __name__ == "__main__":
    main()
