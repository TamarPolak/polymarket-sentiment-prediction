"""Dry-run-first X/Twitter collector for all active Israeli PM candidates.

This script is intentionally safe by default. DRY_RUN=True means it prints the
collection plan and exits before making any X API calls. Real collection can be
enabled later by explicitly changing DRY_RUN to False after reviewing the plan.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from candidates import get_active_candidates


X_RECENT_SEARCH_URL = "https://api.x.com/2/tweets/search/recent"
OUTPUT_PATH = Path("data/raw/x_posts_all_candidates.csv")

DRY_RUN = True
APPEND_EXISTING = True
TARGET_TOTAL_UNIQUE_POSTS = 3500
MAX_POSTS_TOTAL = 3500
MAX_POSTS_PER_QUERY = 100
MAX_POSTS_PER_CANDIDATE = 200
USE_HEBREW_LANGUAGE_FILTER = False
ESTIMATED_COST_PER_POST_USD = 0.005
REQUEST_SLEEP_SECONDS = 1.0

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

HEBREW_CONTEXT_TERMS = [
    "בחירות",
    '"ראש הממשלה"',
    "ממשלה",
    "פוליטיקה",
    "סקר",
    "מנדטים",
    "ישראל",
]
ENGLISH_CONTEXT_TERMS = [
    "Israel",
    "election",
    '"prime minister"',
    "politics",
    "poll",
]


def quote_if_needed(term: str) -> str:
    if term.startswith('"') and term.endswith('"'):
        return term
    if " " in term or "-" in term:
        return f'"{term}"'
    return term


def build_or_terms(terms: list[str], max_terms: int = 4) -> str:
    cleaned_terms = []
    for term in terms:
        if not term or term in cleaned_terms:
            continue
        cleaned_terms.append(term)
    return " OR ".join(quote_if_needed(term) for term in cleaned_terms[:max_terms])


def build_context_terms(terms: list[str]) -> str:
    return " OR ".join(terms)


def slugify(value: str) -> str:
    return (
        value.lower()
        .replace("'", "")
        .replace("’", "")
        .replace("`", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


def build_candidate_queries() -> list[dict[str, str]]:
    queries = []
    for candidate in get_active_candidates():
        candidate_name = candidate["canonical_name"]
        group_prefix = slugify(candidate_name)

        hebrew_terms = candidate.get("hebrew_aliases", [])
        english_terms = candidate.get("english_aliases", [])

        if hebrew_terms:
            query = (
                f"({build_or_terms(hebrew_terms)}) "
                f"({build_context_terms(HEBREW_CONTEXT_TERMS)}) "
                "-is:retweet"
            )
            if USE_HEBREW_LANGUAGE_FILTER:
                query = f"{query} lang:he"
            queries.append(
                {
                    "query_group": f"{group_prefix}_hebrew",
                    "candidate": candidate_name,
                    "language_group": "hebrew",
                    "query": query,
                }
            )

        if english_terms:
            query = (
                f"({build_or_terms(english_terms)}) "
                f"({build_context_terms(ENGLISH_CONTEXT_TERMS)}) "
                "-is:retweet"
            )
            queries.append(
                {
                    "query_group": f"{group_prefix}_english",
                    "candidate": candidate_name,
                    "language_group": "english",
                    "query": query,
                }
            )

    return queries


def load_existing_posts() -> pd.DataFrame:
    if not APPEND_EXISTING or not OUTPUT_PATH.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    existing = pd.read_csv(OUTPUT_PATH)
    for column in OUTPUT_COLUMNS:
        if column not in existing.columns:
            existing[column] = None
    existing = existing[OUTPUT_COLUMNS]
    return existing.drop_duplicates(subset=["post_id"], keep="last")


def print_collection_plan(queries: list[dict[str, str]], existing_count: int) -> int:
    remaining_to_target = max(TARGET_TOTAL_UNIQUE_POSTS - existing_count, 0)
    max_new_posts = min(MAX_POSTS_TOTAL, remaining_to_target)
    estimated_cost = max_new_posts * ESTIMATED_COST_PER_POST_USD

    print("All-candidates X/Twitter collection plan")
    print(f"DRY_RUN: {DRY_RUN}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Append existing: {APPEND_EXISTING}")
    print(f"Existing unique posts: {existing_count}")
    print(f"Target total unique posts: {TARGET_TOTAL_UNIQUE_POSTS}")
    print(f"Max posts total this run: {MAX_POSTS_TOTAL}")
    print(f"Max posts per query: {MAX_POSTS_PER_QUERY}")
    print(f"Max posts per candidate: {MAX_POSTS_PER_CANDIDATE}")
    print(f"Use Hebrew language filter: {USE_HEBREW_LANGUAGE_FILTER}")
    print(f"Active candidates: {len(get_active_candidates())}")
    print(f"Planned query groups: {len(queries)}")
    print(f"Remaining posts to target: {remaining_to_target}")
    print(f"Estimated maximum new posts this run: {max_new_posts}")
    print(f"Estimated maximum cost this run: ${estimated_cost:.2f}")
    print("Bearer token source: X_BEARER_TOKEN environment variable only; token is never printed.")
    print("Queries:")
    for query in queries:
        print(f"- {query['query_group']} [{query['candidate']}]: {query['query']}")

    return max_new_posts


def get_bearer_token() -> str:
    token = os.getenv("X_BEARER_TOKEN")
    if not token:
        raise RuntimeError("Missing X_BEARER_TOKEN environment variable.")
    return token


def fetch_query_posts(query_group: dict[str, str], bearer_token: str, max_results: int) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {bearer_token}"}
    params = {
        "query": query_group["query"],
        "max_results": min(max_results, MAX_POSTS_PER_QUERY),
        "tweet.fields": "created_at,lang,public_metrics,source",
    }
    response = requests.get(X_RECENT_SEARCH_URL, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.json().get("data", [])


def tweet_to_row(tweet: dict[str, Any], query_group: dict[str, str]) -> dict[str, Any]:
    metrics = tweet.get("public_metrics") or {}
    return {
        "timestamp": tweet.get("created_at"),
        "text": tweet.get("text"),
        "candidate": query_group["candidate"],
        "likes": metrics.get("like_count", 0),
        "reposts": metrics.get("retweet_count", 0),
        "comments": metrics.get("reply_count", 0),
        "post_id": tweet.get("id"),
        "query_group": query_group["query_group"],
        "lang": tweet.get("lang"),
        "replies": metrics.get("reply_count", 0),
        "quotes": metrics.get("quote_count", 0),
        "source": tweet.get("source") or "x_recent_search",
    }


def save_posts(existing: pd.DataFrame, new_rows: list[dict[str, Any]]) -> pd.DataFrame:
    new_posts = pd.DataFrame(new_rows, columns=OUTPUT_COLUMNS)
    combined = pd.concat([existing, new_posts], ignore_index=True)
    combined = combined.drop_duplicates(subset=["post_id"], keep="last")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    return combined


def collect_posts(queries: list[dict[str, str]], existing: pd.DataFrame, max_new_posts: int) -> None:
    bearer_token = get_bearer_token()
    existing_post_ids = set(existing["post_id"].dropna().astype(str))
    candidate_counts: dict[str, int] = {}
    new_rows = []
    failed_query_groups = []

    for query_group in queries:
        if len(new_rows) >= max_new_posts:
            break

        candidate = query_group["candidate"]
        if candidate_counts.get(candidate, 0) >= MAX_POSTS_PER_CANDIDATE:
            continue

        remaining_total = max_new_posts - len(new_rows)
        remaining_candidate = MAX_POSTS_PER_CANDIDATE - candidate_counts.get(candidate, 0)
        request_limit = min(MAX_POSTS_PER_QUERY, remaining_total, remaining_candidate)
        if request_limit <= 0:
            continue

        print(f"Collecting query group: {query_group['query_group']} | max_results={request_limit}")
        try:
            tweets = fetch_query_posts(query_group, bearer_token, request_limit)
        except requests.HTTPError as exc:
            failed_query_groups.append(query_group["query_group"])
            print(f"WARNING: query failed: {query_group['query_group']} HTTP {exc.response.status_code}")
            print(exc.response.text)
            continue

        for tweet in tweets:
            post_id = str(tweet.get("id"))
            if not post_id or post_id in existing_post_ids:
                continue
            row = tweet_to_row(tweet, query_group)
            new_rows.append(row)
            existing_post_ids.add(post_id)
            candidate_counts[candidate] = candidate_counts.get(candidate, 0) + 1

            if len(new_rows) >= max_new_posts or candidate_counts[candidate] >= MAX_POSTS_PER_CANDIDATE:
                break

        time.sleep(REQUEST_SLEEP_SECONDS)

    combined = save_posts(existing, new_rows)
    print(f"Total unique posts saved: {len(combined)}")
    print(f"New posts added in this run: {len(new_rows)}")
    print("Query group distribution:")
    print(combined["query_group"].value_counts())
    print(f"Failed query groups: {failed_query_groups}")
    print("Reminder: set DRY_RUN=True after collection to avoid accidental extra X API usage.")


def main() -> None:
    queries = build_candidate_queries()
    existing = load_existing_posts()
    max_new_posts = print_collection_plan(queries, len(existing))

    if DRY_RUN:
        print("DRY_RUN=True; stopping before any X API calls.")
        return

    print("REAL API COLLECTION MODE - this may use X credits.")
    if max_new_posts <= 0:
        print("Target already reached; no collection needed.")
        return

    collect_posts(queries, existing, max_new_posts)


if __name__ == "__main__":
    main()
