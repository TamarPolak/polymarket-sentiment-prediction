"""Rule-based Israeli political topic tagging for all-candidates X posts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from sentiment import clean_text


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "x_posts_with_sentiment_all_candidates.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "x_posts_with_sentiment_topics_all_candidates.csv"

TOPIC_KEYWORDS = {
    "security_war": [
        "security", "war", "idf", "army", "terror", "hamas", "hezbollah", "iran", "gaza",
        "ביטחון", "מלחמה", "צה ל", "צהל", "צבא", "טרור", "חמאס", "חיזבאללה", "איראן", "עזה",
    ],
    "hostages_gaza": [
        "hostages", "hostage", "gaza deal", "deal now", "bring them home",
        "חטופים", "חטוף", "חטופה", "עסקה", "עסקה עכשיו", "להחזיר את החטופים", "עזה",
    ],
    "economy_cost_of_living": [
        "economy", "inflation", "prices", "cost of living", "budget", "tax", "housing",
        "כלכלה", "יוקר המחיה", "מחירים", "תקציב", "מסים", "מיסים", "דיור", "משכנתא",
    ],
    "religion_state": [
        "religion and state", "religious", "secular", "haredi", "draft exemption", "sabbath",
        "דת ומדינה", "מדינת הלכה", "חרדים", "דתיים", "חילונים", "גיוס", "פטור מגיוס", "שבת",
    ],
    "internal_security_crime": [
        "crime", "police", "violence", "internal security", "national security", "ben gvir",
        "פשיעה", "משטרה", "אלימות", "ביטחון פנים", "בטחון פנים", "בן גביר", "פרוטקשן",
    ],
    "judicial_reform_governance": [
        "judicial reform", "supreme court", "court", "judiciary", "governance", "democracy", "dictatorship",
        "רפורמה משפטית", "מערכת המשפט", "בית המשפט", "בגץ", "בג צ", "משילות", "דמוקרטיה", "דיקטטורה",
    ],
    "elections_polls": [
        "election", "elections", "poll", "polls", "mandates", "vote", "campaign", "prime minister",
        "בחירות", "סקר", "סקרים", "מנדטים", "קלפי", "קמפיין", "ראש הממשלה", "ראש ממשלה",
    ],
    "coalition_opposition_government": [
        "coalition", "opposition", "government", "cabinet", "minister", "knesset",
        "קואליציה", "אופוזיציה", "ממשלה", "קבינט", "שר", "שרים", "כנסת", "מנדטים",
    ],
    "foreign_relations": [
        "foreign relations", "biden", "trump", "usa", "united states", "eu", "saudi", "iran", "diplomacy",
        "יחסי חוץ", "ביידן", "טראמפ", "ארצות הברית", "אמריקה", "אירופה", "סעודיה", "איראן", "דיפלומטיה",
    ],
    "corruption_public_trust": [
        "corruption", "trial", "bribery", "trust", "public trust", "scandal", "liar", "lies",
        "שחיתות", "משפט", "שוחד", "אמון הציבור", "אמון", "שערורייה", "מושחת", "שקר", "שקרים",
    ],
}

TOPIC_COLUMNS = [f"topic_{topic}" for topic in TOPIC_KEYWORDS]
OUTPUT_COLUMNS = [
    "timestamp", "text", "candidate", "query_group", "lang", "likes", "reposts", "comments", "post_id",
    "sentiment_score", "sentiment_label", *TOPIC_COLUMNS, "primary_topic", "topics_matched",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tag all-candidates X posts with Israeli political topics.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Input sentiment CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output topic-tagged CSV path.")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def display_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)


def contains_term(cleaned_text: str, term: str) -> bool:
    cleaned_term = clean_text(term)
    if not cleaned_term:
        return False
    return f" {cleaned_term} " in f" {cleaned_text} "


def match_topics(text: str) -> list[str]:
    cleaned = clean_text(text)
    matches = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(contains_term(cleaned, keyword) for keyword in keywords):
            matches.append(topic)
    return matches[:10]


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    print(f"Input path: {display_path(input_path)}")
    print(f"Output path: {display_path(output_path)}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    required_columns = [
        "timestamp", "text", "candidate", "query_group", "lang", "likes", "reposts", "comments", "post_id",
        "sentiment_score", "sentiment_label",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    topic_matches = df["text"].fillna("").apply(match_topics)
    for topic in TOPIC_KEYWORDS:
        df[f"topic_{topic}"] = topic_matches.apply(lambda matches, topic=topic: int(topic in matches))
    df["primary_topic"] = topic_matches.apply(lambda matches: matches[0] if matches else "none")
    df["topics_matched"] = topic_matches.apply(lambda matches: "|".join(matches))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[OUTPUT_COLUMNS].to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Rows tagged: {len(df)}")
    print("Primary topic distribution:")
    print(df["primary_topic"].value_counts().to_string())
    print(f"Saved topic-tagged posts to {display_path(output_path)}")


if __name__ == "__main__":
    main()
