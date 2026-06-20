from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "text_posts_sample.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "text_posts_with_sentiment.csv"

# Editable MVP lexicons. Political identity/context terms are features, not sentiment by themselves.
ENTITY_ALIASES = {
    "Netanyahu": ["netanyahu", "bibi", "benjamin netanyahu", "נתניהו", "ביבי", "בנימין נתניהו"],
    "Bennett": ["bennett", "naftali bennett", "בנט", "נפתלי בנט"],
    "Lieberman": ["avigdor lieberman", "lieberman", "ליברמן", "אביגדור ליברמן"],
    "Gantz": ["gantz", "benny gantz", "גנץ", "בני גנץ"],
    "Lapid": ["lapid", "yair lapid", "לפיד", "יאיר לפיד"],
    "Ben Gvir": ["ben gvir", "itamar ben gvir", "בן גביר", "איתמר בן גביר"],
    "Smotrich": ["smotrich", "bezalel smotrich", "סמוטריץ", "בצלאל סמוטריץ"],
    "Deri": ["deri", "aryeh deri", "דרעי", "אריה דרעי"],
    "Eisenkot": ["eisenkot", "gadi eisenkot", "איזנקוט", "גדי איזנקוט"],
}

PARTY_TERMS = {
    "Yisrael Beiteinu": ["yisrael beiteinu", "ישראל ביתנו"],
    "Likud": ["likud", "הליכוד"],
    "Yesh Atid": ["yesh atid", "יש עתיד"],
    "National Unity": ["national unity", "המחנה הממלכתי", "כחול לבן"],
    "Religious Zionism": ["religious zionism", "הציונות הדתית"],
    "Otzma Yehudit": ["otzma yehudit", "עוצמה יהודית"],
    "Shas": ["shas", "שס", "ש\"ס"],
    "United Torah Judaism": ["united torah judaism", "יהדות התורה"],
    "Meretz": ["meretz", "מרצ"],
    "Labor": ["labor", "labour", "העבודה", "מפלגת העבודה"],
    "Raam": ["raam", "רע\"מ", "רעמ"],
    "Hadash-Taal": ["hadash taal", "hadash-taal", "חדש תעל", "חד\"ש תע\"ל"],
}

BLOC_TERMS = {
    "right": ["right", "right wing", "ימין", "גוש הימין"],
    "strong_right": ["strong right", "ימין חזק"],
    "right_wing_government": ["right wing government", "ממשלת ימין", "ממשלת ימין מלא מלא"],
    "left": ["left", "left wing", "שמאל", "גוש השמאל"],
    "left_wing_government": ["left wing government", "ממשלת שמאל"],
    "center": ["center", "centre", "מרכז", "גוש המרכז"],
    "center_right": ["center right", "centre right", "ימין מרכז"],
    "center_left": ["center left", "centre left", "שמאל מרכז"],
    "nationalist_camp": ["nationalist camp", "המחנה הלאומי"],
    "democratic_camp": ["democratic camp", "המחנה הדמוקרטי"],
    "liberal_camp": ["liberal camp", "המחנה הליברלי"],
    "arab": ["arab", "arabs", "ערבים", "מפלגות ערביות", "פלסטינים"],
    "religious": ["religious", "חרדים", "דתיים", "ציונות דתית"],
    "secular": ["secular", "חילונים", "חילוני"],
    "jewish": ["jewish", "יהודי", "יהודית"],
}

IDEOLOGY_CONTEXT_TERMS = {
    "settlements": ["settlements", "התנחלויות", "התיישבות"],
    "settlers": ["settlers", "מתנחלים"],
    "residents": ["residents", "מתיישבים"],
    "judea_samaria": ["judea and samaria", "יהודה ושומרון", "איו\"ש", "איו ש", "יו\"ש", "יו ש"],
    "west_bank": ["west bank", "הגדה המערבית"],
    "sovereignty": ["sovereignty", "ריבונות"],
    "annexation": ["annexation", "סיפוח"],
    "two_state_solution": ["two state solution", "two-state solution", "שתי מדינות"],
    "palestinian_state": ["palestinian state", "מדינה פלסטינית"],
    "terror": ["terror", "טרור"],
    "security": ["security", "ביטחון"],
    "hostages": ["hostages", "hostage", "חטופים", "חטוף", "חטופה"],
    "war": ["war", "מלחמה"],
    "gaza": ["gaza", "עזה"],
    "hamas": ["hamas", "חמאס"],
    "iran": ["iran", "איראן"],
    "hezbollah": ["hezbollah", "חיזבאללה"],
}

TOPIC_KEYWORDS = {
    "elections": ["election", "elections", "vote", "poll", "polls", "בחירות", "סקר", "סקרים", "קלפי"],
    "leadership": ["leader", "leadership", "prime minister", "ראש ממשלה", "מנהיג", "מנהיגות"],
    "government": ["government", "cabinet", "ממשלה", "קבינט"],
    "protest": ["protest", "demonstration", "מחאה", "הפגנה", "מפגינים", "קפלן"],
    "responsibility_blame": ["responsibility", "blame", "fault", "אחריות", "אשם", "אשמה", "מחדל"],
    "security": ["security", "terror", "army", "idf", "ביטחון", "טרור", "צהל", "צה\"ל"],
    "hostages": ["hostages", "hostage", "חטופים", "חטוף", "חטופה", "עסקה", "להחזיר את החטופים"],
    "war": ["war", "gaza", "hamas", "iran", "hezbollah", "מלחמה", "עזה", "חמאס", "איראן", "חיזבאללה"],
    "religion_and_state": ["religion and state", "religious coercion", "מדינת הלכה", "דת ומדינה", "כפייה דתית", "חרדים"],
    "judiciary": ["judiciary", "court", "supreme court", "בגץ", "בג\"ץ", "בית המשפט", "רפורמה משפטית", "מערכת המשפט"],
    "economy": ["economy", "prices", "cost of living", "כלכלה", "יוקר המחיה", "מחירים"],
    "coalition": ["coalition", "opposition", "קואליציה", "אופוזיציה", "מנדטים"],
    "corruption": ["corruption", "trial", "bribery", "שחיתות", "משפט", "שוחד", "תיקים", "מושחת"],
    "jewish_arab_relations": ["jewish arab", "arab jewish", "יהודים ערבים", "ערבים יהודים", "דו קיום"],
    "left_right_polarization": ["left", "right", "שמאל", "ימין", "רק ביבי", "רק לא ביבי"],
    "settlements_sovereignty": ["settlements", "settlers", "judea and samaria", "west bank", "sovereignty", "annexation", "התנחלויות", "מתנחלים", "מתיישבים", "יהודה ושומרון", "איו\"ש", "יו\"ש", "ריבונות", "סיפוח"],
    "palestinian_issue": ["two state solution", "palestinian state", "palestinians", "שתי מדינות", "מדינה פלסטינית", "פלסטינים"],
}

# These are the terms that can affect sentiment. Context/identity terms above do not.
POSITIVE_WORDS = {
    "strong", "stronger", "confident", "clear", "effective", "support", "positive", "praise",
    "practical", "focused", "stable", "responsible", "trust", "trusted", "win", "credible",
    "alternative", "suitable", "fits", "fit", "חזק", "חזקה", "מתחזק", "מוביל", "חיובי", "חיוביות",
    "תמיכה", "אמין", "יציב", "ברור", "אחראי", "ניצחון", "ראוי", "ממלכתי", "תקווה",
    "אלטרנטיבה", "מתאים", "מתאימה", "מתאים מאוד",
}

NEGATIVE_WORDS = {
    "weak", "criticism", "criticized", "bad", "failure", "failed", "blame", "guilty", "corrupt",
    "dangerous", "disappointing", "chaos", "crisis", "lies", "liar", "unsuitable", "not suitable",
    "כישלון", "נכשל", "אשם", "אשמה", "ביקורת", "קשה", "מתקשה", "איבד", "חוסר", "מאכזב",
    "שחיתות", "מסוכן", "כאוס", "משבר", "שקר", "שקרים", "בגידה", "מושחת", "לא מתאים",
    "לא מתאימה", "הביתה", "מחדל", "דיקטטורה",
}

POLITICAL_PHRASES = {
    "ימין חזק": {"sentiment": 1, "intensity": 2, "topics": ["left_right_polarization"], "context": ["strong_right"]},
    "רק ביבי": {"sentiment": 2, "intensity": 3, "topics": ["leadership", "left_right_polarization"], "context": []},
    "רק לא ביבי": {"sentiment": -2, "intensity": 3, "topics": ["leadership", "left_right_polarization"], "context": []},
    "ביבי מלך ישראל": {"sentiment": 3, "intensity": 4, "topics": ["leadership"], "context": []},
    "ביבי הביתה": {"sentiment": -3, "intensity": 4, "topics": ["protest", "leadership"], "context": []},
    "אתה הראש אתה אשם": {"sentiment": -3, "intensity": 4, "topics": ["responsibility_blame", "protest"], "context": []},
    "בחירות עכשיו": {"sentiment": -1, "intensity": 4, "topics": ["elections", "protest"], "context": []},
    "ממשלת ימין מלא מלא": {"sentiment": 0, "intensity": 2, "topics": ["government", "coalition", "left_right_polarization"], "context": ["right_wing_government"]},
    "ממשלת ימין": {"sentiment": 0, "intensity": 2, "topics": ["government", "coalition", "left_right_polarization"], "context": ["right_wing_government"]},
    "ממשלת שמאל": {"sentiment": 0, "intensity": 2, "topics": ["government", "coalition", "left_right_polarization"], "context": ["left_wing_government"]},
    "מדינת הלכה": {"sentiment": -1, "intensity": 3, "topics": ["religion_and_state"], "context": []},
    "דיקטטורה": {"sentiment": -2, "intensity": 4, "topics": ["protest", "judiciary"], "context": []},
    "דמוקרטיה": {"sentiment": 1, "intensity": 2, "topics": ["protest", "judiciary"], "context": []},
    "בגידה": {"sentiment": -3, "intensity": 4, "topics": ["responsibility_blame"], "context": []},
    "מחדל": {"sentiment": -2, "intensity": 3, "topics": ["responsibility_blame"], "context": []},
    "כישלון": {"sentiment": -2, "intensity": 3, "topics": ["responsibility_blame"], "context": []},
    "מושחת": {"sentiment": -2, "intensity": 3, "topics": ["corruption"], "context": []},
    "אחריות": {"sentiment": 0, "intensity": 2, "topics": ["responsibility_blame"], "context": []},
    "ביטחון לפני הכל": {"sentiment": 1, "intensity": 3, "topics": ["security"], "context": ["security"]},
    "אין ביטחון": {"sentiment": -2, "intensity": 3, "topics": ["security", "responsibility_blame"], "context": ["security"]},
    "להחזיר את החטופים": {"sentiment": 0, "intensity": 3, "topics": ["hostages"], "context": ["hostages"]},
    "עסקה עכשיו": {"sentiment": 0, "intensity": 3, "topics": ["hostages", "protest"], "context": ["hostages"]},
    "התיישבות": {"sentiment": 0, "intensity": 1, "topics": ["settlements_sovereignty"], "context": ["settlements"]},
    "ריבונות עכשיו": {"sentiment": 0, "intensity": 3, "topics": ["settlements_sovereignty"], "context": ["sovereignty"]},
    "שתי מדינות": {"sentiment": 0, "intensity": 2, "topics": ["palestinian_issue"], "context": ["two_state_solution"]},
    "סיפוח עכשיו": {"sentiment": 0, "intensity": 3, "topics": ["settlements_sovereignty"], "context": ["annexation"]},
    "go home": {"sentiment": -2, "intensity": 3, "topics": ["protest", "leadership"], "context": []},
    "elections now": {"sentiment": -1, "intensity": 4, "topics": ["elections", "protest"], "context": []},
}


def clean_text(text: str) -> str:
    """Normalize text while preserving Hebrew and English political terms."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = text.replace("״", '"').replace("׳", "'").replace("`", "'")
    text = re.sub(r"[^0-9a-zA-Z\u0590-\u05FF\"'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_term(term: str) -> str:
    return clean_text(term)


def _contains_term(cleaned_text: str, term: str) -> bool:
    normalized = _normalize_term(term)
    if not normalized:
        return False
    return f" {normalized} " in f" {cleaned_text} "


def _matched_keys(cleaned_text: str, dictionary: dict[str, list[str]]) -> list[str]:
    matches = []
    for key, aliases in dictionary.items():
        if any(_contains_term(cleaned_text, alias) for alias in aliases):
            matches.append(key)
    return matches


def detect_entities(cleaned_text: str) -> tuple[list[str], str]:
    entity_counts = {}
    for entity, aliases in ENTITY_ALIASES.items():
        count = sum(1 for alias in aliases if _contains_term(cleaned_text, alias))
        if count:
            entity_counts[entity] = count
    if not entity_counts:
        return [], ""
    mentioned = sorted(entity_counts)
    main_entity = max(entity_counts, key=entity_counts.get)
    return mentioned, main_entity


def detect_political_phrases(cleaned_text: str) -> list[str]:
    return [phrase for phrase in POLITICAL_PHRASES if _contains_term(cleaned_text, phrase)]


def detect_topics(cleaned_text: str, matched_phrases: list[str]) -> list[str]:
    topics = set(_matched_keys(cleaned_text, TOPIC_KEYWORDS))
    for phrase in matched_phrases:
        topics.update(POLITICAL_PHRASES[phrase]["topics"])
    return sorted(topics)


def detect_context_terms(cleaned_text: str, matched_phrases: list[str]) -> list[str]:
    context = set(_matched_keys(cleaned_text, IDEOLOGY_CONTEXT_TERMS))
    for phrase in matched_phrases:
        context.update(POLITICAL_PHRASES[phrase].get("context", []))
    return sorted(context)


def score_sentiment(cleaned_text: str, matched_phrases: list[str]) -> int:
    score = 0
    tokens = cleaned_text.split()
    token_pairs = {" ".join(tokens[i : i + 2]) for i in range(max(len(tokens) - 1, 0))}

    for token in tokens:
        if token in POSITIVE_WORDS:
            score += 1
        if token in NEGATIVE_WORDS:
            score -= 1

    for phrase in token_pairs:
        if phrase in POSITIVE_WORDS:
            score += 1
        if phrase in NEGATIVE_WORDS:
            score -= 1

    for phrase in matched_phrases:
        score += int(POLITICAL_PHRASES[phrase]["sentiment"])
    return score


def calculate_intensity(cleaned_text: str, sentiment_score: int, matched_phrases: list[str]) -> int:
    phrase_intensity = sum(int(POLITICAL_PHRASES[phrase]["intensity"]) for phrase in matched_phrases)
    emphasis = min(cleaned_text.count("!") + cleaned_text.count("?"), 3)
    return int(abs(sentiment_score) + phrase_intensity + emphasis)


def label_sentiment(score: int) -> str:
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def target_sentiment(main_entity: str, sentiment_label: str) -> str:
    if not main_entity:
        return "unknown"
    return sentiment_label if sentiment_label != "neutral" else "neutral"


def join_values(values: list[str]) -> str:
    return "|".join(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rule-based political sentiment analysis.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Input CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output CSV path.")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    print(f"Input path: {input_path.relative_to(PROJECT_ROOT).as_posix() if input_path.is_relative_to(PROJECT_ROOT) else input_path}")
    print(f"Output path: {output_path.relative_to(PROJECT_ROOT).as_posix() if output_path.is_relative_to(PROJECT_ROOT) else output_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    required_columns = ["timestamp", "text", "candidate", "likes", "reposts", "comments"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "text"]).copy()

    for column in ["likes", "reposts", "comments"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

    df["clean_text"] = df["text"].apply(clean_text)

    analysis_rows = []
    for _, row in df.iterrows():
        cleaned = row["clean_text"]
        mentioned_entities, main_entity = detect_entities(cleaned)
        matched_phrases = detect_political_phrases(cleaned)
        score = score_sentiment(cleaned, matched_phrases)
        label = label_sentiment(score)
        political_blocs = sorted(set(_matched_keys(cleaned, BLOC_TERMS) + detect_context_terms(cleaned, matched_phrases)))
        analysis_rows.append(
            {
                "mentioned_entities": join_values(mentioned_entities),
                "main_entity": main_entity,
                "mentioned_parties": join_values(_matched_keys(cleaned, PARTY_TERMS)),
                "political_bloc_terms": join_values(political_blocs),
                "political_topics": join_values(detect_topics(cleaned, matched_phrases)),
                "matched_political_phrases": join_values(matched_phrases),
                "sentiment_score": score,
                "sentiment_label": label,
                "target_sentiment": target_sentiment(main_entity, label),
                "intensity_score": calculate_intensity(cleaned, score, matched_phrases),
            }
        )

    analysis_df = pd.DataFrame(analysis_rows)
    df = pd.concat([df.reset_index(drop=True), analysis_df], axis=1)

    output_columns = [
        "timestamp", "text", "candidate", "likes", "reposts", "comments", "clean_text",
        "mentioned_entities", "main_entity", "mentioned_parties", "political_bloc_terms",
        "political_topics", "matched_political_phrases", "sentiment_score", "sentiment_label",
        "target_sentiment", "intensity_score",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[output_columns].to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved sentiment output to {output_path.relative_to(PROJECT_ROOT).as_posix() if output_path.is_relative_to(PROJECT_ROOT) else output_path}")
    print(f"Rows processed: {len(df)}")
    print(df["sentiment_label"].value_counts().to_string())


if __name__ == "__main__":
    main()

