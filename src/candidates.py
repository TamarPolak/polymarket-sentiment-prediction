"""Central candidate configuration for the Israeli PM Polymarket project.

This module is intentionally data-only plus small lookup helpers so the rest of
pipeline can move from a 2-candidate MVP to all active market candidates without
hardcoding candidate names in multiple scripts.
"""

from __future__ import annotations

from typing import Optional


CANDIDATES = [
    {
        "canonical_name": "Gadi Eisenkot",
        "hebrew_aliases": ["גדי איזנקוט", "איזנקוט", "גדי אייזנקוט", "אייזנקוט"],
        "english_aliases": ["Gadi Eisenkot", "Eisenkot", "Gadi Eizenkot", "Eizenkot"],
        "active": True,
        "priority": "top",
    },
    {
        "canonical_name": "Benjamin Netanyahu",
        "hebrew_aliases": ["בנימין נתניהו", "נתניהו", "ביבי"],
        "english_aliases": ["Benjamin Netanyahu", "Netanyahu", "Bibi", "Binyamin Netanyahu"],
        "active": True,
        "priority": "top",
    },
    {
        "canonical_name": "Naftali Bennett",
        "hebrew_aliases": ["נפתלי בנט", "בנט"],
        "english_aliases": ["Naftali Bennett", "Bennett"],
        "active": True,
        "priority": "top",
    },
    {
        "canonical_name": "Avigdor Lieberman",
        "hebrew_aliases": ["אביגדור ליברמן", "ליברמן"],
        "english_aliases": ["Avigdor Lieberman", "Lieberman", "Liberman"],
        "active": True,
        "priority": "top",
    },
    {
        "canonical_name": "Itamar Ben Gvir",
        "hebrew_aliases": ["איתמר בן גביר", "בן גביר", "בן-גביר"],
        "english_aliases": ["Itamar Ben Gvir", "Ben Gvir", "Ben-Gvir"],
        "active": True,
        "priority": "mid",
    },
    {
        "canonical_name": "Yariv Levin",
        "hebrew_aliases": ["יריב לוין", "לוין"],
        "english_aliases": ["Yariv Levin", "Levin"],
        "active": True,
        "priority": "mid",
    },
    {
        "canonical_name": "Yossi Cohen",
        "hebrew_aliases": ["יוסי כהן"],
        "english_aliases": ["Yossi Cohen"],
        "active": True,
        "priority": "mid",
    },
    {
        "canonical_name": "Israel Katz",
        "hebrew_aliases": ["ישראל כץ", "כץ"],
        "english_aliases": ["Israel Katz", "Katz"],
        "active": True,
        "priority": "mid",
    },
    {
        "canonical_name": "Gideon Sa'ar",
        "hebrew_aliases": ["גדעון סער", "סער"],
        "english_aliases": ["Gideon Sa'ar", "Gideon Saar", "Sa'ar", "Saar"],
        "active": True,
        "priority": "mid",
    },
    {
        "canonical_name": "Yair Lapid",
        "hebrew_aliases": ["יאיר לפיד", "לפיד"],
        "english_aliases": ["Yair Lapid", "Lapid"],
        "active": True,
        "priority": "top",
    },
    {
        "canonical_name": "Benny Gantz",
        "hebrew_aliases": ["בני גנץ", "גנץ"],
        "english_aliases": ["Benny Gantz", "Gantz"],
        "active": True,
        "priority": "top",
    },
    {
        "canonical_name": "Yair Golan",
        "hebrew_aliases": ["יאיר גולן", "גולן"],
        "english_aliases": ["Yair Golan"],
        "active": True,
        "priority": "mid",
    },
    {
        "canonical_name": "Nir Barkat",
        "hebrew_aliases": ["ניר ברקת", "ברקת"],
        "english_aliases": ["Nir Barkat", "Barkat"],
        "active": True,
        "priority": "mid",
    },
    {
        "canonical_name": "Gilad Erdan",
        "hebrew_aliases": ["גלעד ארדן", "ארדן"],
        "english_aliases": ["Gilad Erdan", "Erdan"],
        "active": True,
        "priority": "long_tail",
    },
    {
        "canonical_name": "Ayelet Shaked",
        "hebrew_aliases": ["איילת שקד", "שקד"],
        "english_aliases": ["Ayelet Shaked", "Shaked"],
        "active": True,
        "priority": "mid",
    },
    {
        "canonical_name": "Amir Ohana",
        "hebrew_aliases": ["אמיר אוחנה", "אוחנה"],
        "english_aliases": ["Amir Ohana", "Ohana"],
        "active": True,
        "priority": "mid",
    },
    {
        "canonical_name": "Moshe Feiglin",
        "hebrew_aliases": ["משה פייגלין", "פייגלין"],
        "english_aliases": ["Moshe Feiglin", "Feiglin"],
        "active": True,
        "priority": "long_tail",
    },
    {
        "canonical_name": "Yoaz Hendel",
        "hebrew_aliases": ["יועז הנדל", "הנדל"],
        "english_aliases": ["Yoaz Hendel", "Yoaz Handel", "Hendel", "Handel"],
        "active": True,
        "priority": "long_tail",
    },
]


def _normalize_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("'", "")
        .replace("’", "")
        .replace("`", "")
        .replace("\"", "")
        .replace("-", " ")
    )


def _candidate_aliases(candidate: dict) -> list[str]:
    return [
        candidate["canonical_name"],
        *candidate.get("hebrew_aliases", []),
        *candidate.get("english_aliases", []),
    ]


def get_active_candidates() -> list[dict]:
    """Return all candidates currently marked as active."""
    return [candidate for candidate in CANDIDATES if candidate.get("active", False)]


def get_candidate_names() -> list[str]:
    """Return canonical names for active candidates."""
    return [candidate["canonical_name"] for candidate in get_active_candidates()]


def get_candidate_by_name(name: str) -> Optional[dict]:
    """Find a candidate by canonical name or alias.

    Matching is case-insensitive for English aliases and tolerant of apostrophe
    variants such as Sa'ar/Saar.
    """
    if not name:
        return None

    normalized = _normalize_name(name)
    for candidate in CANDIDATES:
        aliases = {_normalize_name(alias) for alias in _candidate_aliases(candidate)}
        if normalized in aliases:
            return candidate

    return None
