"""
Tests for pipeline/feature_engineering.py — the parts that don't require
loading transformer models. Aspect-name backwards compat, the urgency
heuristic, sentiment, and the stopword sets get exercised here. The
clustering / merge layer is already covered in tests/test_clustering.py.

Importing pipeline.feature_engineering loads spaCy + three transformer
models at module level, which makes this file noticeably slower than the
others. Worth it: these are the closest-to-business-logic functions in the
pipeline and they were previously untested.
"""
import os
os.environ.setdefault("FORCE_CPU", "1")  # avoid GPU init in CI / cold runs

import pytest

from pipeline import feature_engineering as fe


# ---------------------------------------------------------------------------
# aspect_names — backwards compatibility (mirrors summarizer._aspect_names)
# ---------------------------------------------------------------------------

def test_aspect_names_legacy_string_format():
    """Phase II reviews stored aspects as list[str]."""
    review = {"aspects": ["delivery", "package", "driver"]}
    assert fe.aspect_names(review) == ["delivery", "package", "driver"]


def test_aspect_names_phase3_dict_format():
    """Phase III reviews store aspects as list[{aspect, polarity, confidence}]."""
    review = {"aspects": [
        {"aspect": "delivery", "polarity": -0.8, "confidence": 0.95},
        {"aspect": "package", "polarity": 0.3, "confidence": 0.7},
    ]}
    assert fe.aspect_names(review) == ["delivery", "package"]


def test_aspect_names_mixed_format_does_not_crash():
    """An accidental mix (legacy + Phase III rows in the same list) shouldn't blow up."""
    review = {"aspects": ["legacy", {"aspect": "modern", "polarity": 0.0, "confidence": 1.0}]}
    assert fe.aspect_names(review) == ["legacy", "modern"]


def test_aspect_names_empty_when_missing_or_none():
    assert fe.aspect_names({}) == []
    assert fe.aspect_names({"aspects": None}) == []
    assert fe.aspect_names({"aspects": []}) == []


# ---------------------------------------------------------------------------
# sentiment_features — TextBlob wrapper
# ---------------------------------------------------------------------------

def test_sentiment_features_returns_none_for_empty_text():
    """Empty / missing text gets explicit None, not 0.0 — caller can distinguish."""
    assert fe.sentiment_features("") == {"polarity": None, "subjectivity": None}
    assert fe.sentiment_features(None) == {"polarity": None, "subjectivity": None}


def test_sentiment_features_polarity_signs_align_with_sentiment():
    """Smoke check: 'great' is positive, 'terrible' is negative."""
    pos = fe.sentiment_features("This is great, I love it.")
    neg = fe.sentiment_features("This is terrible, I hate it.")
    assert pos["polarity"] > 0
    assert neg["polarity"] < 0


def test_sentiment_features_rounded_to_4_decimals():
    """Values are rounded — keeps DB / report output stable across builds."""
    out = fe.sentiment_features("good app")
    assert isinstance(out["polarity"], float)
    # 4-decimal rounding → no more than 4 fractional digits
    assert len(str(out["polarity"]).split(".")[-1]) <= 4


# ---------------------------------------------------------------------------
# urgency_score — interpretable heuristic
# ---------------------------------------------------------------------------

def _urgency_review(body, rating=3, subjectivity=0.5, aspects=()):
    return {
        "body": body,
        "rating": rating,
        "subjectivity": subjectivity,
        "aspects": list(aspects),
    }


def test_urgency_zero_for_empty_body():
    """Without a body, no signal can fire."""
    assert fe.urgency_score(_urgency_review("")) == 0.0
    assert fe.urgency_score({"body": None}) == 0.0


def test_urgency_bug_keyword_adds_weight():
    """A bug keyword alone bumps urgency by 0.4."""
    no_bug = fe.urgency_score(_urgency_review("the colors are nice"))
    with_bug = fe.urgency_score(_urgency_review("the app keeps crashing on startup"))
    assert with_bug > no_bug
    assert with_bug >= 0.4


def test_urgency_low_rating_adds_weight():
    """Rating ≤ 2 contributes 0.3."""
    high = fe.urgency_score(_urgency_review("the app crashes", rating=4))
    low = fe.urgency_score(_urgency_review("the app crashes", rating=1))
    assert low > high
    assert (low - high) >= 0.3 - 1e-9


def test_urgency_low_subjectivity_adds_weight():
    """Below 0.5 subjectivity (objective) gets 0.15."""
    subjective = fe.urgency_score(_urgency_review("crash bug here", subjectivity=0.9))
    objective = fe.urgency_score(_urgency_review("crash bug here", subjectivity=0.1))
    assert objective > subjective


def test_urgency_capped_at_one():
    """Heuristic clamps to [0, 1] even when every signal fires."""
    score = fe.urgency_score(_urgency_review(
        body="crash bug not working can't load black screen " * 5,
        rating=1,
        subjectivity=0.1,
        aspects=("login", "payment", "checkout"),
    ))
    assert 0.0 <= score <= 1.0


def test_urgency_works_on_phase3_aspect_format():
    """aspects can be list[dict] post-Phase-III; the score's len() still works."""
    review = {
        "body": "the app keeps crashing",
        "rating": 1,
        "subjectivity": 0.2,
        "aspects": [
            {"aspect": "login", "polarity": -0.8, "confidence": 0.9},
            {"aspect": "checkout", "polarity": -0.6, "confidence": 0.8},
        ],
    }
    score = fe.urgency_score(review)
    assert score > 0  # didn't crash, produced a number


# ---------------------------------------------------------------------------
# Stopword sets — assert specific entries we depend on
# ---------------------------------------------------------------------------

def test_stopwords_contain_filler_pronouns():
    """STOPWORDS filters out pronoun-shaped noun-chunk roots."""
    for w in ("it", "this", "they", "thing"):
        assert w in fe.STOPWORDS


def test_ner_stopwords_filter_phase4_additions():
    """Phase IV B2: OTP / Newest / Tablet / Tablets must be in NER stopwords."""
    for w in ("otp", "newest", "tablet", "tablets"):
        assert w in fe.NER_STOPWORDS


def test_ner_stopwords_filter_ui_button_labels():
    """ALL-CAPS button labels often get mistagged as ORG; they're filtered."""
    for w in ("cancel", "buy", "pay", "submit"):
        assert w in fe.NER_STOPWORDS


def test_bug_keywords_cover_common_terms():
    """BUG_KEYWORDS drives a chunk of the urgency score; smoke check core terms."""
    for w in ("crash", "bug", "broken", "error", "freeze"):
        assert w in fe.BUG_KEYWORDS


# ---------------------------------------------------------------------------
# spacy_features — exercises STOPWORDS / NER_STOPWORDS / brand_stopwords
# (Lightweight: spaCy small model, already loaded at module level.)
# ---------------------------------------------------------------------------

def test_spacy_features_returns_empty_for_empty_text():
    out = fe.spacy_features("")
    assert out == {"aspects": [], "entities": []}


def test_spacy_features_filters_stopwords():
    """Aspects in STOPWORDS shouldn't appear in the output."""
    # 'thing' and 'time' are in STOPWORDS; 'delivery' is not.
    out = fe.spacy_features("the delivery thing was on time")
    assert "thing" not in out["aspects"]
    assert "time" not in out["aspects"]


def test_spacy_features_filters_brand_stopwords():
    """brand_stopwords like 'amazon' should not surface as aspects."""
    out = fe.spacy_features(
        "amazon delivery is great",
        brand_stopwords=frozenset(["amazon"]),
    )
    assert "amazon" not in out["aspects"]


def test_spacy_features_dedupes_aspects():
    """Same aspect mentioned twice in one review counts once."""
    out = fe.spacy_features("the delivery was bad and the delivery was late")
    delivery_count = sum(1 for a in out["aspects"] if a == "delivery")
    assert delivery_count <= 1
