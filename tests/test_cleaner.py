"""
Tests for pipeline/cleaner.py.

Focus: the regression that motivated Phase IV — review_id, thumbs_up, and
app_version must survive clean_review so they're available in --from-csv mode
and in feature engineering. Plus the edge cases on each cleaning function.
"""
from pipeline.cleaner import (
    clean_review,
    clean_reviews,
    clean_text,
    clean_rating,
    clean_date,
    clean_thumbs_up,
)


# ---------------------------------------------------------------------------
# clean_review — the Phase IV regression: new fields must round-trip
# ---------------------------------------------------------------------------

def test_clean_review_carries_review_id():
    """review_id must survive cleaning (was being dropped before Phase IV)."""
    r = clean_review({
        "review_id": "abc-123",
        "reviewer_name": "alice",
        "rating": 4,
        "body": "ok",
        "date": "2026-01-01T00:00:00Z",
    })
    assert r["review_id"] == "abc-123"


def test_clean_review_carries_thumbs_up():
    """thumbs_up must survive cleaning."""
    r = clean_review({
        "reviewer_name": "alice",
        "rating": 4,
        "body": "ok",
        "date": "2026-01-01T00:00:00Z",
        "thumbs_up": 12,
    })
    assert r["thumbs_up"] == 12


def test_clean_review_carries_app_version():
    """app_version must survive cleaning — Phase IV time-series depends on it."""
    r = clean_review({
        "reviewer_name": "alice",
        "rating": 4,
        "body": "ok",
        "date": "2026-01-01T00:00:00Z",
        "app_version": "17.4.0",
    })
    assert r["app_version"] == "17.4.0"


def test_clean_review_handles_missing_new_fields():
    """Older inputs without the new fields should not crash."""
    r = clean_review({
        "reviewer_name": "alice",
        "rating": 4,
        "body": "ok",
        "date": "2026-01-01T00:00:00Z",
    })
    assert r["review_id"] is None
    assert r["thumbs_up"] == 0
    assert r["app_version"] is None


def test_clean_review_returns_none_on_uncatchable_failure():
    """Wholly-malformed inputs must not bring down the whole pipeline."""
    # A non-dict input would explode inside clean_review's .get calls.
    assert clean_review(None) is None


# ---------------------------------------------------------------------------
# clean_thumbs_up
# ---------------------------------------------------------------------------

def test_clean_thumbs_up_passes_through_int():
    assert clean_thumbs_up(42) == 42


def test_clean_thumbs_up_coerces_string():
    """CSV reads come back as strings; coerce them."""
    assert clean_thumbs_up("17") == 17


def test_clean_thumbs_up_defaults_zero_for_missing():
    assert clean_thumbs_up(None) == 0
    assert clean_thumbs_up("") == 0


def test_clean_thumbs_up_clamps_negative_to_zero():
    """Garbage in (negative count is meaningless) → 0, not propagated."""
    assert clean_thumbs_up(-5) == 0


def test_clean_thumbs_up_falls_back_on_garbage():
    assert clean_thumbs_up("not a number") == 0


# ---------------------------------------------------------------------------
# clean_rating
# ---------------------------------------------------------------------------

def test_clean_rating_in_range():
    for r in (1, 2, 3, 4, 5):
        assert clean_rating(r) == r


def test_clean_rating_out_of_range_returns_none():
    """Star ratings outside 1-5 are nonsense — must drop, not clamp."""
    assert clean_rating(0) is None
    assert clean_rating(6) is None
    assert clean_rating(-1) is None


def test_clean_rating_handles_garbage():
    assert clean_rating("not a number") is None
    assert clean_rating(None) is None


# ---------------------------------------------------------------------------
# clean_date
# ---------------------------------------------------------------------------

def test_clean_date_normalizes_iso_with_z():
    """Trustpilot-style trailing Z must parse cleanly."""
    assert clean_date("2026-04-09T02:25:46.000Z") == "2026-04-09"


def test_clean_date_normalizes_naive_iso():
    assert clean_date("2026-04-09T02:25:46") == "2026-04-09"


def test_clean_date_returns_none_on_garbage():
    assert clean_date("not a date") is None
    assert clean_date(None) is None
    assert clean_date("") is None


# ---------------------------------------------------------------------------
# clean_reviews bulk path
# ---------------------------------------------------------------------------

def test_clean_reviews_drops_failures_and_keeps_others():
    """Bad rows return None and are silently dropped — not allowed to take the run down."""
    reviews = [
        {"reviewer_name": "alice", "rating": 4, "body": "good", "date": "2026-01-01T00:00:00Z"},
        None,  # would crash inside clean_review; should be dropped
        {"reviewer_name": "bob", "rating": 5, "body": "great", "date": "2026-01-02T00:00:00Z"},
    ]
    out = clean_reviews(reviews)
    assert len(out) == 2
    assert out[0]["reviewer_name"] == "alice"
    assert out[1]["reviewer_name"] == "bob"
