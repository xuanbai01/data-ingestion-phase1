"""
Tests for pipeline/issue_tracking.py — the cross-run matching layer behind the
Run Delta section. Pure functions on synthetic snapshots; no DB or model loads.
"""
import math

from pipeline.issue_tracking import (
    jaccard,
    cosine,
    match_issues,
    find_resolved,
    classify_delta,
    render_sparkline,
    bucket_dates,
    JACCARD_THRESHOLD,
    CENTROID_THRESHOLD,
)


def _snap(cluster_id, aspects, centroid=None, count=50):
    return {
        "cluster_id": cluster_id,
        "aspect_set": list(aspects),
        "centroid": centroid,
        "review_count": count,
    }


# ---------------------------------------------------------------------------
# jaccard
# ---------------------------------------------------------------------------

def test_jaccard_full_overlap_is_one():
    assert jaccard(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_jaccard_no_overlap_is_zero():
    assert jaccard(["a", "b"], ["c", "d"]) == 0.0


def test_jaccard_partial_overlap():
    # |∩| = 2 (a, b), |∪| = 4 (a, b, c, d) → 0.5
    assert jaccard(["a", "b", "c"], ["a", "b", "d"]) == 0.5


def test_jaccard_empty_sets_safe():
    """Both empty → 0.0, never NaN, never crashes."""
    assert jaccard([], []) == 0.0
    assert jaccard(None, None) == 0.0
    assert jaccard(["a"], None) == 0.0


# ---------------------------------------------------------------------------
# cosine
# ---------------------------------------------------------------------------

def test_cosine_identical_vectors_is_one():
    # cosine() coerces to float32 to match the embedder's native precision,
    # so identical inputs come back ~1.0 with tiny rounding error.
    assert abs(cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) - 1.0) < 1e-6


def test_cosine_orthogonal_is_zero():
    assert abs(cosine([1.0, 0.0], [0.0, 1.0])) < 1e-6


def test_cosine_handles_none_or_zero_vectors():
    """Defensive: a missing or zero-norm centroid must not raise."""
    assert cosine(None, [1.0, 0.0]) == 0.0
    assert cosine([1.0, 0.0], None) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# match_issues — primary path (Jaccard)
# ---------------------------------------------------------------------------

def test_match_jaccard_picks_best_match():
    """The current snapshot should pair with its closest aspect-set neighbor."""
    cur = [_snap(0, ["delivery", "package", "driver", "late"])]
    prior = [
        _snap(1, ["login", "password", "account"]),                   # disjoint
        _snap(2, ["delivery", "package", "driver", "slow"]),          # 3/5 = 0.6
        _snap(3, ["delivery", "shipping"]),                            # 1/5 = 0.2
    ]
    matches = match_issues(cur, prior)
    assert matches[0]["method"] == "jaccard"
    assert matches[0]["prior"]["cluster_id"] == 2


def test_match_below_jaccard_threshold_uses_centroid_fallback():
    """When aspect words drift but the embedding stays close, centroid wins."""
    cur = [_snap(0, ["x", "y"], centroid=[1.0, 0.0])]
    prior = [_snap(7, ["p", "q"], centroid=[0.95, 0.05])]
    matches = match_issues(cur, prior)
    assert matches[0]["method"] == "centroid"
    assert matches[0]["prior"]["cluster_id"] == 7


def test_match_below_both_thresholds_returns_none():
    """Aspect-disjoint AND embedding-disjoint → no match → 'new' on next step."""
    cur = [_snap(0, ["x", "y"], centroid=[1.0, 0.0])]
    prior = [_snap(7, ["p", "q"], centroid=[0.0, 1.0])]
    matches = match_issues(cur, prior)
    assert matches[0]["prior"] is None
    assert matches[0]["method"] is None


def test_match_against_empty_prior_returns_all_unmatched():
    """First-run case: every current snapshot returns prior=None."""
    cur = [_snap(0, ["a", "b"]), _snap(1, ["c", "d"])]
    matches = match_issues(cur, [])
    assert all(m["prior"] is None for m in matches)


def test_match_allows_many_to_one_for_cluster_splits():
    """Two current snapshots can match the same prior — represents a cluster split."""
    cur = [
        _snap(0, ["delivery", "package", "driver", "late"]),
        _snap(1, ["delivery", "package", "driver", "stolen"]),
    ]
    prior = [_snap(7, ["delivery", "package", "driver", "issue"])]
    matches = match_issues(cur, prior)
    assert matches[0]["prior"]["cluster_id"] == 7
    assert matches[1]["prior"]["cluster_id"] == 7


# ---------------------------------------------------------------------------
# find_resolved
# ---------------------------------------------------------------------------

def test_find_resolved_returns_unmatched_priors():
    cur = [_snap(0, ["a", "b", "c", "d"])]
    prior = [
        _snap(1, ["a", "b", "c", "d"]),  # will be matched
        _snap(2, ["x", "y", "z", "w"]),  # not matched → resolved
    ]
    matches = match_issues(cur, prior)
    resolved = find_resolved(matches, prior)
    assert len(resolved) == 1
    assert resolved[0]["cluster_id"] == 2


def test_find_resolved_empty_when_all_matched():
    cur = [_snap(0, ["a", "b", "c", "d"])]
    prior = [_snap(7, ["a", "b", "c", "d"])]
    matches = match_issues(cur, prior)
    assert find_resolved(matches, prior) == []


# ---------------------------------------------------------------------------
# classify_delta
# ---------------------------------------------------------------------------

def test_classify_delta_escalating():
    cur = _snap(0, ["a"], count=130)
    prior = _snap(1, ["a"], count=100)
    assert classify_delta({"current": cur, "prior": prior}) == "escalating"


def test_classify_delta_improving():
    cur = _snap(0, ["a"], count=70)
    prior = _snap(1, ["a"], count=100)
    assert classify_delta({"current": cur, "prior": prior}) == "improving"


def test_classify_delta_stable_in_band():
    """Within ±20% of prior count → stable, not flagged in delta view."""
    cur = _snap(0, ["a"], count=90)  # 0.9× prior
    prior = _snap(1, ["a"], count=100)
    assert classify_delta({"current": cur, "prior": prior}) == "stable"


def test_classify_delta_new_when_prior_is_none():
    cur = _snap(0, ["a"], count=50)
    assert classify_delta({"current": cur, "prior": None}) == "new"


def test_classify_delta_new_when_prior_has_zero_count():
    """Defensive against a degenerate prior; div-by-zero must not propagate."""
    cur = _snap(0, ["a"], count=50)
    prior = _snap(1, ["a"], count=0)
    assert classify_delta({"current": cur, "prior": prior}) == "new"


# ---------------------------------------------------------------------------
# render_sparkline
# ---------------------------------------------------------------------------

def test_sparkline_empty_input_renders_dash():
    assert render_sparkline([]) == "—"
    assert render_sparkline([None, None, None]) == "—"


def test_sparkline_flat_series_renders_lowest_bar():
    """A zero-span series (all equal) renders as the smallest block, not crashes."""
    assert render_sparkline([5, 5, 5]) == "▁▁▁"


def test_sparkline_bounds_match_extremes():
    """Lowest value → lowest block, highest → highest block."""
    out = render_sparkline([1, 2, 3, 4, 5, 6, 7, 8])
    assert out[0] == "▁"
    assert out[-1] == "█"
    assert len(out) == 8


def test_sparkline_renders_gap_for_none():
    out = render_sparkline([1, None, 3])
    assert out[1] == " "


# ---------------------------------------------------------------------------
# bucket_dates
# ---------------------------------------------------------------------------

def test_bucket_dates_returns_none_on_too_few_dates():
    assert bucket_dates([]) == (None, None, None)
    assert bucket_dates(["2026-01-01"]) == (None, None, None)


def test_bucket_dates_returns_none_when_all_same_day():
    """A single-day cluster has no time dimension to plot."""
    out = bucket_dates(["2026-01-01"] * 5)
    assert out == (None, None, None)


def test_bucket_dates_distributes_across_buckets():
    """Counts should sum to the input size and span the requested buckets."""
    dates = ["2026-01-01", "2026-01-15", "2026-02-01", "2026-02-20",
             "2026-03-10", "2026-03-30"]
    counts, earliest, latest = bucket_dates(dates, n_buckets=6)
    assert sum(counts) == len(dates)
    assert len(counts) == 6
    assert earliest == "2026-01-01"
    assert latest == "2026-03-30"


def test_bucket_dates_ignores_malformed_dates():
    """Bad date strings are silently dropped, not crashes."""
    dates = ["2026-01-01", "not-a-date", "2026-03-30"]
    counts, _, _ = bucket_dates(dates, n_buckets=4)
    assert sum(counts) == 2  # malformed entry ignored
