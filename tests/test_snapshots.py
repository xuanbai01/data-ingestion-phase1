"""
Tests for the issue_snapshots persistence layer in database/db.py.

Confirms the four behaviors the summarizer's Run Delta section depends on:
- save → load round-trip preserves every field, including the centroid blob
- a re-save with the same (app_slug, run_id) replaces, doesn't double-up
- recent_run_ids returns runs newest-first
- load_prior_run_snapshots picks the most recent run strictly before the cursor
"""
from database import db


def _snap(cluster_id, aspects, centroid=None, count=50, is_issue=True):
    return {
        "cluster_id":     cluster_id,
        "cluster_label":  ", ".join(aspects[:3]),
        "aspect_set":     list(aspects),
        "centroid":       centroid,
        "review_count":   count,
        "avg_rating":     1.5,
        "avg_polarity":   -0.4,
        "avg_urgency":    0.6,
        "priority_score": 0.7,
        "is_issue":       is_issue,
    }


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_save_and_load_round_trip(temp_db):
    snaps = [
        _snap(0, ["delivery", "package", "driver"], centroid=[0.1, 0.2, 0.3], count=120),
        _snap(1, ["login", "password", "account"], centroid=[0.4, 0.5, 0.6], count=80, is_issue=False),
    ]
    db.save_issue_snapshots("amazon", "20260427T120000Z", snaps)

    loaded = db.load_snapshots("amazon", ["20260427T120000Z"])
    assert "20260427T120000Z" in loaded
    rows = loaded["20260427T120000Z"]
    assert len(rows) == 2

    # Sorted ASC by cluster_id per the load helper, so cluster 0 first
    by_cid = {r["cluster_id"]: r for r in rows}
    assert by_cid[0]["aspect_set"] == ["delivery", "package", "driver"]
    assert by_cid[0]["review_count"] == 120
    assert by_cid[0]["is_issue"] is True
    assert by_cid[1]["is_issue"] is False
    # Centroid round-trips through float32; allow small precision loss.
    for got, want in zip(by_cid[0]["centroid"], [0.1, 0.2, 0.3]):
        assert abs(got - want) < 1e-6


def test_save_with_no_centroid_is_safe(temp_db):
    """Cluster with no embeddings gets a None centroid → must persist as NULL, not crash."""
    snaps = [_snap(0, ["a", "b"], centroid=None)]
    db.save_issue_snapshots("amazon", "20260427T120000Z", snaps)

    loaded = db.load_snapshots("amazon", ["20260427T120000Z"])
    assert loaded["20260427T120000Z"][0]["centroid"] is None


def test_resave_same_run_id_replaces(temp_db):
    """A re-run with the same run_id should overwrite, not double-count."""
    db.save_issue_snapshots("amazon", "run1", [_snap(0, ["a"], count=100)])
    db.save_issue_snapshots("amazon", "run1", [_snap(0, ["a"], count=200)])

    rows = db.load_snapshots("amazon", ["run1"])["run1"]
    assert len(rows) == 1
    assert rows[0]["review_count"] == 200


def test_save_empty_snapshots_is_noop(temp_db):
    """Edge: a run with zero clusters shouldn't write a row or raise."""
    db.save_issue_snapshots("amazon", "run1", [])
    assert db.recent_run_ids("amazon") == []


# ---------------------------------------------------------------------------
# recent_run_ids — ordering
# ---------------------------------------------------------------------------

def test_recent_run_ids_orders_newest_first(temp_db):
    db.save_issue_snapshots("amazon", "20260101T000000Z", [_snap(0, ["a"])])
    db.save_issue_snapshots("amazon", "20260301T000000Z", [_snap(0, ["a"])])
    db.save_issue_snapshots("amazon", "20260201T000000Z", [_snap(0, ["a"])])

    assert db.recent_run_ids("amazon") == [
        "20260301T000000Z",
        "20260201T000000Z",
        "20260101T000000Z",
    ]


def test_recent_run_ids_respects_app_slug(temp_db):
    """One app's runs must not leak into another's history."""
    db.save_issue_snapshots("amazon", "run1", [_snap(0, ["a"])])
    db.save_issue_snapshots("ebay", "run2", [_snap(0, ["a"])])

    assert db.recent_run_ids("amazon") == ["run1"]
    assert db.recent_run_ids("ebay") == ["run2"]


def test_recent_run_ids_honors_limit(temp_db):
    for i in range(5):
        db.save_issue_snapshots("amazon", f"run{i}", [_snap(0, ["a"])])
    assert len(db.recent_run_ids("amazon", limit=3)) == 3


# ---------------------------------------------------------------------------
# load_prior_run_snapshots — strict-before semantics
# ---------------------------------------------------------------------------

def test_prior_run_finds_immediate_predecessor(temp_db):
    """Given runs r1 < r2 < r3, asking for prior of r3 returns r2 (not r1)."""
    db.save_issue_snapshots("amazon", "run_a", [_snap(0, ["one"])])
    db.save_issue_snapshots("amazon", "run_b", [_snap(0, ["two"])])
    db.save_issue_snapshots("amazon", "run_c", [_snap(0, ["three"])])

    prior_run, snapshots = db.load_prior_run_snapshots("amazon", "run_c")
    assert prior_run == "run_b"
    assert snapshots[0]["aspect_set"] == ["two"]


def test_prior_run_is_strictly_before_cursor(temp_db):
    """Asking for prior of run_b returns run_a, not run_b itself."""
    db.save_issue_snapshots("amazon", "run_a", [_snap(0, ["one"])])
    db.save_issue_snapshots("amazon", "run_b", [_snap(0, ["two"])])

    prior_run, _ = db.load_prior_run_snapshots("amazon", "run_b")
    assert prior_run == "run_a"


def test_prior_run_returns_empty_on_first_run(temp_db):
    """No prior runs in DB → (None, []), not a crash."""
    db.save_issue_snapshots("amazon", "run_a", [_snap(0, ["one"])])
    prior_run, snapshots = db.load_prior_run_snapshots("amazon", "run_a")
    assert prior_run is None
    assert snapshots == []


def test_prior_run_isolated_per_app(temp_db):
    """Asking for prior of amazon's first run, with ebay runs in DB, returns None."""
    db.save_issue_snapshots("ebay", "run_old", [_snap(0, ["a"])])
    db.save_issue_snapshots("amazon", "run_new", [_snap(0, ["b"])])

    prior_run, snapshots = db.load_prior_run_snapshots("amazon", "run_new")
    assert prior_run is None
    assert snapshots == []
